#!/usr/bin/env python
"""LightRAG 统一适配 driver。
- LLM→U_LLM_ENDPOINT（自定义 async func 直传 extra_body 关 thinking）、embedding→U_EMBED_ENDPOINT
- 吃统一 chunks（chunk_token_size 设大不二次切，每个统一 chunk = 一个 LightRAG chunk）
- mix 模式检索 top-5，输出统一 predictions.jsonl

跑（LightRAG/.venv）:
  LightRAG/.venv/bin/python run_lightrag.py --dataset 2wikimultihopqa --fresh [--limit_q N]
"""
import os
import sys
import json
import time
import shutil
import asyncio
import argparse
import threading
import numpy as np
from openai import AsyncOpenAI

UNIFIED = os.environ.get("URAG_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LIGHTRAG = os.environ.get("LIGHTRAG_ROOT") or os.path.abspath(os.path.join(UNIFIED, "..", "LightRAG"))
sys.path.insert(0, UNIFIED)
sys.path.insert(0, LIGHTRAG)
import unified_config as U
import token_meter
from lightrag import LightRAG, QueryParam
from lightrag.utils import EmbeddingFunc

_llm = AsyncOpenAI(base_url=U.LLM_ENDPOINT, api_key="EMPTY", timeout=600.0, max_retries=5)
_emb = AsyncOpenAI(base_url=U.EMBED_ENDPOINT, api_key="EMPTY", timeout=600.0, max_retries=5)

# 显式给 LightRAG 的 default_llm_timeout 用（180s 默认是 53% failed 根因）
LLM_TIMEOUT = 600
# 退避序列（秒）：第 1 次失败睡 2s，第 2 次 8s，第 3 次 30s
_BACKOFF = [2, 8, 30]
# 当前 dataset 的 failed-chunk 落盘路径（main 里设）
_FAILED_LOG_PATH = None
_FAILED_LOG_LOCK = threading.Lock()
_METER = token_meter.TokenMeter()        # 旁路 token 记账(进程隔离),与抽取/检索算法无关


def _log_failed_chunk(reason, prompt, system_prompt, kwargs):
    """把一个最终失败的 LLM 调用记到 lightrag_failed_chunks_{dataset}.jsonl。
    绝不抛异常——记账失败也不能连累建图。"""
    if not _FAILED_LOG_PATH:
        return
    rec = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "reason": str(reason)[:500],
        "chunk_id": kwargs.get("chunk_id"),
        "cache_type": kwargs.get("cache_type"),
        "keyword_extraction": kwargs.get("keyword_extraction"),
        "system_prompt_head": (system_prompt or "")[:120],
        "prompt_head": (prompt or "")[:120],
    }
    try:
        with _FAILED_LOG_LOCK:
            with open(_FAILED_LOG_PATH, "a") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"[failed-log-error] {e}", flush=True)


async def llm_func(prompt, system_prompt=None, history_messages=None, keyword_extraction=False, **kwargs):
    msgs = []
    if system_prompt:
        msgs.append({"role": "system", "content": system_prompt})
    msgs += list(history_messages or [])
    msgs.append({"role": "user", "content": prompt})

    last_err = None
    # 手写指数退避重试 3 次（2s/8s/30s）；全部失败也绝不抛异常 → 返回空串占位
    for attempt in range(len(_BACKOFF) + 1):  # attempt 0..3 → 共 4 次尝试
        try:
            r = await _llm.chat.completions.create(
                model=U.LLM_SERVED_NAME, messages=msgs, temperature=0,
                max_tokens=U.GEN_MAX_TOKENS, extra_body=U.LLM_EXTRA_BODY)
            content = r.choices[0].message.content
            _METER.add(getattr(r, "usage", None))   # 旁路记账(当前 phase),容错;不改下面 return
            return content if content is not None else ""
        except Exception as e:
            last_err = e
            if attempt < len(_BACKOFF):
                delay = _BACKOFF[attempt]
                print(f"[llm-retry] attempt={attempt+1} sleep={delay}s err={str(e)[:160]}", flush=True)
                await asyncio.sleep(delay)
            # else: 已是最后一次，落到下面记账 + 返回占位

    # 3 次重试后仍失败：记账并返回空串占位，绝不让整 doc failed
    _log_failed_chunk(last_err, prompt, system_prompt, {**kwargs, "keyword_extraction": keyword_extraction})
    print(f"[llm-give-up] 4 attempts failed, placeholder empty; err={str(last_err)[:160]}", flush=True)
    return ""


async def embed_func(texts):
    r = await _emb.embeddings.create(input=list(texts), model=U.EMBED_SERVED_NAME)
    return np.array([d.embedding for d in r.data], dtype=np.float32)


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--modes", default="hybrid,mix", help="逗号分隔，建一次图跑多个检索 mode")
    ap.add_argument("--skip_index", action="store_true", help="复用已建图，只跑 query（多 mode 对比用）")
    ap.add_argument(
        "--resume_index",
        action="store_true",
        help="只恢复已有 workspace 的未完成建图队列，不重新 enqueue 输入",
    )
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--limit_chunks", type=int, default=0)
    ap.add_argument("--limit_q", type=int, default=0)
    ap.add_argument("--latency_q", type=int, default=0, help="latency模式:前N题一题一题(concurrency=1)计端到端延时, 不含索引加载")
    ap.add_argument("--sample_file", default="", help="JSON含question_ids列表,只评这些题(取代--limit_q的前N偏差采样)")
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--top_k", type=int, default=0,
                    help="top-k消融: 覆盖 QueryParam top_k/chunk_top_k, 输出改落 results/lightrag_k{K}/ 防覆盖主表")
    args = ap.parse_args()
    topk = args.top_k or U.TOP_K
    _bname = f"lightrag_k{topk}" if topk != U.TOP_K else "lightrag"

    chunks = json.load(open(f"{U.DATA_ROOT}/{args.dataset}/chunks.json"))
    # 脏语料清洗(喂 ainsert 前,只为IO不改语义):①0x12等非法XML控制字符→LightRAG写graph_chunk_entity_relation.graphml时崩(N5568+起含);②<|endoftext|>等tiktoken特殊token→编码ValueError(N1434+起含)。两者都是enterprise真实语料脏数据;小点(N1144)无故当前不受影响,提前清防N1434/N5568崩。见 memory project_enterprise_xml_control_char。
    import re as _re
    _XML_BAD = _re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f]')
    chunks = [_XML_BAD.sub(' ', c).replace('<|', '< |').replace('|>', '| >') for c in chunks]
    questions = [json.loads(l) for l in open(f"{U.DATA_ROOT}/{args.dataset}/questions.jsonl")]
    if args.sample_file:
        import json as _json
        _ids = set(_json.load(open(args.sample_file))["question_ids"])
        questions = [q for q in questions if q["id"] in _ids]
        print(f"[sample] {len(questions)} questions from {args.sample_file}", flush=True)
    if args.limit_chunks:
        chunks = chunks[:args.limit_chunks]
    if args.limit_q and not args.sample_file:
        questions = questions[:args.limit_q]

    ws_root = os.environ.get("LIGHTRAG_WS_ROOT") or f"{UNIFIED}/lightrag_ws"
    wd = f"{ws_root}/{args.dataset}"
    if args.fresh and os.path.isdir(wd):
        shutil.rmtree(wd)
    os.makedirs(wd, exist_ok=True)

    # failed-chunk 落盘路径（llm_func 给 4 次仍失败的调用记这里）
    global _FAILED_LOG_PATH
    os.makedirs(U.RESULTS_ROOT, exist_ok=True)
    _FAILED_LOG_PATH = f"{U.RESULTS_ROOT}/lightrag_failed_chunks_{args.dataset}.jsonl"

    rag = LightRAG(
        working_dir=wd, llm_model_func=llm_func,
        embedding_func=EmbeddingFunc(embedding_dim=U.EMBED_DIM, max_token_size=8192, func=embed_func),
        chunk_token_size=100000, chunk_overlap_token_size=0,   # 不二次切：每个统一 chunk = 一个 LightRAG chunk
        llm_model_max_async=32,          # 有界的 LLM 请求并发。
        max_parallel_insert=16,          # 文档级插入并发。
        embedding_func_max_async=16,
        default_llm_timeout=LLM_TIMEOUT, # ★ 默认 180s 是 53% failed 根因：队列堆积时 LightRAG 先杀 future 把 doc 标 failed。
                                         #    提到 600s → worker 级 max_execution_timeout=1200s，给 llm_func 内重试留足空间
    )
    await rag.initialize_storages()
    _METER.set_phase("build")            # 建图(ainsert)阶段的 LLM token 归 build
    if args.resume_index:
        print("[resume_index] resume existing document queue without enqueue", flush=True)
        await rag.apipeline_process_enqueue_documents()
    elif not args.skip_index:
        print(f"[insert] {len(chunks)} chunks ...", flush=True)
        await rag.ainsert(chunks)
    else:
        print("[skip_index] 复用已建图，直接 query", flush=True)

    out_dir = f"{U.RESULTS_ROOT}/{_bname}/{args.dataset}"
    os.makedirs(out_dir, exist_ok=True)
    sem = asyncio.Semaphore(args.concurrency)
    _METER.set_phase("qa")               # 查询(aquery)阶段的 LLM token 归 qa

    # ── latency 模式：索引已就绪后(不含加载),前 N 题一题一题(concurrency=1)计端到端延时 ──
    if args.latency_q:
        import statistics
        lat_dir = f"{U.RESULTS_ROOT}/latency"
        os.makedirs(lat_dir, exist_ok=True)
        lat_path = f"{lat_dir}/LightRAG_{args.dataset}.jsonl"
        lats = []
        with open(lat_path, "w") as lf:
            for _q in questions[:args.latency_q]:
                t0 = time.time()
                ans = await rag.aquery(_q["question"],
                                       param=QueryParam(mode="mix", top_k=topk, chunk_top_k=topk))
                dt = time.time() - t0
                lf.write(json.dumps({"id": _q["id"], "latency_sec": round(dt, 3)}, ensure_ascii=False) + "\n")
                lf.flush()
                lats.append(dt)
        n = len(lats)
        mean = statistics.mean(lats) if n else 0.0
        median = statistics.median(lats) if n else 0.0
        p90 = statistics.quantiles(lats, n=10)[8] if n >= 2 else (lats[0] if n else 0.0)
        print(f"[latency] LightRAG {args.dataset} n={n} mean={round(mean,3)}s median={round(median,3)}s p90={round(p90,3)}s", flush=True)
        await rag.finalize_storages()
        return

    for mode in [m.strip() for m in args.modes.split(",") if m.strip()]:
        print(f"[query] {len(questions)} questions mode={mode} ...", flush=True)

        async def one(q, _mode=mode):
            async with sem:
                try:
                    ans = await rag.aquery(q["question"],
                                           param=QueryParam(mode=_mode, top_k=topk, chunk_top_k=topk))
                except Exception as e:
                    ans = f"[error] {e}"
                return {"id": q["id"], "predicted_answer": str(ans), "retrieved_chunk_ids": []}

        preds = await asyncio.gather(*[one(q) for q in questions])
        with open(f"{out_dir}/predictions_{mode}.jsonl", "w") as f:
            for p in preds:
                f.write(json.dumps(p, ensure_ascii=False) + "\n")
        print(f"[done] {mode}: {len(preds)} -> predictions_{mode}.jsonl", flush=True)
    await rag.finalize_storages()
    # 旁路 token 记账:build=ainsert 阶段, qa=aquery 阶段(缓存命中不进 llm_func,不计=正确)
    _METER.dump("LightRAG" if _bname == "lightrag" else _bname,
                args.dataset, source="wrapper_usage",
                note=f"build=insert(实体/关系抽取+描述总结), qa=aquery; 缓存命中未计(未调LLM); top_k={topk}")


if __name__ == "__main__":
    asyncio.run(main())
