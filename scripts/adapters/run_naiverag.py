#!/usr/bin/env python
"""NaiveRAG 统一适配 driver —— 稠密向量检索 baseline(范式: dense/vector retrieval, 无图无结构)。

- 建图 = 把每个 chunk 用 Qwen3-Embedding-0.6B（U_EMBED_ENDPOINT）编码成向量, 零 LLM → build LLM token=0。
  正好对照 HippoRAG 的 OpenIE 重 build; 与 BM25(sparse, 也零 LLM build)同列"廉价 build"对照组。
  (嵌入向量落盘缓存 naiverag_ws/<dataset>/emb.npy, 重跑不重嵌。)
- 检索 = query 侧加 Qwen3 检索 instruction 编码 → 与 chunk 向量余弦 top-5(统一 U.TOP_K)。下标即统一 chunk_id。
- 生成 = 统一 reader LLM（U_LLM_ENDPOINT，关 thinking，temp=0），1 call/题。

QA 两段式(2026-06-15 用户拍板, 同 BM25):
  ① acc/token: ThreadPoolExecutor 并发(--concurrency, 默认8)。token 进程隔离、acc temp=0 确定 → 零污染。
  ② latency:  --latency_n>0 时并发段后对前 N 题 concurrency=1 测纯净延时(需 build 暂停、独占 vLLM)。

跑:
  <python> run_naiverag.py --dataset enterprise_N1144 [--limit_q M | --sample_file x.json] [--concurrency 8] [--latency_n 40] [--fresh]
"""
import os
import sys
import json
import time
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

os.environ.setdefault("OPENAI_API_KEY", "EMPTY")
UNIFIED = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 2026-07-07 迁移
sys.path.insert(0, UNIFIED)
import unified_config as U
import token_meter

import numpy as np
from openai import OpenAI

_METER = token_meter.TokenMeter()


def _embed(client, texts, instruction="", batch_size=64):
    """统一 embedding API 编码 → 归一化 float32 矩阵(N×dim)。doc 侧 instruction="", query 侧加检索 instruction。"""
    out = []
    for i in range(0, len(texts), batch_size):
        batch = [instruction + (t.replace("\n", " ") if t else " ") for t in texts[i:i + batch_size]]
        resp = client.embeddings.create(input=batch, model=U.EMBED_SERVED_NAME)
        out.extend(d.embedding for d in resp.data)
        if len(texts) > 5000 and (i // batch_size) % 50 == 0:
            print(f"  [embed {i + len(batch)}/{len(texts)}]", flush=True)
    arr = np.asarray(out, dtype=np.float32)
    arr = arr / (np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12)
    return arr


# reader: 统一契约 U.READER_SYSTEM_PROMPT(跨范式同口径)
def _answer(client, context, question, meter=True):
    resp = client.chat.completions.create(
        model=U.READER_MODEL, temperature=U.GEN_TEMPERATURE, max_tokens=U.GEN_MAX_TOKENS,
        seed=U.SEED, extra_body=U.LLM_EXTRA_BODY,
        messages=[{"role": "system", "content": U.READER_SYSTEM_PROMPT},
                  {"role": "user", "content": U.reader_user_msg(context, question)}])
    if meter:
        _METER.add(getattr(resp, "usage", None), phase="qa")
    return (resp.choices[0].message.content or "").strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--limit_chunks", type=int, default=0)
    ap.add_argument("--limit_q", type=int, default=0)
    ap.add_argument("--sample_file", default="", help="JSON含question_ids列表,只评这些题(绝对路径)")
    ap.add_argument("--concurrency", type=int, default=8, help="acc/token阶段并发数(token隔离+acc确定→并发无污染)")
    ap.add_argument("--latency_n", type=int, default=0, help=">0: 并发段后对前N题concurrency=1测纯净延时(需独占vLLM)")
    ap.add_argument("--latency_only", action="store_true", help="只测延时(跳过acc/token并发段, 不覆盖已有predictions/token)")
    ap.add_argument("--fresh", action="store_true", help="忽略嵌入缓存, 重新编码")
    ap.add_argument("--top_k", type=int, default=0,
                    help="top-k消融: 覆盖U.TOP_K, 输出改落 results/naiverag_k{K}/ 防覆盖主表")
    args = ap.parse_args()
    topk = args.top_k or U.TOP_K

    data_dir = f"{U.DATA_ROOT}/{args.dataset}"
    chunks = json.load(open(f"{data_dir}/chunks.json"))
    questions = [json.loads(l) for l in open(f"{data_dir}/questions.jsonl")]
    if args.limit_chunks:
        chunks = chunks[:args.limit_chunks]
    if args.sample_file:
        _ids = set(json.load(open(args.sample_file))["question_ids"])
        questions = [q for q in questions if q["id"] in _ids]
        print(f"[sample] {len(questions)} questions from {args.sample_file}", flush=True)
    elif args.limit_q:
        questions = questions[:args.limit_q]

    emb_client = OpenAI(base_url=U.EMBED_ENDPOINT, api_key="EMPTY", timeout=600.0, max_retries=5)

    # ---- build: 嵌入全部 chunk(零 LLM, build token=0), 落盘缓存 ----
    ws = f"{UNIFIED}/naiverag_ws/{args.dataset}"
    os.makedirs(ws, exist_ok=True)
    emb_path = f"{ws}/emb.npy"
    t_build = time.time()
    if os.path.exists(emb_path) and not args.fresh:
        mat = np.load(emb_path)
        if mat.shape[0] != len(chunks):
            print(f"[index] cache stale ({mat.shape[0]}!={len(chunks)}), re-embed", flush=True)
            mat = _embed(emb_client, chunks)
            np.save(emb_path, mat)
        else:
            print(f"[index] loaded cached emb {mat.shape}", flush=True)
    else:
        print(f"[index] embed {len(chunks)} chunks via configured endpoint ...", flush=True)
        mat = _embed(emb_client, chunks)
        np.save(emb_path, mat)
    build_sec = time.time() - t_build
    print(f"[index] ✓ emb {mat.shape} in {build_sec:.0f}s", flush=True)

    reader = OpenAI(base_url=U.READER_ENDPOINT, api_key="EMPTY", timeout=600.0, max_retries=5)
    _bname = f"naiverag_k{args.top_k}" if (args.top_k and args.top_k != U.TOP_K) else "naiverag"
    out_dir = f"{U.RESULTS_ROOT}/{_bname}/{args.dataset}"
    os.makedirs(out_dir, exist_ok=True)
    lat_dir = f"{U.RESULTS_ROOT}/latency"
    os.makedirs(lat_dir, exist_ok=True)

    def _retrieve(question):
        qv = _embed(emb_client, [question], instruction=U.EMBED_QUERY_INSTRUCTION)[0]   # query 侧加检索 instruction
        idxs = np.argsort(-(mat @ qv))[:topk].tolist()    # 余弦 top-5(消融时=--top_k); 下标即统一 chunk_id
        return idxs, "\n\n".join(chunks[j] for j in idxs)

    # ---- ① acc/token: 并发跑全量题; --latency_only时跳过(不覆盖已有acc/token) ----
    if not args.latency_only:
        print(f"[qa] {len(questions)} questions, concurrency={args.concurrency} (acc/token) ...", flush=True)

        def _one(i, q):
            idxs, ctx = _retrieve(q["question"])
            ans = _answer(reader, ctx, q["question"])
            return i, {"id": q["id"], "predicted_answer": ans, "retrieved_chunk_ids": idxs}

        preds = [None] * len(questions)
        done = 0
        with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
            futs = [ex.submit(_one, i, q) for i, q in enumerate(questions)]
            for fut in as_completed(futs):
                i, p = fut.result()
                preds[i] = p
                done += 1
                if done % 50 == 0:
                    print(f"  [{done}/{len(questions)}]", flush=True)

        with open(f"{out_dir}/predictions.jsonl", "w") as f:
            for p in preds:
                f.write(json.dumps(p, ensure_ascii=False) + "\n")
        _METER.set_bucket("build", 0, 0, calls=0)   # build=0 LLM(只 embedding)
        _METER.dump("NaiveRAG" if _bname == "naiverag" else _bname.upper(), args.dataset, source="wrapper_usage",
                    note=f"build=0 LLM (dense embed, build_sec={build_sec:.0f}); qa=1 LLM/Q, concurrency={args.concurrency}")
        print(f"[done] {len(preds)} predictions -> {out_dir}/predictions.jsonl", flush=True)

    # ---- ② latency: concurrency=1 抽样测纯净延时(meter=False 不污染 token 账) ----
    if args.latency_n:
        qs = questions[:args.latency_n]
        print(f"[latency] {len(qs)} questions, concurrency=1 (需独占vLLM) ...", flush=True)
        import statistics
        lats = []
        with open(f"{lat_dir}/NaiveRAG_{args.dataset}.jsonl", "w") as lf:
            for q in qs:
                t0 = time.time()                                  # 口径=检索(含query嵌入往返)+生成端到端
                idxs, ctx = _retrieve(q["question"])
                _answer(reader, ctx, q["question"], meter=False)
                dt = time.time() - t0
                lats.append(dt)
                lf.write(json.dumps({"id": q["id"], "latency_sec": round(dt, 3)}) + "\n")
                lf.flush()
        _p90 = statistics.quantiles(lats, n=10)[8] if len(lats) >= 2 else lats[0]
        print(f"[latency] NaiveRAG {args.dataset} n={len(lats)} mean={statistics.mean(lats):.2f}s "
              f"median={statistics.median(lats):.2f}s p90={_p90:.2f}s", flush=True)


if __name__ == "__main__":
    main()
