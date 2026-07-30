#!/usr/bin/env python
"""BM25 统一适配 driver —— 词法检索 baseline(范式: sparse/lexical retrieval)。

- 建图 = 纯 BM25Okapi 词法索引(英文分词 + 去停用词),零 LLM、零 embedding → build token=0。
  正好对照 HippoRAG 的 OpenIE 重 build;与 NaiveRAG(dense, 也零 LLM build)同列"廉价 build"对照组。
- 检索 = BM25 top-5(统一 U.TOP_K)→ 还原为统一 chunk_id(就是 chunks 列表下标)。
- 生成 = 统一 reader LLM（U_LLM_ENDPOINT，关 thinking，temp=0），1 call/题。

QA 两段式(2026-06-15 用户拍板: 并发 acc/token + 抽样测延时):
  ① acc/token: ThreadPoolExecutor 并发(--concurrency, 默认8)跑全量题。token 进程隔离记账、
     acc 在 temp=0 确定 → 二者与"题内并发"无关=零污染(只要本 baseline 独占 vLLM、跨 baseline 串行)。
  ② latency: --latency_n>0 时, 并发段后对前 N 题 concurrency=1 逐题测纯净端到端延时(需 build 暂停、独占 vLLM)。
  token 在①后即 dump(=全量并发口径); ②的 LLM 调用不再 dump(不污染 token 账)。

跑(需 rank_bm25 的 env, 如 zzt):
  <python> run_bm25.py --dataset enterprise_N1144 [--limit_q M | --sample_file x.json] [--concurrency 8] [--latency_n 40]
"""
import os
import sys
import json
import time
import argparse
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

os.environ.setdefault("OPENAI_API_KEY", "EMPTY")
UNIFIED = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 2026-07-07 迁移: 动态定位 unified/
sys.path.insert(0, UNIFIED)
import unified_config as U
import token_meter

import numpy as np
from rank_bm25 import BM25Okapi
from openai import OpenAI

_METER = token_meter.TokenMeter()       # 旁路 token 记账(进程隔离), 线程安全(内部有锁)


# ===== BM25 预处理: enterprise 语料是纯英文公司文档 → 标准英文分词(小写 + 词正则 + 去停用词)。 =====
_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "is", "are", "was", "were", "in", "on", "at", "to",
    "for", "of", "with", "it", "this", "that", "these", "those", "i", "you", "he", "she", "we", "they",
}
_WORD = re.compile(r"[a-z0-9]+")


def _tokenize(text):
    return [t for t in _WORD.findall((text or "").lower()) if t not in _STOPWORDS]


# ===== reader LLM(统一契约 U.READER_SYSTEM_PROMPT, 跨范式同口径; 关 thinking, temp=0) =====
def _answer(client, context, question, meter=True):
    resp = client.chat.completions.create(
        model=U.READER_MODEL, temperature=U.GEN_TEMPERATURE, max_tokens=U.GEN_MAX_TOKENS,
        seed=U.SEED, extra_body=U.LLM_EXTRA_BODY,
        messages=[{"role": "system", "content": U.READER_SYSTEM_PROMPT},
                  {"role": "user", "content": U.reader_user_msg(context, question)}])
    if meter:
        _METER.add(getattr(resp, "usage", None), phase="qa")   # 旁路记账, 容错; 不改 return
    return (resp.choices[0].message.content or "").strip()


_TOPK = U.TOP_K   # --top_k 消融时被 main() 覆盖


def _retrieve(question, bm25, chunks):
    idxs = np.argsort(-bm25.get_scores(_tokenize(question)))[:_TOPK].tolist()   # 下标即统一 chunk_id
    return idxs, "\n\n".join(chunks[j] for j in idxs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--limit_chunks", type=int, default=0)
    ap.add_argument("--limit_q", type=int, default=0)
    ap.add_argument("--sample_file", default="", help="JSON含question_ids列表,只评这些题(绝对路径)")
    ap.add_argument("--concurrency", type=int, default=8, help="acc/token阶段并发数(token隔离+acc确定→并发无污染)")
    ap.add_argument("--latency_n", type=int, default=0, help=">0: 并发段后对前N题concurrency=1测纯净延时(需独占vLLM)")
    ap.add_argument("--latency_only", action="store_true", help="只测延时(跳过acc/token并发段, 不覆盖已有predictions/token)")
    ap.add_argument("--top_k", type=int, default=0,
                    help="top-k消融: 覆盖U.TOP_K, 输出改落 results/bm25_k{K}/ 防覆盖主表")
    args = ap.parse_args()
    if args.top_k and args.top_k != U.TOP_K:
        global _TOPK
        _TOPK = args.top_k

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

    # ---- build: 纯词法索引, 零 LLM(build token=0) ----
    t_build = time.time()
    print(f"[index] BM25 tokenize {len(chunks)} chunks ...", flush=True)
    bm25 = BM25Okapi([_tokenize(c) for c in chunks])
    build_sec = time.time() - t_build
    print(f"[index] ✓ BM25 built in {build_sec:.0f}s", flush=True)

    client = OpenAI(base_url=U.READER_ENDPOINT, api_key="EMPTY", timeout=600.0, max_retries=5)
    _bname = f"bm25_k{args.top_k}" if (args.top_k and args.top_k != U.TOP_K) else "bm25"
    out_dir = f"{U.RESULTS_ROOT}/{_bname}/{args.dataset}"
    os.makedirs(out_dir, exist_ok=True)
    lat_dir = f"{U.RESULTS_ROOT}/latency"
    os.makedirs(lat_dir, exist_ok=True)

    # ---- ① acc/token: 并发跑全量题(token隔离+acc确定→并发无污染); --latency_only时跳过(不覆盖已有acc/token) ----
    if not args.latency_only:
        print(f"[qa] {len(questions)} questions, concurrency={args.concurrency} (acc/token) ...", flush=True)

        def _one(i, q):
            idxs, ctx = _retrieve(q["question"], bm25, chunks)
            ans = _answer(client, ctx, q["question"])
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
        _METER.set_bucket("build", 0, 0, calls=0)   # build=0(纯词法, 无 LLM)
        _METER.dump("BM25" if _bname == "bm25" else _bname.upper(), args.dataset, source="wrapper_usage",
                    note=f"build=0 (lexical, build_sec={build_sec:.0f}); qa=1 LLM/Q, concurrency={args.concurrency}")
        print(f"[done] {len(preds)} predictions -> {out_dir}/predictions.jsonl", flush=True)

    # ---- ② latency: concurrency=1 抽样测纯净延时(meter=False 不污染 token 账) ----
    if args.latency_n:
        qs = questions[:args.latency_n]
        print(f"[latency] {len(qs)} questions, concurrency=1 (需独占vLLM) ...", flush=True)
        import statistics
        lats = []
        with open(f"{lat_dir}/BM25_{args.dataset}.jsonl", "w") as lf:
            for q in qs:
                t0 = time.time()                                  # 口径=检索+生成端到端(与HippoRAG/sandbox一致)
                idxs, ctx = _retrieve(q["question"], bm25, chunks)
                _answer(client, ctx, q["question"], meter=False)
                dt = time.time() - t0
                lats.append(dt)
                lf.write(json.dumps({"id": q["id"], "latency_sec": round(dt, 3)}) + "\n")
                lf.flush()
        _p90 = statistics.quantiles(lats, n=10)[8] if len(lats) >= 2 else lats[0]
        print(f"[latency] BM25 {args.dataset} n={len(lats)} mean={statistics.mean(lats):.2f}s "
              f"median={statistics.median(lats):.2f}s p90={_p90:.2f}s", flush=True)


if __name__ == "__main__":
    main()
