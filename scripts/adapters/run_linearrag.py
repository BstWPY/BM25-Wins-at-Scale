#!/usr/bin/env python
"""LinearRAG 统一适配 driver。

- spaCy NER 本地跑 GPU7（保留原生形态）
- embedding 走 U_EMBED_ENDPOINT（避开 transformers 版本冲突 + 与其他 baseline 数值统一）
- LLM 走 U_LLM_ENDPOINT，关 thinking
- 吃 unified/data/<dataset> 的统一 chunks，输出统一 predictions.jsonl
  （retrieved_chunk_ids 从 passage 前缀 "idx:" 还原，idx 即统一 chunk_id）

跑（linearrag 环境）:
  CUDA_VISIBLE_DEVICES=7 .../envs/linearrag/bin/python run_linearrag.py --dataset 2wikimultihopqa --fresh
"""
import os
if os.environ.get("LINEARRAG_CUDA_VISIBLE_DEVICES"):
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", os.environ["LINEARRAG_CUDA_VISIBLE_DEVICES"])
os.environ["OPENAI_BASE_URL"] = os.environ["U_LLM_ENDPOINT"]
os.environ["OPENAI_API_KEY"] = "EMPTY"

import sys
import json
import argparse
import re
import shutil
import numpy as np
from openai import OpenAI
import spacy
_gpu_ok = spacy.prefer_gpu()  # 关键修复:原代码只设CUDA_VISIBLE_DEVICES却没require/prefer_gpu,en_core_web_trf一直退回CPU奇慢;prefer_gpu真正启用GPU7,不可用时优雅回CPU
print(f"[spacy] GPU enabled = {_gpu_ok}", flush=True)

UNIFIED = os.environ.get("URAG_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LINEARRAG = os.environ.get("LINEARRAG_ROOT") or os.path.abspath(os.path.join(UNIFIED, "..", "LinearRAG"))
sys.path.insert(0, UNIFIED)
sys.path.insert(0, LINEARRAG)
import unified_config as U
os.chdir(LINEARRAG)                                            # LinearRAG 用相对 working_dir ./import

from src.config import LinearRAGConfig
from src.LinearRAG import LinearRAG
from src.utils import LLM_Model

import token_meter
_METER = token_meter.TokenMeter()       # 旁路 token 记账(进程隔离),与检索/线性结构算法无关

# enterprise 语料含非法 XML 控制字符(如 0x12),LinearRAG 用 igraph 写 GraphML(XML)会崩;
# 喂图前剥掉这些脏字符(XML 1.0 本就不接受,不影响检索/语义)。只动写 GraphML 的 baseline,不全局清洗。
_XML_BAD = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f]')


class APIEmbedder:
    """duck-type SentenceTransformer：.encode() 走统一 embedding API，签名兼容 LinearRAG 调用。"""
    def __init__(self, base_url, model, batch_size=64):
        self.client = OpenAI(base_url=base_url, api_key="EMPTY", timeout=600.0, max_retries=5)
        self.model = model
        self._bs = batch_size

    def encode(self, texts, normalize_embeddings=True, show_progress_bar=False, batch_size=None, **kw):
        single = isinstance(texts, str)
        if single:
            texts = [texts]
        if len(texts) == 0:                       # 建图缓存全命中时 texts 为空，避免 norm(axis=1) 报错
            return np.zeros((0, 0), dtype=np.float32)
        bs = batch_size or self._bs
        out = []
        for i in range(0, len(texts), bs):
            batch = [(t.replace("\n", " ") if t else " ") for t in texts[i:i + bs]]
            resp = self.client.embeddings.create(input=batch, model=self.model)
            out.extend(d.embedding for d in resp.data)
        arr = np.asarray(out, dtype=np.float32)
        if normalize_embeddings:
            arr = arr / (np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12)
        return arr[0] if single else arr


class PatchedLLM(LLM_Model):
    """关 thinking + 统一 max_tokens。"""
    def __init__(self, model, max_tokens):
        super().__init__(model)
        self.llm_config["max_tokens"] = max_tokens
        # LinearRAG 原生 LLM_Model 写死 httpx timeout=60s，全量+排队易超时 → 加大超时+重试
        self.openai_client = OpenAI(base_url=os.environ["OPENAI_BASE_URL"],
                                    api_key="EMPTY", timeout=600.0, max_retries=5)

    def infer(self, messages):
        resp = self.openai_client.chat.completions.create(
            **self.llm_config, messages=messages, extra_body=U.LLM_EXTRA_BODY)
        _METER.add(getattr(resp, "usage", None), phase="qa")   # 旁路记账,容错;不改下面 return
        return resp.choices[0].message.content


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--limit_chunks", type=int, default=0, help="调试：只用前 N chunks")
    ap.add_argument("--limit_q", type=int, default=0, help="调试：只用前 M questions")
    ap.add_argument("--sample_file", default="", help="JSON含question_ids列表,只评这些题(取代--limit_q的前N偏差采样)")
    ap.add_argument("--fresh", action="store_true", help="清该数据集建图缓存后重建")
    ap.add_argument("--latency_q", type=int, default=0, help="latency模式:前N题一题一题(concurrency=1)计端到端延时, 不含索引加载")
    ap.add_argument("--top_k", type=int, default=0,
                    help="top-k消融: 覆盖 retrieval_top_k, 输出改落 results/linearrag_k{K}/ 防覆盖主表")
    args = ap.parse_args()
    topk = args.top_k or U.TOP_K
    _bname = f"linearrag_k{topk}" if topk != U.TOP_K else "linearrag"

    data_dir = f"{U.DATA_ROOT}/{args.dataset}"
    chunks = json.load(open(f"{data_dir}/chunks.json"))
    questions = [json.loads(l) for l in open(f"{data_dir}/questions.jsonl")]
    if args.limit_chunks:
        chunks = chunks[:args.limit_chunks]
    if args.sample_file:
        import json as _json
        _ids = set(_json.load(open(args.sample_file))["question_ids"])
        questions = [q for q in questions if q["id"] in _ids]
        print(f"[sample] {len(questions)} questions from {args.sample_file}", flush=True)
    elif args.limit_q:
        questions = questions[:args.limit_q]
    passages = [f"{idx}:{_XML_BAD.sub('', c)}" for idx, c in enumerate(chunks)]
    lr_questions = [{"question": q["question"], "answer": q["answer"]} for q in questions]

    wd = os.path.join("./import", args.dataset)
    if args.fresh and os.path.isdir(wd):
        shutil.rmtree(wd)
        print(f"[fresh] cleared {wd}")

    config = LinearRAGConfig(
        dataset_name=args.dataset,
        embedding_model=APIEmbedder(U.EMBED_ENDPOINT, U.EMBED_SERVED_NAME, batch_size=64),
        spacy_model="en_core_web_trf",
        max_workers=16,
        llm_model=PatchedLLM(U.LLM_SERVED_NAME, U.GEN_MAX_TOKENS),
        max_iterations=3, iteration_threshold=0.4, passage_ratio=2, top_k_sentence=3,
        retrieval_top_k=topk,                    # 统一 top-5; 消融时可覆写
        use_vectorized_retrieval=False,
    )
    rag = LinearRAG(global_config=config)
    print(f"[index] {len(passages)} chunks ...")
    rag.index(passages)

    # latency 模式:索引已建好(上面 rag.index),不含索引加载;前 N 题严格一题一题(concurrency=1)
    # 计单题端到端 query 延时,只计 rag.qa 单题调用本身,写 results/latency/<BASELINE>_<dataset>.jsonl
    if args.latency_q:
        import time
        import statistics
        lat_dir = f"{U.RESULTS_ROOT}/latency"
        os.makedirs(lat_dir, exist_ok=True)
        lat_path = f"{lat_dir}/LinearRAG_{args.dataset}.jsonl"
        lats = []
        with open(lat_path, "w") as lf:
            for _q in questions[:args.latency_q]:
                t0 = time.time()
                rag.qa([{"question": _q["question"], "answer": _q["answer"]}])
                dt = time.time() - t0
                lats.append(dt)
                lf.write(json.dumps({"id": _q["id"], "latency_sec": round(dt, 3)},
                                    ensure_ascii=False) + "\n")
                lf.flush()
        n = len(lats)
        mean = statistics.mean(lats) if lats else 0.0
        median = statistics.median(lats) if lats else 0.0
        p90 = (statistics.quantiles(lats, n=10)[8] if len(lats) >= 2 else (lats[0] if lats else 0.0))
        print(f"[latency] LinearRAG {args.dataset} n={n} "
              f"mean={round(mean, 3)}s median={round(median, 3)}s p90={round(p90, 3)}s")
        return

    print(f"[qa] {len(lr_questions)} questions ...")
    results = rag.qa(lr_questions)

    out_dir = f"{U.RESULTS_ROOT}/{_bname}/{args.dataset}"
    os.makedirs(out_dir, exist_ok=True)
    out_path = f"{out_dir}/predictions.jsonl"
    with open(out_path, "w") as f:
        for r, q in zip(results, questions):
            cids = [int(p.split(":", 1)[0]) for p in r["sorted_passage"]
                    if p.split(":", 1)[0].isdigit()]
            f.write(json.dumps({"id": q["id"], "predicted_answer": r["pred_answer"],
                                "retrieved_chunk_ids": cids}, ensure_ascii=False) + "\n")
    print(f"[done] {len(results)} predictions -> {out_path}")
    # 旁路 token 记账:建图零 LLM(spaCy NER),QA 全经 PatchedLLM.infer
    _METER.dump("LinearRAG" if _bname == "linearrag" else _bname,
                args.dataset, source="wrapper_usage",
                note=f"build=0 (spaCy NER, no LLM); qa via PatchedLLM.infer; top_k={topk}")


if __name__ == "__main__":
    main()
