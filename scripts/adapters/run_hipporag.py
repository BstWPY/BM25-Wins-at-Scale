#!/usr/bin/env python
"""HippoRAG 2 统一适配 driver。
- LLM→U_LLM_ENDPOINT、embedding→U_EMBED_ENDPOINT（由环境变量配置）
- 吃统一 chunks（每个 chunk 作为一个 passage）、统一 top-5、输出统一 predictions.jsonl
- retrieved docs(passage 文本) 映射回统一 chunk_id

跑（HippoRAG conda env）:
  envs/HippoRAG/bin/python run_hipporag.py --dataset 2wikimultihopqa [--limit_chunks N --limit_q M]
"""
import os
import sys
import json
import argparse

os.environ.setdefault("OPENAI_API_KEY", "EMPTY")
UNIFIED = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 2026-07-07 迁移: 动态定位 unified/
sys.path.insert(0, UNIFIED)
import unified_config as U
import token_meter

from hipporag import HippoRAG
from hipporag.utils.config_utils import BaseConfig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--limit_chunks", type=int, default=0)
    ap.add_argument("--limit_q", type=int, default=0)
    ap.add_argument("--skip_qa", action="store_true",
                    help="只建图+记 build token,跳过慢 QA(token-scaling 只需 build)")
    ap.add_argument("--save_dir", default="",
                    help="共享 workspace(增量建图用),默认每点独立 hipporag_ws/{dataset}")
    ap.add_argument("--latency_q", type=int, default=0,
                    help="latency模式:前N题一题一题(concurrency=1)计端到端延时, 不含索引加载")
    ap.add_argument("--latency_only", action="store_true",
                    help="配合--latency_q: 只写latency文件, 不写predictions/不dump token(Block B专用, 避免覆盖Block A的acc/token)")
    ap.add_argument("--sample_file", default="", help="JSON含question_ids列表,只评这些题(取代--limit_q的前N偏差采样)")
    ap.add_argument("--top_k", type=int, default=0,
                    help="top-k消融: 覆盖 retrieval_top_k/qa_top_k, 输出改落 results/hipporag_k{K}/ 防覆盖主表")
    ap.add_argument("--q_shard_idx", type=int, default=0,
                    help="QA分片序号(0-based)。2026-07-09 用户批准: 大档checkpoint QA的PPR串行太慢, 按题分片多进程")
    ap.add_argument("--q_shard_num", type=int, default=1,
                    help="QA分片总数; >1时取 questions[idx::num] 且 predictions 写 shard 文件, 各分片独立 URAG_ROOT 隔离记账")
    args = ap.parse_args()

    chunks = json.load(open(f"{U.DATA_ROOT}/{args.dataset}/chunks.json"))
    questions = [json.loads(l) for l in open(f"{U.DATA_ROOT}/{args.dataset}/questions.jsonl")]
    if args.limit_chunks:
        chunks = chunks[:args.limit_chunks]
    if args.sample_file:
        import json as _json
        _ids = set(_json.load(open(args.sample_file))["question_ids"])
        questions = [q for q in questions if q["id"] in _ids]
        print(f"[sample] {len(questions)} questions from {args.sample_file}", flush=True)
    elif args.limit_q:
        questions = questions[:args.limit_q]
    if args.q_shard_num > 1:
        questions = questions[args.q_shard_idx::args.q_shard_num]
        print(f"[shard] {args.q_shard_idx}/{args.q_shard_num} -> {len(questions)} questions", flush=True)

    topk = args.top_k or U.TOP_K
    _bname = f"hipporag_k{topk}" if topk != U.TOP_K else "hipporag"
    _ws = args.save_dir or f"{UNIFIED}/hipporag_ws/{args.dataset}"
    config = BaseConfig(
        save_dir=_ws,
        dataset="enterprise",     # 2026-06-15 公平性修复(审计#3): 渲染 rag_qa_enterprise 的 zero-shot reader, 不回退 musique CoT
        llm_name=U.LLM_SERVED_NAME, llm_base_url=U.LLM_ENDPOINT,
        embedding_model_name=U.EMBED_SERVED_NAME, embedding_base_url=U.EMBED_ENDPOINT,
        temperature=0, max_new_tokens=U.GEN_MAX_TOKENS, seed=U.SEED,
        retrieval_top_k=topk, qa_top_k=topk, linking_top_k=5,
        synonymy_edge_topk=256,   # 2026-06-15: 默认2047致大点KNN中间张量爆RAM(60万实体×2047≈12亿对≈700GB→N66932 OOM)。
                                  # 建图loop实际每实体只取≤100条(score≥0.8且num_nns>100即break,实测均~50),256>>该上限=同义图严格无损,
                                  # 仅省RAM ~16×(→~30GB),让HippoRAG铺到28点。已建18点本就~100截断、与256口径一致、不用重建。
        synonymy_edge_query_batch_size=8000,   # 默认1000/10000致KNN批迭代极多(N66932=1251×125批≈1h,纯Python循环+显存搬运开销)。
        synonymy_edge_key_batch_size=50000,    # 加大→批数砍~8×/5×、峰值显存仅~2.3GB(空闲GPU16G足);topk跨batch合并与batch大小无关=结果严格无损,仅省wall。
        embedding_return_as_normalized=True,
    )
    rag = HippoRAG(global_config=config)
    print(f"[index] {len(chunks)} chunks ...", flush=True)
    rag.index(chunks)
    # 旁路 token 记账:OpenIE 建图后扫它自己的 llm_cache(纯读取)=build token
    _sq = f"{_ws}/llm_cache/{U.LLM_SERVED_NAME.replace('/', '_')}_cache.sqlite"
    _bp, _bc, _ = token_meter.scan_hipporag_sqlite(_sq)
    if args.latency_q:
        # 串行逐题 QA(concurrency=1, 独占vLLM): 一次过拿 acc(predictions) + token(扫sqlite差值) + 纯净延时。
        # 避开 PPR(igraph)非线程安全; 串行独占=延时干净。--latency_q N 取(--sample_file 采样后)前 N 题。
        # 注意: HippoRAG llm_cache 按内容 hash 去重, 同题勿重跑(命中缓存→延时失真), 每题只跑一次。
        import time, statistics
        lat_dir = f"{U.RESULTS_ROOT}/latency"
        os.makedirs(lat_dir, exist_ok=True)
        out_dir = f"{U.RESULTS_ROOT}/{_bname}/{args.dataset}"
        os.makedirs(out_dir, exist_ok=True)
        text2cid = {c: i for i, c in enumerate(chunks)}
        _qs = questions[:args.latency_q]
        _lats, _preds = [], []
        with open(f"{lat_dir}/HippoRAG_{args.dataset}.jsonl", "w") as lf:
            for _i, _q in enumerate(_qs):
                t0 = time.time()
                solutions, responses, metadata = rag.rag_qa(queries=[_q["question"]])
                dt = time.time() - t0
                _lats.append(dt)
                _sol = solutions[0]
                _ans = getattr(_sol, "answer", None) or ""
                _docs = getattr(_sol, "docs", None) or []
                _cids = [text2cid[d] for d in _docs if d in text2cid]
                _preds.append({"id": _q["id"], "predicted_answer": str(_ans), "retrieved_chunk_ids": _cids})
                lf.write(json.dumps({"id": _q["id"], "latency_sec": round(dt, 3)}) + "\n")
                lf.flush()
                if _i < 5 or (_i + 1) % 20 == 0:
                    print(f"  [qa {_i + 1}/{len(_qs)}] {dt:.0f}s", flush=True)
        if not args.latency_only:
            # Block A(全量acc/token)才写predictions+dump token; Block B只测延时, 跳过(不覆盖Block A)
            with open(f"{out_dir}/predictions.jsonl", "w") as pf:
                for _p in _preds:
                    pf.write(json.dumps(_p, ensure_ascii=False) + "\n")
            # token: qa = qa后扫sqlite - qa前(_bp); 差值法不受拷贝全缓存影响(build部分抵消)。
            # 审计#1修复: QA-mode绝不写build桶(此处_bp扫的是拷贝来的全缓存=污染); build token只认ledger。
            _tp, _tc, _ = token_meter.scan_hipporag_sqlite(_sq)
            _meter = token_meter.TokenMeter()
            _meter.set_bucket("build", 0, 0)
            _meter.set_bucket("qa", max(0, _tp - _bp), max(0, _tc - _bc))
            _meter.dump("HippoRAG" if _bname == "hipporag" else _bname,
                        args.dataset, source="hipporag_sqlite",
                        note=f"serial QA n={len(_qs)} top_k={topk}: qa=delta(进程隔离); build=0(QA-mode不记,见ledger)")
        _n = len(_lats)
        _mean = statistics.mean(_lats) if _lats else 0.0
        _median = statistics.median(_lats) if _lats else 0.0
        _p90 = (statistics.quantiles(_lats, n=10)[8] if _n >= 2 else (_lats[0] if _lats else 0.0))
        print(f"[serial-qa] HippoRAG {args.dataset} n={_n} preds={len(_preds)} "
              f"mean={round(_mean, 1)}s median={round(_median, 1)}s p90={round(_p90, 1)}s", flush=True)
        return
    if args.skip_qa:
        # token-scaling 只需 build token;HippoRAG QA 极慢(~194s/题),跳过,建图完即记账返回
        _meter = token_meter.TokenMeter()
        _meter.set_bucket("build", _bp, _bc)
        _meter.dump("HippoRAG", args.dataset, source="hipporag_sqlite",
                    note="build=OpenIE llm_cache; QA skipped (token-scaling only needs build)")
        print(f"[done-build-only] build_tok={_bp + _bc} (QA skipped)", flush=True)
        return
    print(f"[qa] {len(questions)} questions ...", flush=True)
    solutions, responses, metadata = rag.rag_qa(queries=[q["question"] for q in questions])
    # QA 后再扫=总量,差值=qa token(cache 按内容 hash 去重,反映真实成功调用)
    _tp, _tc, _ = token_meter.scan_hipporag_sqlite(_sq)
    _meter = token_meter.TokenMeter()
    _meter.set_bucket("build", 0, 0)   # 审计#1修复: QA-mode不记build(子规模扫的是拷贝全缓存=污染); build只认ledger
    _meter.set_bucket("qa", max(0, _tp - _bp), max(0, _tc - _bc))
    _meter.dump("HippoRAG" if _bname == "hipporag" else _bname,
                args.dataset, source="hipporag_sqlite",
                note=f"qa=rag_qa增量delta(进程隔离); build=0(QA-mode不记,见ledger); top_k={topk}")

    text2cid = {c: i for i, c in enumerate(chunks)}
    out_dir = f"{U.RESULTS_ROOT}/{_bname}/{args.dataset}"
    os.makedirs(out_dir, exist_ok=True)
    _pred = f"predictions_shard{args.q_shard_idx}.jsonl" if args.q_shard_num > 1 else "predictions.jsonl"
    with open(f"{out_dir}/{_pred}", "w") as f:
        for q, sol in zip(questions, solutions):
            ans = getattr(sol, "answer", None) or ""
            docs = getattr(sol, "docs", None) or []
            cids = [text2cid[d] for d in docs if d in text2cid]
            f.write(json.dumps({"id": q["id"], "predicted_answer": str(ans),
                                "retrieved_chunk_ids": cids}, ensure_ascii=False) + "\n")
    print(f"[done] {len(solutions)} -> {out_dir}/{_pred}", flush=True)


if __name__ == "__main__":
    main()
