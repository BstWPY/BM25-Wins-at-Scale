#!/usr/bin/env python
"""graphrag 3.1 统一查询 driver（建图已由 `graphrag index` 完成）。

- load output/*.parquet → api.local_search 批量查询（关 thinking 已在 settings.call_args.extra_body 配好）
- response_type 引导简短答案，便于统一 EM/F1
- 从 context_data 尽力提取检索到的 text_unit → 映射回统一 chunk_id（graphrag 选 B：固有范式，recall 仅供参考）
- 输出统一 predictions.jsonl

用 graphrag .venv 跑：
  graphrag/.venv/bin/python run_graphrag_query.py --dataset 2wikimultihopqa [--limit_q N]
"""
import sys
import os
import json
import asyncio
import argparse
from pathlib import Path
import pandas as pd

UNIFIED = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 2026-07-07 迁移
sys.path.insert(0, UNIFIED)
import unified_config as U
import token_meter

from graphrag.config.load_config import load_config
import graphrag.api as api

# ---------- 查询 token hook(2026-06-15 补):graphrag/fnllm 经 LiteLLM 调 LLM(不是原生 openai),
# usage 在 LiteLLM 层。注册 LiteLLM CustomLogger 回调,抓每次 success event 的 response.usage(phase=qa)。
# 不碰 openai 层(在 openai 层换返回类型会破坏 LiteLLM 的 .parse 契约,踩过),只旁路记账,绝不改 query 逻辑。 ----------
_QA_METER = token_meter.TokenMeter()
try:
    import litellm
    from litellm.integrations.custom_logger import CustomLogger

    class _QAUsageLogger(CustomLogger):
        def log_success_event(self, kwargs, response_obj, start_time, end_time):
            try: _QA_METER.add(getattr(response_obj, "usage", None), phase="qa")
            except Exception: pass

        async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
            try: _QA_METER.add(getattr(response_obj, "usage", None), phase="qa")
            except Exception: pass

    litellm.callbacks = [_QAUsageLogger()]
    _HOOK_STATUS = "litellm.callbacks registered"
except Exception as _he:
    _HOOK_STATUS = f"litellm hook FAILED: {_he}"


def load_artifacts(ws: Path):
    out = ws / "output"
    def rd(name, optional=False):
        p = out / f"{name}.parquet"
        if not p.exists():
            if optional:
                return None
            raise FileNotFoundError(p)
        return pd.read_parquet(p)
    return {
        "entities": rd("entities"),
        "communities": rd("communities"),
        "community_reports": rd("community_reports"),
        "text_units": rd("text_units"),
        "relationships": rd("relationships"),
        "covariates": rd("covariates", optional=True),
    }


def extract_chunk_ids(ctx, text2cid):
    """从 context_data 尽力提取检索到的 text_unit 文本 → 统一 chunk_id。"""
    cids = []
    if not isinstance(ctx, dict):
        return cids
    for key, df in ctx.items():
        if not isinstance(df, pd.DataFrame):
            continue
        col = next((c for c in ("text", "content", "source", "Source") if c in df.columns), None)
        if col is None:
            continue
        for t in df[col].astype(str).tolist():
            cid = text2cid.get(t)
            if cid is not None:
                cids.append(cid)
    return list(dict.fromkeys(cids))  # 去重保序


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--method", default="local", choices=["local", "global", "drift"])
    ap.add_argument("--response_type", default="Short factual answer only, no explanation")
    ap.add_argument("--community_level", type=int, default=2)
    ap.add_argument("--concurrency", type=int, default=12)
    ap.add_argument("--limit_q", type=int, default=0)
    ap.add_argument("--sample_file", default="", help="JSON含question_ids列表,只评这些题(取代--limit_q的前N偏差采样)")
    ap.add_argument("--latency_q", type=int, default=0, help="latency模式:前N题一题一题(concurrency=1)计端到端延时, 不含索引加载")
    ap.add_argument("--workspace_dataset", default="",
                    help="复用另一个数据集的 GraphRAG index/workspace；用于 paraphrase 只替换 questions 不重建图")
    args = ap.parse_args()

    ws_dataset = args.workspace_dataset or args.dataset
    ws = Path(f"{UNIFIED}/graphrag_ws/{ws_dataset}")
    config = load_config(root_dir=ws)
    dfs = load_artifacts(ws)

    chunks = json.load(open(f"{U.DATA_ROOT}/{args.dataset}/chunks.json"))
    text2cid = {c: i for i, c in enumerate(chunks)}

    questions = [json.loads(l) for l in open(f"{U.DATA_ROOT}/{args.dataset}/questions.jsonl")]
    if args.sample_file:
        import json as _json
        _ids = set(_json.load(open(args.sample_file))["question_ids"])
        questions = [q for q in questions if q["id"] in _ids]
        print(f"[sample] {len(questions)} questions from {args.sample_file}", flush=True)
    elif args.limit_q:
        questions = questions[:args.limit_q]

    sem = asyncio.Semaphore(args.concurrency)
    _prog = {"done": 0, "t0": __import__("time").time()}   # 2026-07-20: global 长跑可观测

    async def one(q):
        async with sem:
            # 2026-07-20: 单题异常不再炸掉整个 gather(global 150题×2h 跑一半全丢代价太大);
            # 重试一次, 仍失败落 error 行(空答案, judge 记 0 分), 不影响 local 原行为(成功路径不变)。
            last_err = ""
            for _try in range(2):
                try:
                    if args.method == "global":
                        resp, ctx = await api.global_search(
                            config=config, entities=dfs["entities"], communities=dfs["communities"],
                            community_reports=dfs["community_reports"], community_level=args.community_level,
                            dynamic_community_selection=False, response_type=args.response_type, query=q["question"])
                    elif args.method == "drift":
                        resp, ctx = await api.drift_search(
                            config=config, entities=dfs["entities"], communities=dfs["communities"],
                            community_reports=dfs["community_reports"], text_units=dfs["text_units"],
                            relationships=dfs["relationships"], community_level=args.community_level,
                            response_type=args.response_type, query=q["question"])
                    else:  # local
                        resp, ctx = await api.local_search(
                            config=config, entities=dfs["entities"], communities=dfs["communities"],
                            community_reports=dfs["community_reports"], text_units=dfs["text_units"],
                            relationships=dfs["relationships"], covariates=dfs["covariates"],
                            community_level=args.community_level, response_type=args.response_type, query=q["question"])
                    _prog["done"] += 1
                    if _prog["done"] % 5 == 0:
                        _el = __import__("time").time() - _prog["t0"]
                        print(f"[prog] {_prog['done']}/{len(questions)} {_el:.0f}s "
                              f"qa_tok={_QA_METER._acc['qa']['total_tokens']:,}", flush=True)
                    return {"id": q["id"], "predicted_answer": str(resp),
                            "retrieved_chunk_ids": extract_chunk_ids(ctx, text2cid)}
                except Exception as e:
                    last_err = f"{type(e).__name__}: {e}"
                    print(f"[warn] {q['id']} try{_try} failed: {last_err[:200]}", flush=True)
            _prog["done"] += 1
            return {"id": q["id"], "predicted_answer": "", "retrieved_chunk_ids": [], "error": last_err[:500]}

    # ---------- latency 模式(2026-06-15):前 N 题严格顺序(concurrency=1)逐题计端到端延时,
    # 只计单题 query 调用(答案生成),不含索引加载。与正常路径完全隔离,latency_q==0 时不影响原行为。 ----------
    if args.latency_q:
        import time
        import statistics
        lat_dir = f"{U.RESULTS_ROOT}/latency"
        os.makedirs(lat_dir, exist_ok=True)
        lat_path = f"{lat_dir}/graphrag_{args.dataset}.jsonl"
        lat_qs = questions[:args.latency_q]
        latencies = []
        with open(lat_path, "w") as lf:
            for _q in lat_qs:
                t0 = time.time()
                if args.method == "global":
                    resp, ctx = await api.global_search(
                        config=config, entities=dfs["entities"], communities=dfs["communities"],
                        community_reports=dfs["community_reports"], community_level=args.community_level,
                        dynamic_community_selection=False, response_type=args.response_type, query=_q["question"])
                elif args.method == "drift":
                    resp, ctx = await api.drift_search(
                        config=config, entities=dfs["entities"], communities=dfs["communities"],
                        community_reports=dfs["community_reports"], text_units=dfs["text_units"],
                        relationships=dfs["relationships"], community_level=args.community_level,
                        response_type=args.response_type, query=_q["question"])
                else:  # local
                    resp, ctx = await api.local_search(
                        config=config, entities=dfs["entities"], communities=dfs["communities"],
                        community_reports=dfs["community_reports"], text_units=dfs["text_units"],
                        relationships=dfs["relationships"], covariates=dfs["covariates"],
                        community_level=args.community_level, response_type=args.response_type, query=_q["question"])
                dt = time.time() - t0
                latencies.append(dt)
                lf.write(json.dumps({"id": _q["id"], "latency_sec": round(dt, 3)}, ensure_ascii=False) + "\n")
        n = len(latencies)
        mean = statistics.mean(latencies) if n else 0.0
        median = statistics.median(latencies) if n else 0.0
        p90 = statistics.quantiles(latencies, n=10)[8] if n >= 2 else (latencies[0] if n else 0.0)
        print(f"[latency] graphrag {args.dataset} n={n} mean={mean:.3f}s median={median:.3f}s p90={p90:.3f}s")
        return

    results = await asyncio.gather(*[one(q) for q in questions])

    out_dir = f"{U.RESULTS_ROOT}/graphrag/{args.dataset}"
    os.makedirs(out_dir, exist_ok=True)
    out_path = f"{out_dir}/predictions_{args.method}.jsonl"
    with open(out_path, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[done] {len(results)} predictions -> {out_path}")
    # 旁路 token 记账:build 读 indexing-engine.log(纯读取);qa 由上面的 openai create hook 抓真实 usage(2026-06-15 补)
    _log = f"{UNIFIED}/graphrag_ws/{ws_dataset}/logs/indexing-engine.log"
    _bp, _bc, _bn = token_meter.parse_graphrag_indexing_log(_log, chat_model=U.LLM_SERVED_NAME)
    _QA_METER.set_bucket("build", _bp, _bc, calls=_bn)
    _qa_tot = _QA_METER._acc["qa"]["total_tokens"]
    print(f"[token] build={_bp + _bc} qa={_qa_tot} (qa via openai hook; calls={_QA_METER._acc['qa']['calls']})")
    if _qa_tot == 0:
        print("[WARN] qa token=0 — hook 可能没拦到 graphrag 的 LLM client(fnllm 走了别的路径),需排查")
    # 2026-07-20: 非 local 方法(global/drift)落独立 token 文件, 防覆盖主表 graphrag_<ds>.json 账本
    _tname = "graphrag" if args.method == "local" else f"graphrag_{args.method}"
    _QA_METER.dump(_tname, args.dataset, source="graphrag_log+openai_hook",
                   note="build=indexing-engine.log末块; qa=openai create hook真实usage(2026-06-15补)")


if __name__ == "__main__":
    asyncio.run(main())
