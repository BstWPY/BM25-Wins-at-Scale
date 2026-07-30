#!/usr/bin/env python
"""sandbox agent harness (Qwen-Agent + vLLM 原生 function calling) — v2 充足预算 + 细粒度计量。

端点③ sandbox(扁平原文树)与端点② mapping(图展平树)共用此 harness,只换 --tree_root。

- LLM: Qwen3.6-27B @ U_LLM_ENDPOINT, use_raw_api(走 vLLM qwen3_xml tool parser), 关 thinking
- 工具: list_dir / grep / read_doc, 约束 tree_root、只读
- 预算(关键): --max_calls 控 LLM 调用轮数。qwen-agent 用 settings 的 MAX_LLM_CALL_PER_RUN、默认仅 20、
  且不认 run() 的 max_llm_calls kwarg,只能 monkey-patch 模块变量。大规模 agent 步数会涨,给足以免提前放弃。
  context 截断 max_input_tokens 是优雅降级(留最近历史)非放弃,统计触发轮数量化瓶颈。
- 计量(per-question): prompt/completion token、LLM调用次数与平均token、工具调用次数与平均token、
  总延时、LLM延时、工具延时、截断轮数。全 hook vLLM 真实 usage、thread-local 隔离。
- retrieved: read_doc 读过的文件 -> filemap -> chunk_ids, 对齐 gold_chunk_ids 评 recall

跑: qwenagent/bin/python run_sandbox_agent.py --dataset enterprise_N1144 [--limit_q N --workers 8 --max_calls 80]
"""
import os
import sys
import json
import time
import argparse
import threading
import subprocess
import importlib.util
import hashlib
import re

import numpy as np

W = os.environ.get("URAG_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, W)
import token_meter

# ---------- token hook: 拦 openai client 拿 vLLM 真实 usage + LLM 调用计数/计时(thread-local 隔离) ----------
import openai
from openai.resources.chat.completions import Completions

_TL = threading.local()
_orig_create = Completions.create


def _hook_usage(u, sec):
    m = getattr(_TL, "meter", None)
    if m is None:
        return
    m.add(u, phase="qa")
    st = _TL.stat
    st["llm_calls"] += 1
    st["llm_sec"] += sec
    if int(getattr(u, "prompt_tokens", 0) or 0) >= _TL.trunc_thresh:  # 该轮 prompt 顶到上限=触发截断
        st["trunc_rounds"] += 1


def _patched_create(self, *args, **kwargs):
    t0 = time.time()
    if kwargs.get("stream"):
        so = dict(kwargs.get("stream_options") or {})
        so["include_usage"] = True
        kwargs["stream_options"] = so
        resp = _orig_create(self, *args, **kwargs)

        def _gen():
            for chunk in resp:
                u = getattr(chunk, "usage", None)
                if u is not None:
                    _hook_usage(u, time.time() - t0)
                yield chunk
        return _gen()
    resp = _orig_create(self, *args, **kwargs)
    u = getattr(resp, "usage", None)
    if u is not None:
        _hook_usage(u, time.time() - t0)
    return resp


Completions.create = _patched_create

# ---------- 文件工具(约束 tree_root + 计时/计数, thread-local 支持并发) ----------
from qwen_agent.agents import Assistant
from qwen_agent.agents import fncall_agent           # monkey-patch 步数上限用
from qwen_agent.tools.base import BaseTool, register_tool


# ---------- Agent + native-BM25 ranked-search control ----------
_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "is", "are", "was", "were", "in", "on", "at", "to",
    "for", "of", "with", "it", "this", "that", "these", "those", "i", "you", "he", "she", "we", "they",
}
_WORD = re.compile(r"[a-z0-9]+")


def _tokenize(text):
    """Match the tokenizer in adapters/run_bm25.py exactly."""
    return [t for t in _WORD.findall((text or "").lower()) if t not in _STOPWORDS]


def _load_bm25_class():
    """Load rank_bm25 without mixing the Qwen-Agent and judge environments."""
    module_path = os.path.abspath(os.path.join(
        W, "..", "..", "envs", "vllm_judge", "lib", "python3.12",
        "site-packages", "rank_bm25.py",
    ))
    spec = importlib.util.spec_from_file_location("_urag_rank_bm25", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load rank_bm25 from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.BM25Okapi


def _safe_path_component(name):
    return re.sub(r"[^A-Za-z0-9._-]", "_", str(name))[:120]


def _build_ranked_search(dataset):
    """Build the same chunk-level BM25 index as the native baseline."""
    ds = f"{W}/data/{dataset}"
    t0 = time.time()
    print(f"[ranked-search] loading chunks for {dataset} ...", flush=True)
    chunks = json.load(open(f"{ds}/chunks.json"))
    BM25Okapi = _load_bm25_class()
    print(f"[ranked-search] tokenizing {len(chunks)} chunks ...", flush=True)
    bm25 = BM25Okapi([_tokenize(c) for c in chunks])

    chunk_paths = [""] * len(chunks)
    with open(f"{ds}/chunks_meta.jsonl") as f:
        for line in f:
            row = json.loads(line)
            cid = int(row["chunk_id"])
            if 0 <= cid < len(chunk_paths):
                chunk_paths[cid] = (
                    f"{_safe_path_component(row.get('source', 'unknown'))}/"
                    f"{_safe_path_component(row.get('dsid', 'unknown'))}.md"
                )
    elapsed = time.time() - t0
    print(f"[ranked-search] ✓ built in {elapsed:.1f}s", flush=True)
    return bm25, chunks, chunk_paths, elapsed


def _root():
    return getattr(_TL, "root", W)


def _arg(params, key):
    try:
        d = json.loads(params) if isinstance(params, str) else params
        if isinstance(d, dict):
            return d.get(key, "")
    except Exception:
        pass
    import re
    m = re.search(r'"%s"\s*:\s*"([^"]*)"' % re.escape(key), str(params))  # 坏JSON尽力提取,不抛
    return m.group(1) if m else ""


def _timed(fn):
    """工具 call 计时 + 计数(thread-local)。"""
    def wrap(self, params, **kwargs):
        t0 = time.time()
        try:
            return fn(self, params, **kwargs)
        finally:
            st = getattr(_TL, "stat", None)
            if st is not None:
                st["tool_calls"] += 1
                st["tool_sec"] += time.time() - t0
    return wrap


@register_tool("list_dir")
class ListDir(BaseTool):
    description = ("List entries under a directory (relative path; empty string = corpus root). "
                  "Use to discover the source folders and documents.")
    parameters = {"type": "object", "properties": {
        "path": {"type": "string", "description": "relative dir, empty for root"}}, "required": []}

    @_timed
    def call(self, params, **kwargs):
        rel = _arg(params, "path")
        rt = os.path.realpath(_root())
        fp = os.path.realpath(os.path.join(rt, rel or ""))
        if fp != rt and not fp.startswith(rt + os.sep):   # 带分隔符防同前缀兄弟目录
            return "error: outside corpus"
        try:
            return json.dumps(sorted(os.listdir(fp))[:200], ensure_ascii=False)
        except Exception as e:
            return f"error: {e}"


@register_tool("grep")
class Grep(BaseTool):
    description = ("Search all documents for a keyword/phrase (fixed string, case-insensitive). "
                  "Returns up to 30 matching relative file paths.")
    parameters = {"type": "object", "properties": {
        "pattern": {"type": "string", "description": "keyword or phrase"}}, "required": ["pattern"]}

    @_timed
    def call(self, params, **kwargs):
        pat = _arg(params, "pattern")
        if not pat:
            return "error: empty pattern"
        try:
            r = subprocess.run(["grep", "-rilF", "--", pat, _root()],   # -- 防 pattern 以 - 开头被当选项
                               capture_output=True, text=True, timeout=90)
            fs = [os.path.relpath(x, _root()) for x in r.stdout.strip().split("\n") if x]
            return json.dumps(fs[:30], ensure_ascii=False) if fs else "no match"
        except Exception as e:
            return f"error: {e}"


@register_tool("bm25_search")
class BM25Search(BaseTool):
    description = (
        "Search the corpus with the native BM25 retriever. Returns the exact top-5 chunks, "
        "including rank, chunk ID, source path, and full chunk text. Start with the user's "
        "question; reformulate only when the first result set is insufficient."
    )
    parameters = {"type": "object", "properties": {
        "query": {"type": "string", "description": "lexical search query"}}, "required": ["query"]}

    @_timed
    def call(self, params, **kwargs):
        requested_query = _arg(params, "query")
        if not requested_query:
            return "error: empty query"
        search_no = int(getattr(_TL, "bm25_search_calls", 0) or 0)
        # Hard guarantee for the control: the first retrieval is exactly Native BM25(question).
        # Later calls may use agent reformulations, which is the agency being tested.
        query = getattr(_TL, "original_question", requested_query) if search_no == 0 else requested_query
        _TL.bm25_search_calls = search_no + 1
        bm25 = getattr(_TL, "bm25", None)
        chunks = getattr(_TL, "ranked_chunks", None)
        paths = getattr(_TL, "ranked_paths", None)
        if bm25 is None or chunks is None or paths is None:
            return "error: ranked search is not initialized"
        idxs = np.argsort(-bm25.get_scores(_tokenize(query)))[:5].tolist()
        retrieved = getattr(_TL, "retrieved_chunk_ids", None)
        if retrieved is not None:
            retrieved.extend(idxs)
        trace = getattr(_TL, "search_trace", None)
        if trace is not None:
            trace.append({
                "search_no": search_no + 1,
                "requested_query": requested_query,
                "effective_query": query,
                "chunk_ids": idxs,
            })
        rows = [
            {"rank": rank, "chunk_id": cid, "path": paths[cid], "content": chunks[cid]}
            for rank, cid in enumerate(idxs, 1)
        ]
        return json.dumps({"effective_query": query, "results": rows}, ensure_ascii=False)


@register_tool("read_doc")
class ReadDoc(BaseTool):
    description = "Read a document by its relative path under the corpus (up to 8000 chars)."
    parameters = {"type": "object", "properties": {
        "path": {"type": "string", "description": "relative file path"}}, "required": ["path"]}

    @_timed
    def call(self, params, **kwargs):
        rel = _arg(params, "path")
        rt = os.path.realpath(_root())
        fp = os.path.realpath(os.path.join(rt, rel or ""))
        if fp != rt and not fp.startswith(rt + os.sep):   # 带分隔符防同前缀兄弟目录
            return "error: outside corpus"
        try:
            txt = open(fp, errors="ignore").read()
            rf = getattr(_TL, "read_files", None)
            if rf is not None:
                rf.append(rel)
            return txt[:8000]
        except Exception as e:
            return f"error: {e}"


MAX_INPUT_TOKENS = 26000
LLM_CFG = {
    "model": "qwen3.6-27b",
    "model_server": os.environ["U_LLM_ENDPOINT"],
    "api_key": "EMPTY",
    "generate_cfg": {"temperature": 0, "use_raw_api": True, "max_input_tokens": MAX_INPUT_TOKENS,
                     "extra_body": {"chat_template_kwargs": {"enable_thinking": False}}},
}
SYS = ("You are a research assistant over an enterprise document corpus, organized as files under source "
       "folders (slack, gmail, jira, confluence, google_drive, linear, github, fireflies, hubspot). "
       "Use list_dir to orient, grep to locate relevant documents by keyword, and read_doc to read them. "
       "Answer ONLY from the documents. Be concise and factual, and cite the relative file paths you used.")
SYS_RANKED = (
    "You are a research assistant over an enterprise document corpus. Use bm25_search with the user's "
    "original question first. Inspect the returned top-5 chunks carefully. If they are sufficient, answer "
    "immediately; otherwise reformulate the search query or use read_doc on a returned source path to inspect "
    "more context. Answer ONLY from retrieved documents. Be concise and factual, and cite source paths or "
    "chunk IDs. Do not use outside knowledge."
)


def run_one(question, tree_root, filemap, ranked=None):
    _TL.meter = token_meter.TokenMeter()
    _TL.root = tree_root
    _TL.read_files = []
    _TL.retrieved_chunk_ids = []
    _TL.original_question = question
    _TL.bm25_search_calls = 0
    _TL.search_trace = []
    if ranked is None:
        _TL.bm25 = None
        _TL.ranked_chunks = None
        _TL.ranked_paths = None
        function_list = ["list_dir", "grep", "read_doc"]
        system_message = SYS
    else:
        _TL.bm25, _TL.ranked_chunks, _TL.ranked_paths = ranked
        function_list = ["bm25_search", "read_doc"]
        system_message = SYS_RANKED
    _TL.trunc_thresh = MAX_INPUT_TOKENS - 1500          # prompt 顶到此=该轮被截断
    _TL.stat = {"llm_calls": 0, "llm_sec": 0.0, "tool_calls": 0, "tool_sec": 0.0, "trunc_rounds": 0}
    answer, err, failure_type = "", "", ""
    t0 = time.time()
    try:
        bot = Assistant(llm=LLM_CFG, function_list=function_list, system_message=system_message)
        last = None
        for resp in bot.run([{"role": "user", "content": question}]):
            last = resp
        for m in (last or []):
            if m.get("role") == "assistant" and m.get("content"):
                answer = m["content"]
    except Exception as e:
        msg = f"{type(e).__name__}: {e}"
        if (
            ranked is not None
            and "BadRequestError" in msg
            and "Unterminated string" in msg
        ):
            # The model emitted malformed/truncated function-call JSON.  This
            # is an agent trajectory failure, not an infrastructure outage:
            # retain all retrieval/cost telemetry and score the empty answer 0.
            failure_type = "invalid_tool_call_json"
        else:
            err = msg
    total_sec = time.time() - t0
    rc = []
    for rel in dict.fromkeys(_TL.read_files):
        rc.extend(filemap.get(rel, []))
    rc.extend(_TL.retrieved_chunk_ids)
    qa = _TL.meter._acc["qa"]
    st = _TL.stat
    ptok, ctok = qa["prompt_tokens"], qa["completion_tokens"]
    ttok = ptok + ctok
    return {"answer": answer, "retrieved_chunk_ids": sorted(set(rc)),
            "read_files": list(dict.fromkeys(_TL.read_files)),
            "search_trace": _TL.search_trace,
            "prompt_tok": ptok, "completion_tok": ctok, "total_tok": ttok,
            "llm_calls": st["llm_calls"], "tool_calls": st["tool_calls"],
            "tok_per_llm_call": round(ttok / st["llm_calls"], 1) if st["llm_calls"] else 0,
            "tok_per_tool_call": round(ttok / st["tool_calls"], 1) if st["tool_calls"] else 0,
            "total_sec": round(total_sec, 2), "llm_sec": round(st["llm_sec"], 2),
            "tool_sec": round(st["tool_sec"], 2), "trunc_rounds": st["trunc_rounds"],
            "error": err, "failure_type": failure_type}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--tree_root", default="")
    ap.add_argument("--limit_q", type=int, default=0)
    ap.add_argument("--sample_file", default="", help="JSON含question_ids列表,只评这些题(取代--limit_q的前N偏差采样)")
    ap.add_argument("--workers", type=int, default=16)  # 07-21: 8→16, N259837 引擎只有4请求在跑、瓶颈在工具IO, 提并发压尾部ETA
    ap.add_argument("--max_calls", type=int, default=80,
                    help="LLM 调用轮数上限(预算);默认 80 给足,防大规模 agent 提前放弃")
    ap.add_argument("--ranked_search", action="store_true",
                    help="Agent+BM25 control: expose native BM25 top-5 search plus read_doc; write isolated results")
    ap.add_argument("--serial_sample", type=int, default=0,
                    help=">0: 并发段后对前N题串行(workers=1)重跑测纯净端到端延时(需独占vLLM),写latency文件;不再dump token")
    ap.add_argument("--latency_only", action="store_true",
                    help="Block B专用: 跳过并发acc/token, 只跑serial_sample测延时(配合--serial_sample, 不覆盖Block A的acc/token)")
    ap.add_argument("--latency_out_dir", default="",
                    help="延时结果目录; 默认 results/latency。重测应指定独立目录，避免覆盖论文当前结果")
    ap.add_argument("--latency_resume", action="store_true",
                    help="从 latency_out_dir 中已有的无错误题继续；错误尝试单独记入 *.errors.jsonl")
    args = ap.parse_args()
    # 步数上限调大:qwen-agent 用 settings 常量、默认 20、不认 run() 的 kwarg,只能 monkey-patch 模块变量
    fncall_agent.MAX_LLM_CALL_PER_RUN = args.max_calls

    ds = f"{W}/data/{args.dataset}"
    tree_root = args.tree_root or f"{W}/sandbox_agent/trees/{args.dataset}/flat"
    filemap = json.load(open(f"{W}/sandbox_agent/trees/{args.dataset}/filemap_flat.json"))
    ranked = None
    ranked_build_sec = 0.0
    if args.ranked_search:
        bm25, ranked_chunks, ranked_paths, ranked_build_sec = _build_ranked_search(args.dataset)
        ranked = (bm25, ranked_chunks, ranked_paths)
    questions = [json.loads(l) for l in open(f"{ds}/questions.jsonl")]
    if args.sample_file:
        import json as _json
        _ids = set(_json.load(open(args.sample_file))["question_ids"])
        questions = [q for q in questions if q["id"] in _ids]
        print(f"[sample] {len(questions)} questions from {args.sample_file}", flush=True)
    elif args.limit_q:
        questions = questions[:args.limit_q]
    baseline_name = "agent_bm25" if args.ranked_search else "sandbox"
    out_dir = f"{W}/results/{baseline_name}/{args.dataset}"
    os.makedirs(out_dir, exist_ok=True)

    from concurrent.futures import ThreadPoolExecutor
    agg = {"p": 0, "c": 0, "llm": 0, "tool": 0, "err": 0, "trunc": 0, "hit_cap": 0}
    done = [0]
    lock = threading.Lock()
    n = len(questions)
    t0 = time.time()
    fout = None   # Block A(acc/token)才开predictions; --latency_only(Block B)只测延时, 不写predictions/不dump token

    def _work(q):
        r = run_one(q["question"], tree_root, filemap, ranked=ranked)   # 全 thread-local 隔离
        rec = {"id": q["id"], "predicted_answer": r["answer"],
               "retrieved_chunk_ids": r["retrieved_chunk_ids"], "read_files": r["read_files"],
               "search_trace": r["search_trace"],
               "llm_calls": r["llm_calls"], "tool_calls": r["tool_calls"],
               "prompt_tok": r["prompt_tok"], "completion_tok": r["completion_tok"], "total_tok": r["total_tok"],
               "tok_per_llm_call": r["tok_per_llm_call"], "tok_per_tool_call": r["tok_per_tool_call"],
               "total_sec": r["total_sec"], "llm_sec": r["llm_sec"], "tool_sec": r["tool_sec"],
               "trunc_rounds": r["trunc_rounds"], "error": r["error"],
               "failure_type": r["failure_type"]}
        with lock:
            agg["p"] += r["prompt_tok"]; agg["c"] += r["completion_tok"]
            agg["llm"] += r["llm_calls"]; agg["tool"] += r["tool_calls"]
            if r["error"]:
                agg["err"] += 1
            if r["trunc_rounds"] > 0:
                agg["trunc"] += 1
            if r["llm_calls"] >= args.max_calls:          # 撞步数上限=预算不足的信号
                agg["hit_cap"] += 1
            done[0] += 1
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n"); fout.flush()
            if done[0] % 20 == 0 or done[0] == n:
                print(f"[{done[0]}/{n}] tok={agg['p']}/{agg['c']} err={agg['err']} "
                      f"截断题={agg['trunc']} 撞{args.max_calls}步={agg['hit_cap']} "
                      f"{time.time()-t0:.0f}s", flush=True)
        return rec

    if not args.latency_only:    # Block A: 并发跑全量acc/token(token隔离+acc确定→并发零污染)
        # 2026-07-08 断点续跑: 服务事故(vLLM wedge)会让整跑作废重来, 代价太大。已有无error预测行的题跳过,
        # predictions 改追加。逐题token在行内自带 → cell总账=全部行求和, 与单次整跑口径严格一致。
        _done_ids = set()
        _pf = f"{out_dir}/predictions.jsonl"
        if os.path.exists(_pf):
            import json as _j
            for _l in open(_pf):
                try:
                    _r = _j.loads(_l)
                    if not _r.get("error"):
                        _done_ids.add(_r["id"])
                except Exception:
                    pass
        if _done_ids:
            # Keep failed attempts in the append-only ledger: accuracy uses the
            # last successful row, while real query cost must include work spent
            # before a retry.  Dropping error rows here would under-report cost.
            _before = len(questions)
            questions = [q for q in questions if q["id"] not in _done_ids]
            print(f"[resume] 跳过已完成 {_before - len(questions)} 题, 剩 {len(questions)}", flush=True)
            n = len(questions)
        fout = open(_pf, "a" if os.path.exists(_pf) else "w")
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            list(ex.map(_work, questions))
        fout.close()
        # Resume-safe accounting: rebuild totals from the final prediction ledger, not only this invocation.
        final_rows = {}
        attempt_rows = []
        for line in open(_pf):
            try:
                row = json.loads(line)
                attempt_rows.append(row)
                if not row.get("error"):
                    final_rows[row["id"]] = row
            except Exception:
                pass
        # Charge every attempted row, including failed attempts later retried.
        final_p = sum(int(r.get("prompt_tok", 0) or 0) for r in attempt_rows)
        final_c = sum(int(r.get("completion_tok", 0) or 0) for r in attempt_rows)
        final_llm = sum(int(r.get("llm_calls", 0) or 0) for r in attempt_rows)
        final_tool = sum(int(r.get("tool_calls", 0) or 0) for r in attempt_rows)
        tot = token_meter.TokenMeter()
        if args.ranked_search:
            tot.set_bucket("build", 0, 0, 0)
        tot.set_bucket("qa", final_p, final_c, final_llm)
        note = (
            f"agent探索token,hook vLLM真实usage;max_calls={args.max_calls};"
            f"llm_calls={final_llm} tool_calls={final_tool};"
            f"attempt_rows={len(attempt_rows)} failed_attempts="
            f"{sum(bool(r.get('error')) for r in attempt_rows)}"
        )
        if args.ranked_search:
            note += f";native_BM25_top5;build_llm_tokens=0;build_sec={ranked_build_sec:.1f}"
        tot.dump(baseline_name, args.dataset, source="qwen_agent_hook", note=note)
        if args.ranked_search:
            with open(f"{out_dir}/experiment_config.json", "w") as cf:
                json.dump({
                    "baseline": baseline_name,
                    "dataset": args.dataset,
                    "sample_file": os.path.abspath(args.sample_file) if args.sample_file else "",
                    "n_questions": len(final_rows),
                    "attempt_rows": len(attempt_rows),
                    "failed_attempts": sum(bool(r.get("error")) for r in attempt_rows),
                    "bm25_top_k": 5,
                    "max_calls": args.max_calls,
                    "max_input_tokens": MAX_INPUT_TOKENS,
                    "reader_endpoint": LLM_CFG["model_server"],
                    "script_sha256": hashlib.sha256(open(__file__, "rb").read()).hexdigest(),
                    "build_llm_tokens": 0,
                    "build_sec": round(ranked_build_sec, 3),
                }, cf, ensure_ascii=False, indent=2)
        dt = time.time() - t0
        print(f"[done] {n}q {dt:.0f}s tok={agg['p']}/{agg['c']} err={agg['err']} "
              f"截断题={agg['trunc']} 撞上限题={agg['hit_cap']} llm_calls={agg['llm']} "
              f"tool_calls={agg['tool']} workers={args.workers} max_calls={args.max_calls}", flush=True)

    # ② latency: 并发段后对前N题串行(workers=1)重跑测纯净端到端延时(需独占vLLM)。
    # 不再 dump token(token 账已是上面并发口径); run_one 的 _TL 每题独立, 串行调用安全。
    if args.serial_sample:
        import statistics
        sqs = questions[:args.serial_sample]
        lat_dir = os.path.abspath(args.latency_out_dir) if args.latency_out_dir else f"{W}/results/latency"
        os.makedirs(lat_dir, exist_ok=True)
        lat_path = f"{lat_dir}/sandbox_{args.dataset}.jsonl"
        err_path = f"{lat_dir}/sandbox_{args.dataset}.errors.jsonl"
        wanted_ids = {q["id"] for q in sqs}
        completed = {}
        if args.latency_resume and os.path.exists(lat_path):
            extra_ids = set()
            for line in open(lat_path):
                try:
                    row = json.loads(line)
                    if row.get("id") in wanted_ids and not row.get("error"):
                        completed[row["id"]] = row
                    elif row.get("id") not in wanted_ids:
                        extra_ids.add(row.get("id"))
                except Exception:
                    pass
            if extra_ids:
                raise RuntimeError(
                    f"{lat_path} contains {len(extra_ids)} IDs outside the requested fixed sample"
                )
        pending = [q for q in sqs if q["id"] not in completed]
        print(
            f"[serial-latency] target={len(sqs)} completed={len(completed)} pending={len(pending)} "
            f"workers=1 endpoint={LLM_CFG['model_server']} out={lat_path}",
            flush=True,
        )
        mode = "a" if args.latency_resume and os.path.exists(lat_path) else "w"
        run_t0 = time.time()
        with open(lat_path, mode) as lf:
            for q in pending:
                r = run_one(q["question"], tree_root, filemap, ranked=ranked)
                rec = {
                    "id": q["id"],
                    "latency_sec": round(r["total_sec"], 3),
                    "llm_calls": r["llm_calls"],
                    "tool_calls": r["tool_calls"],
                    "error": r["error"],
                    "failure_type": r["failure_type"],
                    "endpoint": LLM_CFG["model_server"],
                    "max_calls": args.max_calls,
                }
                if r["error"]:
                    with open(err_path, "a") as ef:
                        ef.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        ef.flush()
                    raise RuntimeError(
                        f"latency probe failed for {q['id']}: {r['error']}; "
                        "successful rows are preserved for --latency_resume"
                    )
                completed[q["id"]] = rec
                lf.write(json.dumps(rec, ensure_ascii=False) + "\n")
                lf.flush()
                vals = [float(x["latency_sec"]) for x in completed.values()]
                p90 = float(np.quantile(vals, 0.90, method="linear"))
                print(
                    f"[serial-latency] {args.dataset} {len(completed)}/{len(sqs)} "
                    f"id={q['id']} sec={r['total_sec']:.2f} calls={r['llm_calls']} "
                    f"median={statistics.median(vals):.2f}s p90={p90:.2f}s "
                    f"run_elapsed={time.time()-run_t0:.0f}s",
                    flush=True,
                )
        lats = [float(completed[q["id"]]["latency_sec"]) for q in sqs]
        p10, p90 = np.quantile(lats, [0.10, 0.90], method="linear")
        print(
            f"[serial-latency] sandbox {args.dataset} n={len(lats)} "
            f"mean={statistics.mean(lats):.1f}s median={statistics.median(lats):.1f}s "
            f"p10={p10:.1f}s p90={p90:.1f}s",
            flush=True,
        )


if __name__ == "__main__":
    main()
