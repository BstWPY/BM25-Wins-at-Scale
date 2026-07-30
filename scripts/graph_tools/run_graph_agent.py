#!/usr/bin/env python
"""agentic+graph runner(QwenAgent 形态) —— 与 sandbox runner 同 harness 同口径, 工具换成图 index server。

- 工具(全只读, 打 GRAPH_SERVER 的 HTTP): search_entities / search_facts / neighbors /
  ppr_chunks / retrieve(原生一轮式排序器) / read_chunk
- retrieved 口径: read_chunk 读过的 chunk -> 数据集 chunk_ids, 对齐 gold_chunk_ids 评 recall
  (与 sandbox 的 read_doc 口径对应: "读过"才算检索到)
- 计量: 与 run_sandbox_agent 相同的 vLLM usage hook + thread-local 逐题统计
- LLM: 默认 qwen3.6-27b（U_LLM_ENDPOINT，use_raw_api）；--llm gateway 时走 AI Gateway deepseek（冒烟用，
  fncall 由 qwen_agent 文本协议兜底), key 取 GW_KEY 环境变量

运行前设置 U_LLM_ENDPOINT 和 GRAPH_SERVER，或通过 --server 传入图工具服务地址。
"""
import os
import sys
import json
import time
import argparse
import threading
import urllib.request
import urllib.parse

W = os.environ.get("URAG_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, W)
import token_meter

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
    if int(getattr(u, "prompt_tokens", 0) or 0) >= _TL.trunc_thresh:
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

from qwen_agent.agents import Assistant                      # noqa: E402
from qwen_agent.agents import fncall_agent                   # noqa: E402
from qwen_agent.tools.base import BaseTool, register_tool    # noqa: E402

_srv_opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def _get(path, **params):
    url = f"{ARGS.server}{path}?{urllib.parse.urlencode(params)}"
    try:
        with _srv_opener.open(url, timeout=180) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        try:
            return json.loads(e.read())
        except Exception:
            return {"error": f"HTTP {e.code}"}
    except Exception as e:
        return {"error": str(e)}


def _arg(params, key, default=""):
    try:
        d = json.loads(params) if isinstance(params, str) else params
        if isinstance(d, dict):
            return d.get(key, default)
    except Exception:
        pass
    import re
    m = re.search(r'"%s"\s*:\s*"([^"]*)"' % re.escape(key), str(params))
    return m.group(1) if m else default


def _timed(fn):
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


def _record_chunks(payloads):
    """chunk payload -> 记入 retrieved(数据集 chunk_ids)。"""
    rf = getattr(_TL, "read_chunks", None)
    if rf is None:
        return
    for c in payloads:
        rf.update(int(x) for x in (c.get("chunk_ids") or []))


@register_tool("search_entities")
class SearchEntities(BaseTool):
    description = ("Semantic search over knowledge-graph entity nodes. "
                   "Returns matching entities with their graph degree and fact counts.")
    parameters = {"type": "object", "properties": {
        "query": {"type": "string"}, "k": {"type": "integer", "description": "default 8"}},
        "required": ["query"]}

    @_timed
    def call(self, params, **kwargs):
        d = _get("/search_entities", q=_arg(params, "query"), k=int(_arg(params, "k", 8) or 8))
        return json.dumps(d, ensure_ascii=False)


@register_tool("search_facts")
class SearchFacts(BaseTool):
    description = ("Semantic search over (subject, predicate, object) triplet facts extracted "
                   "from the corpus. Each fact lists its source chunk ids.")
    parameters = {"type": "object", "properties": {
        "query": {"type": "string"}, "k": {"type": "integer", "description": "default 8"}},
        "required": ["query"]}

    @_timed
    def call(self, params, **kwargs):
        d = _get("/search_facts", q=_arg(params, "query"), k=int(_arg(params, "k", 8) or 8))
        return json.dumps(d, ensure_ascii=False)


@register_tool("neighbors")
class Neighbors(BaseTool):
    description = ("Look up one entity in the knowledge graph: its outgoing/incoming facts "
                   "(with predicates and source chunks), synonym entities, and the chunks that mention it.")
    parameters = {"type": "object", "properties": {
        "entity": {"type": "string"}}, "required": ["entity"]}

    @_timed
    def call(self, params, **kwargs):
        d = _get("/neighbors", entity=_arg(params, "entity"), k=30)
        return json.dumps(d, ensure_ascii=False)


@register_tool("ppr_chunks")
class PprChunks(BaseTool):
    description = ("Run Personalized PageRank from seed entities (separate multiple seeds with |) "
                   "and return the top chunks. Good for multi-hop evidence gathering.")
    parameters = {"type": "object", "properties": {
        "seeds": {"type": "string", "description": "entity names separated by |"},
        "k": {"type": "integer", "description": "default 5"}}, "required": ["seeds"]}

    @_timed
    def call(self, params, **kwargs):
        d = _get("/ppr", seeds=_arg(params, "seeds"), k=int(_arg(params, "k", 5) or 5))
        return json.dumps(d, ensure_ascii=False)


@register_tool("retrieve")
class Retrieve(BaseTool):
    description = ("One-shot native retriever: link the query to top facts, seed PageRank, "
                   "return the best chunks with previews. A strong first move for most questions.")
    parameters = {"type": "object", "properties": {
        "query": {"type": "string"}, "k": {"type": "integer", "description": "default 5"}},
        "required": ["query"]}

    @_timed
    def call(self, params, **kwargs):
        d = _get("/retrieve", q=_arg(params, "query"), k=int(_arg(params, "k", 5) or 5))
        return json.dumps(d, ensure_ascii=False)


@register_tool("read_chunk")
class ReadChunk(BaseTool):
    description = "Read the full text of a chunk by its id. Counts as retrieving that chunk."
    parameters = {"type": "object", "properties": {
        "id": {"type": "string"}}, "required": ["id"]}

    @_timed
    def call(self, params, **kwargs):
        d = _get("/read_chunk", id=_arg(params, "id"))
        if "chunk_ids" in d:
            _record_chunks([d])
        txt = d.pop("text", "")
        d["text"] = txt[:8000]
        return json.dumps(d, ensure_ascii=False)


@register_tool("get_entity")
class GetEntity(BaseTool):
    description = ("Look up one entity: its description, typed relations to other entities "
                   "(with descriptions), and the chunks it appears in.")
    parameters = {"type": "object", "properties": {
        "name": {"type": "string"}}, "required": ["name"]}

    @_timed
    def call(self, params, **kwargs):
        d = _get("/get_entity", name=_arg(params, "name"))
        return json.dumps(d, ensure_ascii=False)


@register_tool("search_relations")
class SearchRelations(BaseTool):
    description = "Semantic search over entity-entity relations (edge descriptions/keywords)."
    parameters = {"type": "object", "properties": {
        "query": {"type": "string"}, "k": {"type": "integer", "description": "default 8"}},
        "required": ["query"]}

    @_timed
    def call(self, params, **kwargs):
        d = _get("/search_relations", q=_arg(params, "query"), k=int(_arg(params, "k", 8) or 8))
        return json.dumps(d, ensure_ascii=False)


@register_tool("search_reports")
class SearchReports(BaseTool):
    description = ("Semantic search over hierarchical community reports (thematic summaries of "
                   "entity clusters; level 0 = coarsest). Good for high-level/thematic questions.")
    parameters = {"type": "object", "properties": {
        "query": {"type": "string"}, "k": {"type": "integer", "description": "default 5"}},
        "required": ["query"]}

    @_timed
    def call(self, params, **kwargs):
        d = _get("/search_reports", q=_arg(params, "query"), k=int(_arg(params, "k", 5) or 5))
        return json.dumps(d, ensure_ascii=False)


@register_tool("read_report")
class ReadReport(BaseTool):
    description = "Read the full content of one community report by community id."
    parameters = {"type": "object", "properties": {
        "community": {"type": "integer"}}, "required": ["community"]}

    @_timed
    def call(self, params, **kwargs):
        d = _get("/read_report", community=int(_arg(params, "community", 0) or 0))
        return json.dumps(d, ensure_ascii=False)


@register_tool("search_chunks")
class SearchChunks(BaseTool):
    description = "Semantic search directly over corpus chunks."
    parameters = {"type": "object", "properties": {
        "query": {"type": "string"}, "k": {"type": "integer", "description": "default 5"}},
        "required": ["query"]}

    @_timed
    def call(self, params, **kwargs):
        d = _get("/search_chunks", q=_arg(params, "query"), k=int(_arg(params, "k", 5) or 5))
        return json.dumps(d, ensure_ascii=False)


MAX_INPUT_TOKENS = 26000
_COMMON = ("You are a research assistant answering questions over an enterprise document corpus. "
           "You cannot browse files; you query a pre-built index via tools. "
           "Chase conflicting or outdated versions carefully. "
           "Answer ONLY from chunk contents you have read. Be concise and factual. ")
PARADIGM = {
    "hippo": {
        "port": 8801,
        "tools": ["retrieve", "search_facts", "search_entities", "neighbors", "ppr_chunks", "read_chunk"],
        "sys": _COMMON + ("Index: an entity/fact knowledge graph (HippoRAG). Tools: retrieve "
                          "(one-shot ranked chunks), search_facts / search_entities (semantic), "
                          "neighbors (expand one entity), ppr_chunks (multi-hop PageRank from seeds), "
                          "read_chunk.")},
    "lightrag": {
        "port": 8802,
        "tools": ["retrieve", "search_entities", "search_relations", "get_entity", "read_chunk"],
        "sys": _COMMON + ("Index: an entity-relation graph with descriptions (LightRAG). Tools: "
                          "retrieve (one-shot ranked chunks), search_entities / search_relations "
                          "(semantic), get_entity (expand one entity), read_chunk.")},
    "graphrag": {
        "port": 8803,
        "tools": ["retrieve", "search_entities", "get_entity", "search_reports", "read_report",
                  "search_chunks", "read_chunk"],
        "sys": _COMMON + ("Index: an entity graph plus hierarchical community reports (MS GraphRAG). "
                          "Tools: retrieve (one-shot ranked chunks), search_entities / get_entity, "
                          "search_reports / read_report (thematic summaries, good for high-level "
                          "questions), search_chunks, read_chunk.")},
}


def llm_cfg():
    if ARGS.llm == "gateway":
        key = os.environ.get("GW_KEY", "")
        base = os.environ.get("GW_BASE", "").rstrip("/") + "/v1"
        return {"model": os.environ.get("GW_MODEL", "deepseek-v4-flash"),
                "model_server": base, "api_key": key,
                "generate_cfg": {"temperature": 0, "max_input_tokens": MAX_INPUT_TOKENS}}
    return {"model": "qwen3.6-27b",
            "model_server": os.environ["U_LLM_ENDPOINT"], "api_key": "EMPTY",
            "generate_cfg": {"temperature": 0, "use_raw_api": True,
                             "max_input_tokens": MAX_INPUT_TOKENS,
                             "extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}}


def run_one(question):
    _TL.meter = token_meter.TokenMeter()
    _TL.read_chunks = set()
    _TL.trunc_thresh = MAX_INPUT_TOKENS - 1500
    _TL.stat = {"llm_calls": 0, "llm_sec": 0.0, "tool_calls": 0, "tool_sec": 0.0, "trunc_rounds": 0}
    answer, err = "", ""
    t0 = time.time()
    try:
        P = PARADIGM[ARGS.paradigm]
        bot = Assistant(llm=llm_cfg(), function_list=P["tools"], system_message=P["sys"])
        last = None
        for resp in bot.run([{"role": "user", "content": question}]):
            last = resp
        for m in (last or []):
            if m.get("role") == "assistant" and m.get("content"):
                answer = m["content"]
    except Exception as e:
        err = f"{type(e).__name__}: {e}"
    st = _TL.stat
    qa = _TL.meter._acc["qa"]
    return {"answer": answer, "err": err, "retrieved": sorted(_TL.read_chunks),
            "total_sec": round(time.time() - t0, 2), **st,
            "prompt_tokens": qa["prompt_tokens"], "completion_tokens": qa["completion_tokens"]}


def main():
    ds = f"{W}/data/{ARGS.dataset}"
    questions = [json.loads(l) for l in open(f"{ds}/questions.jsonl")]
    if ARGS.sample_file:  # 2026-07-14: 分层子集(与 native 同题配对比较), 取代 limit_q 的前N偏差采样
        import json as _json
        _ids = set(_json.load(open(ARGS.sample_file))["question_ids"])
        questions = [q for q in questions if q["id"] in _ids]
        print(f"[sample] {len(questions)} questions from {ARGS.sample_file}", flush=True)
    elif ARGS.limit_q:
        questions = questions[:ARGS.limit_q]
    out_dir = f"{W}/results/graphagent_{ARGS.tag}/{ARGS.dataset}"
    os.makedirs(out_dir, exist_ok=True)
    outp = f"{out_dir}/predictions.jsonl"
    meterp = f"{out_dir}/meter.jsonl"
    done = set()
    if os.path.exists(outp):
        # Append-only ledgers may contain a failed attempt followed by a repair.
        # Only the latest attempt is eligible for resume, and API/program errors
        # must remain retryable instead of being silently treated as complete.
        pred_last = {}
        for l in open(outp):
            r = json.loads(l)
            if r.get("id"):
                pred_last[r["id"]] = r
        meter_last = {}
        if os.path.exists(meterp):
            for l in open(meterp):
                r = json.loads(l)
                if r.get("id"):
                    meter_last[r["id"]] = r
        done = {
            qid for qid, r in pred_last.items()
            if not r.get("error") and not (meter_last.get(qid, {}).get("err") or "")
        }
    todo = [q for q in questions if q["id"] not in done]
    print(f"[graphagent:{ARGS.tag}] {ARGS.dataset} llm={ARGS.llm} 待跑 {len(todo)} (已完成 {len(done)})", flush=True)
    fout = open(outp, "a")
    mout = open(meterp, "a")
    lock = threading.Lock()
    t0 = time.time()
    cnt = [0]

    def work(q):
        r = run_one(q["question"])
        with lock:
            prow = {"id": q["id"], "predicted_answer": r["answer"],
                    "retrieved_chunk_ids": r["retrieved"]}
            if r["err"]:
                prow["error"] = r["err"]
            fout.write(json.dumps(prow, ensure_ascii=False) + "\n")
            fout.flush()
            meta = {k: v for k, v in r.items() if k not in ("answer", "retrieved")}
            meta["id"] = q["id"]
            mout.write(json.dumps(meta, ensure_ascii=False) + "\n")
            mout.flush()
            cnt[0] += 1
            if cnt[0] % 5 == 0 or cnt[0] == len(todo):
                el = time.time() - t0
                print(f"  {cnt[0]}/{len(todo)}  {el:.0f}s  {cnt[0]/max(el,1):.2f}q/s", flush=True)

    import concurrent.futures as cf
    fncall_agent.MAX_LLM_CALL_PER_RUN = ARGS.max_calls
    with cf.ThreadPoolExecutor(ARGS.workers) as ex:
        list(ex.map(work, todo))
    fout.close(); mout.close()
    print("[done]", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="enterprise_N1144")
    ap.add_argument("--paradigm", default="hippo", choices=["hippo", "lightrag", "graphrag"])
    ap.add_argument("--server", default="")
    ap.add_argument("--tag", default="")
    ap.add_argument("--llm", default="vllm", choices=["vllm", "gateway"])
    ap.add_argument("--limit_q", type=int, default=0)
    ap.add_argument("--sample_file", default="", help="JSON含question_ids列表,只评这些题(绝对路径)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max_calls", type=int, default=80)
    ARGS = ap.parse_args()
    ARGS.tag = ARGS.tag or ARGS.paradigm
    ARGS.server = ARGS.server or os.environ.get("GRAPH_SERVER", "")
    if not ARGS.server:
        raise RuntimeError("Set GRAPH_SERVER or pass --server.")
    main()
