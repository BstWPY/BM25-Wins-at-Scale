#!/usr/bin/env python
"""按 EnterpriseRAG-Bench 官方 metrics_based_eval 口径重评已有预测(走 AI Gateway)。

官方口径(src/scripts/answer_evaluation/metrics_based_eval.py::score_answer):
  - answer_correct: ANSWER_WHOLISTIC_EVALUATION_PROMPT 整体对齐判定(query+gold+candidate -> aligned yes/no)
  - completeness_pct: 每条 answer_facts 一次 INDIVIDUAL_FACT_VALIDATOR_PROMPT, 通过比例
  - document_recall_pct: retrieved dsids ∩ gold_dsids / gold_dsids (纯集合, 无LLM)
  - combined = mean(completeness if correct else 0)
与官方的注记差异(报告须写明):
  ① judge 模型用网关 deepseek-v4-flash(官方用其默认LLM), 各家一致故内部可比;
  ② 引用剥离用正则轻量版(官方是LLM剥离), 我们的预测几乎无引用格式, 影响可忽略;
  ③ 无 valid_doc_ids(官方活数据集维护产物), extra_docs 报原始计数而非 invalid_extra_docs;
  ④ 不启用官方"gold 集动态更新"路径 —— 固定 gold 才能跨家/跨档对比。
逐题增量写盘+断点续跑(aligned/completeness 均非 null 视为已完成)。
跑: GW_KEY=<judge-api-key> GW_BASE=<judge-base-url> python3 judge_official.py \
      --baselines bm25,naiverag,hipporag,graphrag,lightrag,linearrag,sandbox [--scales ...] [--limit N]
"""
import os, sys, json, glob, time, argparse, random, re, statistics, threading, hashlib
import urllib.request, urllib.error
import concurrent.futures as cf
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import unified_config as U

W = os.environ.get("URAG_ROOT") or os.path.dirname(os.path.abspath(__file__))
BASE = os.environ.get("GW_BASE", "").rstrip("/")
KEY = os.environ.get("GW_KEY", "")
if os.environ.get("GW_PROXY") == "1":       # SG 端点须经代理出口(直连被判大陆IP拒绝)
    _opener = urllib.request.build_opener()  # 用系统 http(s)_proxy
else:
    for v in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(v, None)
    _opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

# ---- 全局实时 token 计费(线程安全, 精确整数, 账本落盘可跨重启累计); --budget_tokens 熔断 ----
_tok_lock = threading.Lock()
_ledger_io_lock = threading.Lock()
TOK = {"prompt": 0, "completion": 0, "reasoning": 0, "calls": 0}
_STOP = threading.Event()          # 预算耗尽 -> 全局停发新任务


def _ledger_path():
    return f"{W}/results/{ARGS.out}/tokens_ledger.json"


def ledger_load():
    p = _ledger_path()
    if os.path.exists(p):
        try:
            d = json.load(open(p))
            for k in TOK:
                TOK[k] = int(d.get(k, 0))
            print(f"[ledger] 载入既有账本: {tok_line()}", flush=True)
        except Exception as e:
            print(f"[ledger] 读取失败(从0计): {e}", flush=True)


def ledger_save():
    p = _ledger_path()
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with _tok_lock:
        d = dict(TOK, model=ARGS.model, updated_at=time.strftime("%Y-%m-%d %H:%M:%S"))
    with _ledger_io_lock:
        tmp = p + ".tmp"
        json.dump(d, open(tmp, "w"), indent=1)
        os.replace(tmp, p)                  # 原子替换


def _account(usage):
    if not usage:
        return
    with _tok_lock:
        TOK["prompt"] += int(usage.get("prompt_tokens", 0) or 0)
        TOK["completion"] += int(usage.get("completion_tokens", 0) or 0)
        det = usage.get("completion_tokens_details") or {}
        TOK["reasoning"] += int(det.get("reasoning_tokens", 0) or 0)
        TOK["calls"] += 1
        n = TOK["calls"]
    if n % 200 == 0:
        ledger_save()
    if ARGS.budget_tokens and (TOK["prompt"] + TOK["completion"]) >= ARGS.budget_tokens:
        _STOP.set()


def tok_line():
    t = TOK["prompt"] + TOK["completion"]
    return (f"tok(in={TOK['prompt']:,} out={TOK['completion']:,} "
            f"reason={TOK['reasoning']:,} 总={t:,} calls={TOK['calls']:,})")


def over_budget():
    return _STOP.is_set()

sys.path.insert(0, f"{os.path.dirname(W)}/EnterpriseRAG-Bench")
from src.prompts.answer_evaluation import (          # noqa: E402  官方prompt原文直用
    ANSWER_WHOLISTIC_EVALUATION_PROMPT, INDIVIDUAL_FACT_VALIDATOR_PROMPT)

_CITE = re.compile(r"\[\d+\](\(\S+\))?")             # 轻量引用剥离(注记差异②)


def call_llm(prompt, max_tokens=1024, max_tries=5):
    """成功但解析不出 → None(temp=0 重试无意义); 仅瞬时错误退避。
    deepseek 是推理模型: reasoning 吃满预算时 finish=length 且 content 空 —— temp=0 下同前缀
    确定性复现, 换更大 max_tokens 重试一次可收尾; 仍空则从 reasoning 尾部兜底取结论。"""
    def once(mt):
        payload = {"model": ARGS.model, "messages": [{"role": "user", "content": prompt}]}
        if ARGS.model.startswith(("gpt-5", "o1", "o3", "o4")):
            payload["max_completion_tokens"] = mt      # OpenAI 推理系: 不接受 temperature=0/max_tokens
        else:
            payload["temperature"] = 0
            payload["max_tokens"] = mt
        req = urllib.request.Request(f"{BASE}/v1/chat/completions", data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json",
                                              "Authorization": f"Bearer {KEY}"})
        r = json.load(_opener.open(req, timeout=90))
        _account(r.get("usage") or {})
        ch = r["choices"][0]
        msg = ch["message"]
        return ((msg.get("content") or "").strip(), ch.get("finish_reason"),
                (msg.get("reasoning_content") or "").strip())
    for attempt in range(max_tries):
        try:
            out, fin, reas = once(max_tokens)
            if not out and fin == "length":
                out, fin, reas = once(max_tokens * 4)     # 推理截断: 放大预算确定性收尾
            if not out and reas:
                return "[[reasoning-tail]] " + reas[-400:]  # 兜底: 结论常在推理末尾
            return out
        except urllib.error.HTTPError as e:
            if e.code not in (429, 500, 502, 503, 504):
                return None
        except Exception:
            pass
        time.sleep(min(1.6 ** attempt + random.random(), 8))
    return None


def eval_wholistic(q, gold, pred):
    out = call_llm(ANSWER_WHOLISTIC_EVALUATION_PROMPT.format(
        query=q, gold_answer=gold, candidate_answer=pred))
    if not out:
        return None, ""
    reason = ""
    m = re.search(r'"reason"\s*:\s*"([^"]*)"', out)
    if m:
        reason = m.group(1)
    m = re.search(r'"aligned"\s*:\s*"?\s*(yes|no)\b', out, re.I)
    if m:
        return int(m.group(1).lower() == "yes"), reason
    # 官方兜底: 全文找独立 yes/no
    m = re.search(r"\b(yes|no)\b", out, re.I)
    return (int(m.group(1).lower() == "yes"), reason) if m else (None, "")


def eval_fact(pred, statement):
    out = call_llm(INDIVIDUAL_FACT_VALIDATOR_PROMPT.format(
        answer=pred, statement=statement), max_tokens=1024)
    if not out:
        return None
    m = re.search(r"\b(yes|no)\b", out, re.I)
    return int(m.group(1).lower() == "yes") if m else None


_wlock = threading.Lock()


def prediction_sha256(row):
    """Bind a cached judgment to the exact answer and retrieved evidence."""
    payload = {
        "predicted_answer": row.get("predicted_answer") or "",
        "retrieved_chunk_ids": row.get("retrieved_chunk_ids") or [],
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def judge_cell(b, scale, predpath):
    ds = f"{U.DATA_ROOT}/enterprise_N{scale}"
    gold = {q["id"]: q for q in (json.loads(l) for l in open(f"{ds}/questions.jsonl"))}
    c2d = {}   # chunk_id -> dsid (评官方 doc recall)
    for l in open(f"{ds}/chunks_meta.jsonl"):
        m = json.loads(l)
        c2d[int(m["chunk_id"])] = m["dsid"]
    # Prediction ledgers are append-only across retries. Use the last
    # successful row per ID and never score an API/program-error placeholder.
    sample_ids = None
    if ARGS.sample_file:
        sample_ids = set(json.load(open(ARGS.sample_file))["question_ids"])
    pred_by_id = {}
    for l in open(predpath):
        p = json.loads(l)
        if p.get("id") and not p.get("error"):
            pred_by_id[p["id"]] = p
    if ARGS.strict_expected_rows:
        expected_ids = (sample_ids & set(gold)) if sample_ids is not None else set(gold)
        actual_ids = set(pred_by_id) & expected_ids
        if len(actual_ids) != ARGS.strict_expected_rows or actual_ids != expected_ids:
            missing = sorted(expected_ids - actual_ids)[:5]
            raise RuntimeError(
                f"{b} N{scale}: strict prediction check failed: "
                f"matched_rows={len(actual_ids)} expected={ARGS.strict_expected_rows}, "
                f"missing={missing}"
            )
    preds = list(pred_by_id.values())
    if sample_ids is not None:
        preds = [p for p in preds if p["id"] in sample_ids]
    if ARGS.limit:
        preds = preds[:ARGS.limit]

    os.makedirs(f"{W}/results/{ARGS.out}", exist_ok=True)
    outp = f"{W}/results/{ARGS.out}/{b}_N{scale}.jsonl"
    pred_hashes = {p["id"]: prediction_sha256(p) for p in preds}
    done = set()
    if os.path.exists(outp):
        for l in open(outp):
            try:
                r = json.loads(l)
                if (
                    r.get("aligned") is not None
                    and r.get("completeness_pct") is not None
                    and r.get("prediction_sha256") == pred_hashes.get(r.get("id"))
                ):
                    done.add(r["id"])
            except Exception:
                pass
    todo = [p for p in preds if p["id"] in gold and p["id"] not in done]
    print(f"  {b} N{scale}: 待评 {len(todo)} (已完成 {len(done)})", flush=True)
    if not todo:
        return summarize(b, scale, outp)
    fout = open(outp, "a")
    t0 = time.time(); cnt = [0]

    def work(p):
        if _STOP.is_set():
            return                          # 预算耗尽: 不再开工新题, 已写盘的不受影响
        q = gold[p["id"]]
        pred = _CITE.sub("", (p.get("predicted_answer") or ""))[:6000]
        facts = q.get("answer_facts") or []
        if pred.strip():
            aligned, reason = eval_wholistic(q["question"], q["answer"], pred)
            fr = [eval_fact(pred, s) for s in facts]
            ok = [x for x in fr if x is not None]
            comp = (100.0 * sum(ok) / len(facts)) if facts and len(ok) == len(facts) \
                else (100.0 if not facts else None)   # 有fact判定失败 → completeness记None
        else:
            aligned, reason, comp, fr = 0, "empty prediction", 0.0, []
        rset = {c2d.get(int(c)) for c in (p.get("retrieved_chunk_ids") or []) if int(c) in c2d}
        rset.discard(None)
        gset = set(q.get("gold_dsids") or [])
        recall = round(100.0 * len(rset & gset) / len(gset), 2) if gset else None
        extra = len(rset - gset) if gset else None
        with _wlock:
            fout.write(json.dumps({
                "id": p["id"], "question_type": q.get("question_type", "NA"),
                "aligned": aligned, "reason": reason[:300],
                "completeness_pct": comp, "n_facts": len(facts),
                "doc_recall_pct": recall, "extra_docs": extra,
                "prediction_sha256": pred_hashes[p["id"]]}) + "\n")
            fout.flush()
            cnt[0] += 1
            if cnt[0] % 25 == 0 or cnt[0] == len(todo):
                el = time.time() - t0
                print(f"    {b} N{scale}: {cnt[0]}/{len(todo)}  {el:.0f}s  "
                      f"{cnt[0]/max(el,1):.2f}q/s  {tok_line()}", flush=True)

    per_cell_workers = max(1, ARGS.workers // max(1, ARGS.cell_workers))
    with cf.ThreadPoolExecutor(per_cell_workers) as ex:
        list(ex.map(work, todo))
    fout.close()
    ledger_save()
    if over_budget():
        print(f"[STOP] token 预算 {ARGS.budget_tokens:,} 已耗尽, 主动熔断并保存账本: {tok_line()}", flush=True)
        summarize(b, scale, outp)
        sys.exit(2)
    return summarize(b, scale, outp)


def summarize(b, scale, outp):
    if not os.path.exists(outp):
        print(f"  [no-results] {b} N{scale}: no judgment file at {outp}", flush=True)
        return None
    rows = [json.loads(l) for l in open(outp)]
    seen = {}
    for r in rows:                                   # 续跑可能重复, 取最后一次
        seen[r["id"]] = r
    rows = [r for r in seen.values() if r["aligned"] is not None and r["completeness_pct"] is not None]
    if not rows:
        return None
    corr = 100.0 * sum(r["aligned"] for r in rows) / len(rows)
    comp = statistics.mean(r["completeness_pct"] for r in rows)
    comb = statistics.mean(r["completeness_pct"] if r["aligned"] else 0.0 for r in rows)
    rec = [r["doc_recall_pct"] for r in rows if r["doc_recall_pct"] is not None]
    res = dict(baseline=b, scale=scale, n=len(rows), correctness=round(corr, 2),
               completeness=round(comp, 2), combined=round(comb, 2),
               doc_recall=round(statistics.mean(rec), 2) if rec else None,
               fail=len(seen) - len(rows))
    print(f"  ✓ {b:<10} N{scale:<7} corr={res['correctness']:<6} comp={res['completeness']:<6} "
          f"combined={res['combined']:<6} recall={res['doc_recall']} n={res['n']} fail={res['fail']}", flush=True)
    return res


def main():
    if not KEY:
        sys.exit("✗ 未设置 GW_KEY")
    PREDNAME = {
        "lightrag": "predictions_hybrid.jsonl",
        "lightrag_k10": "predictions_hybrid.jsonl",
        "graphrag": "predictions_local.jsonl",
        "graphrag_para": "predictions_local.jsonl",
    }
    scales = [int(s) for s in ARGS.scales.split(",")] if ARGS.scales else None
    cells = []
    pred_root = os.path.abspath(ARGS.pred_root) if ARGS.pred_root else f"{W}/results"
    for b in ARGS.baselines.split(","):
        for p in sorted(glob.glob(
                f"{pred_root}/{b}/enterprise_N*/{PREDNAME.get(b, 'predictions.jsonl')}")):
            suffix = os.path.basename(os.path.dirname(p)).split("_N")[1]
            if not suffix.isdigit():
                continue                      # 跳过 _updq 等变体数据集
            sc = int(suffix)
            if scales and sc not in scales:
                continue
            if sum(1 for _ in open(p)) < ARGS.min_rows:
                print(f"  [skip-stale] {b} N{sc}"); continue
            cells.append((b, sc, p))
    print(f"[judge_official] model={ARGS.model} cells={len(cells)} total_workers={ARGS.workers} "
          f"cell_workers={ARGS.cell_workers} "
          f"budget={ARGS.budget_tokens:,}", flush=True)
    ledger_load()
    ordered = sorted(cells, key=lambda x: (x[0], x[1]))
    if ARGS.cell_workers > 1:
        with cf.ThreadPoolExecutor(ARGS.cell_workers) as ex:
            res = list(ex.map(lambda x: judge_cell(*x), ordered))
    else:
        res = [judge_cell(*x) for x in ordered]
    json.dump([r for r in res if r], open(f"{W}/results/{ARGS.out}/summary.json", "w"), indent=1)
    ledger_save()
    print(f"[done] 最终账本: {tok_line()}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--baselines", default="bm25,naiverag,hipporag,graphrag,lightrag,linearrag,sandbox")
    ap.add_argument("--scales", default="")
    ap.add_argument("--model", default="deepseek-v4-flash")
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--cell_workers", type=int, default=1,
                    help="parallel cells; --workers is divided across them")
    ap.add_argument("--out", default="judge_official")
    ap.add_argument("--min_rows", type=int, default=470, help="predictions行数低于此值跳过(抽样cell用较小值)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--sample_file", default="",
                    help="JSON含question_ids列表; each cell is judged on exactly these IDs")
    ap.add_argument("--pred_root", default="",
                    help="prediction snapshot root (default: results/)")
    ap.add_argument("--strict_expected_rows", type=int, default=0,
                    help="require exactly this many unique prediction IDs and exact equality with gold IDs")
    ap.add_argument("--budget_tokens", type=int, default=0, help="in+out 总 token 硬熔断(0=不限)")
    ARGS = ap.parse_args()
    main()
