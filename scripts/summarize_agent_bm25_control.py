#!/usr/bin/env python3
"""Strict audit and paired analysis for the Agent+BM25 control experiment."""
import json
import hashlib
import math
import os
import statistics
import sys
from collections import defaultdict

import numpy as np


W = os.environ.get("URAG_ROOT") or os.path.dirname(os.path.abspath(__file__))
SAMPLE_PATH = f"{W}/results/query_sample_150.json"
AGENT_JUDGE_DIR = f"{W}/results/judge_agent_bm25_control"
REFERENCE_JUDGE_DIR = f"{W}/results/judge_official_resweep"
OUT_JSON = f"{W}/results/agent_bm25_control_summary.json"
OUT_MD = f"{W}/results/agent_bm25_control_report.md"
SCALES = (1144, 511959)
BASELINES = ("bm25", "sandbox", "agent_bm25")


def load_last(path):
    rows = {}
    with open(path) as f:
        for line in f:
            try:
                row = json.loads(line)
                rows[row["id"]] = row
            except Exception:
                pass
    return rows


def load_all(path):
    rows = []
    with open(path) as f:
        for line in f:
            try:
                row = json.loads(line)
                if row.get("id"):
                    rows.append(row)
            except Exception:
                pass
    return rows


def prediction_sha256(row):
    payload = {
        "predicted_answer": row.get("predicted_answer") or "",
        "retrieved_chunk_ids": row.get("retrieved_chunk_ids") or [],
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def profile_jsonl(path, sample_ids, judge=False):
    parseable = []
    parse_errors = 0
    with open(path) as f:
        for line in f:
            try:
                parseable.append(json.loads(line))
            except Exception:
                parse_errors += 1
    last = {r.get("id"): r for r in parseable if r.get("id")}
    sample_set = set(sample_ids)
    profile = {
        "path": path,
        "rows": len(parseable),
        "parse_errors": parse_errors,
        "unique_ids": len(last),
        "duplicate_rows": len(parseable) - len(last),
        "sample_unique_ids": len(set(last) & sample_set),
        "missing_sample_ids": len(sample_set - set(last)),
        "out_of_sample_unique_ids": len(set(last) - sample_set),
    }
    if judge:
        selected = [last[qid] for qid in sample_ids if qid in last]
        profile["invalid_sample_judgments"] = sum(
            r.get("aligned") is None or r.get("completeness_pct") is None for r in selected
        )
    return profile


def require_ids(label, rows, sample_ids):
    got = set(rows) & set(sample_ids)
    missing = sorted(set(sample_ids) - got)
    if missing:
        raise RuntimeError(f"{label}: missing {len(missing)} sampled IDs; first={missing[:5]}")
    return {qid: rows[qid] for qid in sample_ids}


def judge_path(baseline, scale):
    return f"{AGENT_JUDGE_DIR}/{baseline}_N{scale}.jsonl"


def pred_path(baseline, scale):
    return f"{W}/results/{baseline}/enterprise_N{scale}/predictions.jsonl"


def score(row):
    return float(row["completeness_pct"]) if int(row["aligned"]) else 0.0


def metric_summary(rows):
    vals = list(rows.values())
    recalls = [float(r["doc_recall_pct"]) for r in vals if r.get("doc_recall_pct") is not None]
    return {
        "n": len(vals),
        "correctness": round(100.0 * statistics.mean(int(r["aligned"]) for r in vals), 2),
        "completeness": round(statistics.mean(float(r["completeness_pct"]) for r in vals), 2),
        "combined": round(statistics.mean(score(r) for r in vals), 2),
        "doc_recall": round(statistics.mean(recalls), 2) if recalls else None,
    }


def gold_hit_decomposition(rows):
    eligible = [r for r in rows.values() if r.get("doc_recall_pct") is not None]
    hit = [r for r in eligible if float(r["doc_recall_pct"]) > 0]
    miss = [r for r in eligible if float(r["doc_recall_pct"]) == 0]
    return {
        "eligible_n": len(eligible),
        "any_gold_document_hit_n": len(hit),
        "any_gold_document_hit_pct": round(100.0 * len(hit) / len(eligible), 2) if eligible else None,
        "combined_given_hit": round(statistics.mean(score(r) for r in hit), 2) if hit else None,
        "combined_given_miss": round(statistics.mean(score(r) for r in miss), 2) if miss else None,
    }


def pred_telemetry(rows, max_calls=80, attempt_rows=None):
    """Summarize final outcomes while charging every attempted run.

    Accuracy uses the last row per question, but a failed attempt followed by
    a successful retry still consumed real calls and tokens.  Aggregate usage
    by question across all attempts so the control's cost is not understated.
    """
    vals = list(rows.values())
    sample_ids = set(rows)
    attempts = (
        [r for r in attempt_rows if r.get("id") in sample_ids]
        if attempt_rows is not None else vals
    )
    usage = {
        qid: {"calls": 0, "tools": 0, "tokens": 0, "searches": 0, "truncated": False, "cap": False}
        for qid in rows
    }
    for r in attempts:
        qid = r["id"]
        calls = int(r.get("llm_calls", 1) or 0)
        usage[qid]["calls"] += calls
        usage[qid]["tools"] += int(r.get("tool_calls", 0) or 0)
        usage[qid]["tokens"] += int(r.get("total_tok", 0) or 0)
        usage[qid]["searches"] += len(r.get("search_trace") or [])
        usage[qid]["truncated"] |= int(r.get("trunc_rounds", 0) or 0) > 0
        usage[qid]["cap"] |= calls >= max_calls
    calls = [u["calls"] for u in usage.values()]
    tools = [u["tools"] for u in usage.values()]
    tokens = [u["tokens"] for u in usage.values()]
    searches = [u["searches"] for u in usage.values()]
    return {
        "mean_llm_calls": round(statistics.mean(calls), 2),
        "median_llm_calls": round(statistics.median(calls), 2),
        "mean_tool_calls": round(statistics.mean(tools), 2),
        "mean_total_tokens": round(statistics.mean(tokens), 2),
        "total_tokens": sum(tokens),
        "attempt_rows": len(attempts),
        "retry_attempts": len(attempts) - len(vals),
        "failed_attempts": sum(bool(r.get("error")) for r in attempts),
        "truncated_questions": sum(u["truncated"] for u in usage.values()),
        "truncated_pct": round(100.0 * sum(u["truncated"] for u in usage.values()) / len(vals), 2),
        "hit_call_cap": sum(u["cap"] for u in usage.values()),
        "hit_call_cap_pct": round(100.0 * sum(u["cap"] for u in usage.values()) / len(vals), 2),
        "empty_answers": sum(not (r.get("predicted_answer") or "").strip() for r in vals),
        "errors": sum(bool(r.get("error")) for r in vals),
        "invalid_tool_call_json_failures": sum(
            r.get("failure_type") == "invalid_tool_call_json" for r in vals
        ),
        "mean_searches": round(statistics.mean(searches), 2) if any(searches) else 0.0,
        "multi_search_pct": round(100.0 * sum(s > 1 for s in searches) / len(vals), 2) if any(searches) else 0.0,
    }


def native_bm25_tokens_per_q(scale):
    p = f"{W}/results/token_usage/BM25_enterprise_N{scale}.json"
    row = json.load(open(p))
    calls = int(row["qa"]["calls"])
    return round(float(row["qa"]["total_tokens"]) / calls, 2) if calls else None


def paired_bootstrap(agent, ref, sample_ids, seed):
    a = np.array([score(agent[qid]) for qid in sample_ids], dtype=float)
    b = np.array([score(ref[qid]) for qid in sample_ids], dtype=float)
    ca = np.array([int(agent[qid]["aligned"]) for qid in sample_ids], dtype=float) * 100.0
    cb = np.array([int(ref[qid]["aligned"]) for qid in sample_ids], dtype=float) * 100.0
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), size=(30000, len(a)))
    bd = (a - b)[idx].mean(axis=1)
    cd = (ca - cb)[idx].mean(axis=1)
    n10 = int(np.sum((ca == 100) & (cb == 0)))
    n01 = int(np.sum((ca == 0) & (cb == 100)))
    discordant = n10 + n01
    if discordant:
        tail = sum(math.comb(discordant, k) for k in range(min(n10, n01) + 1)) / (2 ** discordant)
        mcnemar_p = min(1.0, 2.0 * tail)
    else:
        mcnemar_p = 1.0
    return {
        "combined_delta_pp": round(float((a - b).mean()), 2),
        "combined_bootstrap_95ci": [round(float(x), 2) for x in np.percentile(bd, [2.5, 97.5])],
        "correctness_delta_pp": round(float((ca - cb).mean()), 2),
        "correctness_bootstrap_95ci": [round(float(x), 2) for x in np.percentile(cd, [2.5, 97.5])],
        "left_only_correct": n10,
        "right_only_correct": n01,
        "mcnemar_agent_only_correct": n10,
        "mcnemar_ref_only_correct": n01,
        "mcnemar_exact_p": round(mcnemar_p, 6),
    }


def type_breakdown(agent, ref, sample_ids):
    grouped = defaultdict(list)
    for qid in sample_ids:
        grouped[agent[qid].get("question_type", "NA")].append(qid)
    out = {}
    for qtype, ids in sorted(grouped.items()):
        av = [score(agent[qid]) for qid in ids]
        rv = [score(ref[qid]) for qid in ids]
        out[qtype] = {
            "n": len(ids),
            "agent_bm25_combined": round(statistics.mean(av), 2),
            "reference_combined": round(statistics.mean(rv), 2),
            "delta_pp": round(statistics.mean(a - b for a, b in zip(av, rv)), 2),
        }
    return out


def failure_cases(agent_judge, ref_judge, agent_pred, ref_pred, questions, sample_ids):
    def pack(qid):
        p = agent_pred[qid]
        trace = p.get("search_trace") or []
        return {
            "id": qid,
            "question_type": agent_judge[qid].get("question_type", "NA"),
            "question": questions[qid].get("question", ""),
            "gold_answer": questions[qid].get("answer", ""),
            "agent_answer": (p.get("predicted_answer") or "")[:3000],
            "reference_answer": (ref_pred[qid].get("predicted_answer") or "")[:3000],
            "agent_combined": score(agent_judge[qid]),
            "reference_combined": score(ref_judge[qid]),
            "agent_doc_recall": agent_judge[qid].get("doc_recall_pct"),
            "reference_doc_recall": ref_judge[qid].get("doc_recall_pct"),
            "llm_calls": p.get("llm_calls"),
            "searches": len(trace),
            "search_queries": [s.get("effective_query", "") for s in trace],
            "trunc_rounds": p.get("trunc_rounds"),
            "judge_reason": agent_judge[qid].get("reason", ""),
        }
    return {
        "agent_wrong_reference_right": [
            pack(qid) for qid in sample_ids
            if int(agent_judge[qid]["aligned"]) == 0 and int(ref_judge[qid]["aligned"]) == 1
        ],
        "agent_right_reference_wrong": [
            pack(qid) for qid in sample_ids
            if int(agent_judge[qid]["aligned"]) == 1 and int(ref_judge[qid]["aligned"]) == 0
        ],
    }


def main():
    sample_ids = json.load(open(SAMPLE_PATH))["question_ids"]
    if len(sample_ids) != 150 or len(set(sample_ids)) != 150:
        raise RuntimeError("sample manifest is not exactly 150 unique IDs")
    report = {
        "complete": False,
        "sample_file": SAMPLE_PATH,
        "sample_n": len(sample_ids),
        "provenance": {
            "sample_sha256": file_sha256(SAMPLE_PATH),
            "agent_script_sha256": file_sha256(f"{W}/sandbox_agent/run_sandbox_agent.py"),
            "judge_script_sha256": file_sha256(f"{W}/judge_official.py"),
            "summary_script_sha256": file_sha256(__file__),
        },
        "judge": {
            "script": f"{W}/judge_official.py",
            "model": "deepseek-v4-flash",
            "agent_output": AGENT_JUDGE_DIR,
            "reference_output": REFERENCE_JUDGE_DIR,
        },
        "token_cost_scope": {
            "agent_bm25": "exact usage across all attempted rows on the fixed 150-question sample, including failed attempts that were retried",
            "sandbox": "exact per-row usage on the same fixed 150-question sample",
            "bm25": "full-500 aggregate divided by 500; shown only as a descriptive cost reference",
        },
        "source_audit": {},
        "scales": {},
    }
    for scale in SCALES:
        questions = load_last(f"{W}/data/enterprise_N{scale}/questions.jsonl")
        preds = {}
        judged = {}
        for baseline in BASELINES:
            report["source_audit"][f"{baseline}_N{scale}_predictions"] = profile_jsonl(
                pred_path(baseline, scale), sample_ids
            )
            report["source_audit"][f"{baseline}_N{scale}_judge"] = profile_jsonl(
                judge_path(baseline, scale), sample_ids, judge=True
            )
            preds[baseline] = require_ids(
                f"{baseline} N{scale} predictions", load_last(pred_path(baseline, scale)), sample_ids
            )
            judged[baseline] = require_ids(
                f"{baseline} N{scale} judge", load_last(judge_path(baseline, scale)), sample_ids
            )
            bad = [
                qid for qid, r in judged[baseline].items()
                if r.get("aligned") is None or r.get("completeness_pct") is None
            ]
            if bad:
                raise RuntimeError(f"{baseline} N{scale}: {len(bad)} invalid judge rows")
            hash_mismatches = [
                qid for qid in sample_ids
                if judged[baseline][qid].get("prediction_sha256")
                != prediction_sha256(preds[baseline][qid])
            ]
            report["source_audit"][f"{baseline}_N{scale}_judge"][
                "prediction_hash_mismatches"
            ] = len(hash_mismatches)
            if hash_mismatches:
                raise RuntimeError(
                    f"{baseline} N{scale}: {len(hash_mismatches)} judgments do not match final predictions"
                )

        # The first ranked tool call must reproduce Native BM25's ordered top-5 for every question.
        exact = []
        for qid in sample_ids:
            trace = preds["agent_bm25"][qid].get("search_trace") or []
            native = preds["bm25"][qid].get("retrieved_chunk_ids") or []
            exact.append(bool(trace) and trace[0].get("chunk_ids") == native)
        if not all(exact):
            raise RuntimeError(f"agent_bm25 N{scale}: initial top-5 mismatch on {len(exact) - sum(exact)} questions")

        cell = {
            "metrics": {},
            "gold_hit_decomposition": {},
            "telemetry": {},
            "paired_comparisons": {},
            "judge_reconciliation": {},
        }
        for baseline in BASELINES:
            cell["metrics"][baseline] = metric_summary(judged[baseline])
            cell["gold_hit_decomposition"][baseline] = gold_hit_decomposition(judged[baseline])
        for baseline in ("bm25", "sandbox"):
            old_path = f"{REFERENCE_JUDGE_DIR}/{baseline}_N{scale}.jsonl"
            report["source_audit"][f"{baseline}_N{scale}_reference_resweep"] = profile_jsonl(
                old_path, sample_ids, judge=True
            )
            old_rows = require_ids(
                f"{baseline} N{scale} reference resweep", load_last(old_path), sample_ids
            )
            old_metric = metric_summary(old_rows)
            new_metric = cell["metrics"][baseline]
            cell["judge_reconciliation"][baseline] = {
                "same_session_combined": new_metric["combined"],
                "reference_resweep_combined": old_metric["combined"],
                "delta_pp": round(new_metric["combined"] - old_metric["combined"], 2),
            }
        cell["telemetry"]["bm25"] = {
            "mean_llm_calls": 1.0,
            "mean_total_tokens": native_bm25_tokens_per_q(scale),
            "initial_top5_exact_pct": 100.0,
        }
        cell["telemetry"]["sandbox"] = pred_telemetry(preds["sandbox"])
        cell["telemetry"]["agent_bm25"] = pred_telemetry(
            preds["agent_bm25"],
            attempt_rows=load_all(pred_path("agent_bm25", scale)),
        )
        cell["telemetry"]["agent_bm25"]["initial_top5_exact_pct"] = 100.0
        for i, ref in enumerate(("bm25", "sandbox")):
            comp = paired_bootstrap(judged["agent_bm25"], judged[ref], sample_ids, seed=20260723 + scale + i)
            comp["by_question_type"] = type_breakdown(judged["agent_bm25"], judged[ref], sample_ids)
            comp["cases"] = failure_cases(
                judged["agent_bm25"], judged[ref], preds["agent_bm25"], preds[ref], questions, sample_ids
            )
            cell["paired_comparisons"][f"agent_bm25_minus_{ref}"] = comp
        bm25_vs_sandbox = paired_bootstrap(
            judged["bm25"], judged["sandbox"], sample_ids, seed=20260723 + scale + 2
        )
        bm25_vs_sandbox["by_question_type"] = type_breakdown(
            judged["bm25"], judged["sandbox"], sample_ids
        )
        cell["paired_comparisons"]["bm25_minus_sandbox"] = bm25_vs_sandbox
        report["scales"][str(scale)] = cell

    report["complete"] = True
    with open(OUT_JSON, "w") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    lines = [
        "# Agent + BM25 ranked-search control",
        "",
        "Fixed stratified sample: 150 identical question IDs at each scale. "
        "The first Agent search is audited to equal Native BM25's ordered top-5 on every question.",
        "",
        "| N | Method | Combined | Correct | Complete | Doc recall | Any-gold hit | Mean calls | Mean tokens/q |",
        "|---:|:--|--:|--:|--:|--:|--:|--:|--:|",
    ]
    labels = {"bm25": "Native BM25", "sandbox": "File-System Agent", "agent_bm25": "Agent + BM25"}
    for scale in SCALES:
        cell = report["scales"][str(scale)]
        for baseline in BASELINES:
            m = cell["metrics"][baseline]
            h = cell["gold_hit_decomposition"][baseline]
            t = cell["telemetry"][baseline]
            lines.append(
                f"| {scale:,} | {labels[baseline]} | {m['combined']:.2f} | {m['correctness']:.2f} | "
                f"{m['completeness']:.2f} | {m['doc_recall']:.2f} | {h['any_gold_document_hit_pct']:.2f} | "
                f"{t.get('mean_llm_calls', 1):.2f} | {t.get('mean_total_tokens', 0):,.0f} |"
            )
    lines.append("")
    for scale in SCALES:
        cell = report["scales"][str(scale)]
        for ref in ("bm25", "sandbox"):
            c = cell["paired_comparisons"][f"agent_bm25_minus_{ref}"]
            lines.extend([
                f"- N={scale:,}, Agent+BM25 − {labels[ref]}: combined "
                f"{c['combined_delta_pp']:+.2f} pp, paired bootstrap 95% CI "
                f"[{c['combined_bootstrap_95ci'][0]:+.2f}, {c['combined_bootstrap_95ci'][1]:+.2f}]; "
                f"correctness {c['correctness_delta_pp']:+.2f} pp, McNemar p={c['mcnemar_exact_p']:.4g}.",
                "",
            ])
        c = cell["paired_comparisons"]["bm25_minus_sandbox"]
        lines.extend([
            f"- N={scale:,}, Native BM25 − File-System Agent: combined "
            f"{c['combined_delta_pp']:+.2f} pp, paired bootstrap 95% CI "
            f"[{c['combined_bootstrap_95ci'][0]:+.2f}, {c['combined_bootstrap_95ci'][1]:+.2f}]; "
            f"correctness {c['correctness_delta_pp']:+.2f} pp, McNemar p={c['mcnemar_exact_p']:.4g}.",
            "",
        ])
    lines.extend([
        "",
        "Cost scope: Agent+BM25 charges every attempted row, including failed attempts that were retried; "
        "File-System Agent uses exact per-row usage on the same fixed sample. Native BM25 token/q is the "
        "full-500 aggregate divided by 500 and is only a descriptive cost reference.",
        "",
        "Full telemetry, question-type breakdowns, and both directions of paired failure cases are in "
        "`agent_bm25_control_summary.json`.",
        "",
    ])
    with open(OUT_MD, "w") as f:
        f.write("\n".join(lines))
    print(json.dumps({
        "complete": True,
        "summary": OUT_JSON,
        "report": OUT_MD,
        "scales": {
            s: report["scales"][str(s)]["metrics"]["agent_bm25"] for s in SCALES
        },
    }, ensure_ascii=False))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"STRICT AUDIT FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        sys.exit(2)
