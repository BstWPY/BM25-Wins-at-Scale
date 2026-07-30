#!/usr/bin/env python3
"""Analyze why native BM25 can beat the iterative File-System Agent.

The script is intentionally read-only. It reconciles official judge rows with
prediction traces and writes a reviewer-facing audit under results/.
"""
import argparse
import json
import os
import statistics
from pathlib import Path


ROOT = Path(os.environ.get("URAG_ROOT") or Path(__file__).resolve().parent)
OUT = ROOT / "results" / "bm25_fs_failure_analysis_20260725"


def load_last_jsonl(path):
    rows = {}
    if not path.exists():
        return rows
    with path.open() as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except Exception:
                continue
            if row.get("id"):
                rows[row["id"]] = row
    return rows


def score(row):
    if not row:
        return None
    if int(row.get("aligned") or 0):
        return float(row.get("completeness_pct") or 0.0)
    return 0.0


def correct(row):
    return bool(row and int(row.get("aligned") or 0))


def doc_hit(row):
    val = None if not row else row.get("doc_recall_pct")
    return val is not None and float(val) > 0.0


def mean(vals):
    vals = [v for v in vals if v is not None]
    return round(statistics.mean(vals), 3) if vals else None


def pct(n, d):
    return round(100.0 * n / d, 2) if d else None


def analyze_scale(scale, sample_ids=None):
    bm25_j = load_last_jsonl(ROOT / "results" / "judge_official_resweep" / f"bm25_N{scale}.jsonl")
    fs_j = load_last_jsonl(ROOT / "results" / "judge_official_resweep" / f"sandbox_N{scale}.jsonl")
    bm25_p = load_last_jsonl(ROOT / "results" / "bm25" / f"enterprise_N{scale}" / "predictions.jsonl")
    fs_p = load_last_jsonl(ROOT / "results" / "sandbox" / f"enterprise_N{scale}" / "predictions.jsonl")
    ids = sorted(set(bm25_j) & set(fs_j) & set(bm25_p) & set(fs_p))
    if sample_ids:
        ids = [qid for qid in ids if qid in sample_ids]
    if not ids:
        return {"scale": scale, "status": "missing"}

    bm25_only = [qid for qid in ids if correct(bm25_j[qid]) and not correct(fs_j[qid])]
    fs_only = [qid for qid in ids if correct(fs_j[qid]) and not correct(bm25_j[qid])]
    both = [qid for qid in ids if correct(bm25_j[qid]) and correct(fs_j[qid])]
    neither = [qid for qid in ids if not correct(bm25_j[qid]) and not correct(fs_j[qid])]

    fs_calls = [int(fs_p[qid].get("llm_calls", 0) or 0) for qid in ids]
    fs_tools = [int(fs_p[qid].get("tool_calls", 0) or 0) for qid in ids]
    fs_tokens = [int(fs_p[qid].get("total_tok", 0) or 0) for qid in ids]
    fs_read_files = [len(fs_p[qid].get("read_files") or []) for qid in ids]
    fs_trunc = [int(fs_p[qid].get("trunc_rounds", 0) or 0) for qid in ids]

    bm25_win_decomp = {
        "n": len(bm25_only),
        "bm25_gold_doc_hit_pct": pct(sum(doc_hit(bm25_j[qid]) for qid in bm25_only), len(bm25_only)),
        "fs_gold_doc_hit_pct": pct(sum(doc_hit(fs_j[qid]) for qid in bm25_only), len(bm25_only)),
        "fs_search_miss_n": sum(not doc_hit(fs_j[qid]) for qid in bm25_only),
        "fs_hit_but_wrong_n": sum(doc_hit(fs_j[qid]) for qid in bm25_only),
        "fs_empty_answer_n": sum(not (fs_p[qid].get("predicted_answer") or "").strip() for qid in bm25_only),
        "fs_error_n": sum(bool(fs_p[qid].get("error")) for qid in bm25_only),
        "fs_truncated_n": sum(int(fs_p[qid].get("trunc_rounds", 0) or 0) > 0 for qid in bm25_only),
        "mean_fs_llm_calls": mean([int(fs_p[qid].get("llm_calls", 0) or 0) for qid in bm25_only]),
        "mean_fs_tool_calls": mean([int(fs_p[qid].get("tool_calls", 0) or 0) for qid in bm25_only]),
        "mean_fs_read_files": mean([len(fs_p[qid].get("read_files") or []) for qid in bm25_only]),
    }

    examples = []
    for qid in bm25_only[:12]:
        examples.append({
            "id": qid,
            "bm25_doc_recall": bm25_j[qid].get("doc_recall_pct"),
            "fs_doc_recall": fs_j[qid].get("doc_recall_pct"),
            "fs_llm_calls": fs_p[qid].get("llm_calls"),
            "fs_tool_calls": fs_p[qid].get("tool_calls"),
            "fs_read_files": fs_p[qid].get("read_files", [])[:3],
            "bm25_chunks": bm25_p[qid].get("retrieved_chunk_ids", [])[:5],
        })

    return {
        "scale": scale,
        "status": "ok",
        "n_common": len(ids),
        "combined": {
            "bm25": mean([score(bm25_j[qid]) for qid in ids]),
            "file_system_agent": mean([score(fs_j[qid]) for qid in ids]),
            "delta_bm25_minus_fs": round(
                mean([score(bm25_j[qid]) for qid in ids]) - mean([score(fs_j[qid]) for qid in ids]), 3),
        },
        "correctness_pct": {
            "bm25": pct(sum(correct(bm25_j[qid]) for qid in ids), len(ids)),
            "file_system_agent": pct(sum(correct(fs_j[qid]) for qid in ids), len(ids)),
        },
        "paired_correctness_counts": {
            "both_correct": len(both),
            "bm25_only_correct": len(bm25_only),
            "fs_only_correct": len(fs_only),
            "neither_correct": len(neither),
        },
        "retrieval_doc_hit_pct": {
            "bm25": pct(sum(doc_hit(bm25_j[qid]) for qid in ids), len(ids)),
            "file_system_agent": pct(sum(doc_hit(fs_j[qid]) for qid in ids), len(ids)),
        },
        "file_system_agent_telemetry": {
            "mean_llm_calls": mean(fs_calls),
            "mean_tool_calls": mean(fs_tools),
            "mean_total_tokens": mean(fs_tokens),
            "mean_read_files": mean(fs_read_files),
            "truncated_questions_pct": pct(sum(v > 0 for v in fs_trunc), len(ids)),
        },
        "bm25_win_decomposition": bm25_win_decomp,
        "bm25_win_examples": examples,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scales", default="1144,2254,6980,42587,131876,511959")
    parser.add_argument("--sample_file", default="")
    args = parser.parse_args()
    sample_ids = None
    if args.sample_file:
        sample_ids = set(json.load(open(args.sample_file))["question_ids"])
    OUT.mkdir(parents=True, exist_ok=True)
    scales = [int(x) for x in args.scales.split(",") if x]
    rows = [analyze_scale(scale, sample_ids=sample_ids) for scale in scales]
    manifest = {
        "schema_version": 1,
        "source_judgments": str(ROOT / "results" / "judge_official_resweep"),
        "source_predictions": str(ROOT / "results"),
        "sample_file": os.path.abspath(args.sample_file) if args.sample_file else "",
        "scales": rows,
    }
    (OUT / "summary.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    lines = ["# BM25 vs File-System Agent Failure Analysis", ""]
    for row in rows:
        if row["status"] != "ok":
            lines.append(f"- N={row['scale']}: missing aligned artifacts")
            continue
        lines.append(
            f"- N={row['scale']}: n={row['n_common']}, combined BM25={row['combined']['bm25']}, "
            f"FS={row['combined']['file_system_agent']}, delta={row['combined']['delta_bm25_minus_fs']} pp; "
            f"BM25-only correct={row['paired_correctness_counts']['bm25_only_correct']}, "
            f"FS-only correct={row['paired_correctness_counts']['fs_only_correct']}; "
            f"FS search-miss among BM25 wins={row['bm25_win_decomposition']['fs_search_miss_n']}, "
            f"FS hit-but-wrong={row['bm25_win_decomposition']['fs_hit_but_wrong_n']}."
        )
    (OUT / "report.md").write_text("\n".join(lines) + "\n")
    print(f"[done] {OUT / 'summary.json'}")


if __name__ == "__main__":
    main()
