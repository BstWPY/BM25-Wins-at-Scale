#!/usr/bin/env python3
"""Audit token accounting files used by the paper figures/tables."""
import csv
import json
import os
from pathlib import Path


ROOT = Path(os.environ.get("URAG_ROOT") or Path(__file__).resolve().parent)
TOKEN_DIR = ROOT / "results" / "token_usage"
OUT = ROOT / "results" / "token_accounting_audit_20260725"


def classify(row):
    baseline = str(row.get("baseline") or "")
    note = str(row.get("note") or "")
    build = int((row.get("build") or {}).get("total_tokens") or 0)
    qa = int((row.get("qa") or {}).get("total_tokens") or 0)
    flags = []
    interpretation = ""
    if baseline.startswith("BM25"):
        interpretation = "lexical index; no LLM or embedding build token"
    elif baseline.startswith("NaiveRAG"):
        interpretation = "dense embedding build; LLM build token intentionally zero"
    elif baseline.startswith("LinearRAG"):
        interpretation = "local spaCy/small-model extraction plus embedding; LLM API build token intentionally zero"
        flags.append("rename_or_caveat_linear_build_token")
    elif baseline.startswith("HippoRAG"):
        interpretation = "OpenIE graph construction uses LLM; build ledger is separate from QA-mode runs"
        if build == 0 and "QA-mode" in note:
            flags.append("qa_mode_build_zero_expected")
    elif baseline.startswith("LightRAG"):
        interpretation = "entity/relation extraction uses LLM during build"
    elif baseline.startswith("graphrag"):
        interpretation = "MS-GraphRAG indexing log parsed for build token"
        if qa == 0:
            flags.append("qa_token_hook_zero_check_needed")
    elif baseline.startswith("sandbox") or baseline.startswith("agent_bm25"):
        interpretation = "agent query-time token; build cost not applicable or lexical build zero"
    return interpretation, flags


def parse_name(path):
    name = path.stem
    if "_enterprise_N" not in name:
        return name, ""
    baseline, dataset = name.split("_enterprise_N", 1)
    return baseline, "enterprise_N" + dataset


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in sorted(TOKEN_DIR.glob("*.json")):
        try:
            row = json.loads(path.read_text())
        except Exception:
            continue
        interp, flags = classify(row)
        fallback_baseline, fallback_dataset = parse_name(path)
        rows.append({
            "path": str(path),
            "baseline": row.get("baseline") or fallback_baseline,
            "dataset": row.get("dataset") or fallback_dataset,
            "source": row.get("source") or "",
            "build_tokens": int((row.get("build") or {}).get("total_tokens") or 0),
            "qa_tokens": int((row.get("qa") or {}).get("total_tokens") or 0),
            "total_tokens": int(row.get("build_qa_total_tokens") or 0),
            "interpretation": interp,
            "flags": ";".join(flags),
            "note": row.get("note") or "",
        })
    csv_path = OUT / "token_accounting_audit.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["path"])
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "schema_version": 1,
        "token_dir": str(TOKEN_DIR),
        "n_files": len(rows),
        "flag_counts": {},
        "linear_rag_caveat": (
            "LinearRAG build LLM-token=0 is defensible only as API LLM-token accounting; "
            "the paper should state that local extractor/embedding compute is tracked outside LLM-token cost."
        ),
    }
    for row in rows:
        for flag in filter(None, row["flags"].split(";")):
            summary["flag_counts"][flag] = summary["flag_counts"].get(flag, 0) + 1
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(f"[done] {csv_path}")


if __name__ == "__main__":
    main()
