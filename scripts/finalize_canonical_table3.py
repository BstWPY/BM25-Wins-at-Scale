#!/usr/bin/env python3
"""Validate Table-3 judgments and emit the matched factorial table data."""

import csv
import hashlib
import json
import os
from collections import defaultdict


HERE = os.path.dirname(os.path.abspath(__file__))
PRED_ROOT = os.path.join(HERE, "results", "canonical_table3_20260724")
JUDGE_ROOT = os.path.join(HERE, "results", "judge_official_table3_20260724")
MANIFEST_IN = os.path.join(PRED_ROOT, "manifest.json")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def latest(path):
    rows = {}
    for line in open(path):
        row = json.loads(line)
        if row.get("id"):
            rows[row["id"]] = row
    return rows


def prediction_hash(row):
    payload = {"predicted_answer": row.get("predicted_answer") or "",
               "retrieved_chunk_ids": row.get("retrieved_chunk_ids") or []}
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True,
                     separators=(",", ":"))
    return hashlib.sha256(raw.encode()).hexdigest()


def main():
    source_manifest = json.load(open(MANIFEST_IN))
    scores = []
    frozen = []
    for cell in source_manifest["cells"]:
        family, scale, n = cell["family"], int(cell["scale"]), int(cell["n"])
        preds = latest(cell["prediction_path"])
        judge_path = os.path.join(JUDGE_ROOT, f"{family}_N{scale}.jsonl")
        judgments = latest(judge_path)
        if len(preds) != n or set(preds) != set(judgments):
            raise RuntimeError(
                f"{family} N{scale}: preds={len(preds)} judge={len(judgments)} n={n}")
        values = []
        for qid, pred in preds.items():
            row = judgments[qid]
            if row.get("prediction_sha256") != prediction_hash(pred):
                raise RuntimeError(f"{family} N{scale}: stale hash {qid}")
            if row.get("aligned") is None or row.get("completeness_pct") is None:
                raise RuntimeError(f"{family} N{scale}: incomplete {qid}")
            values.append(float(row["completeness_pct"]) if row["aligned"] else 0.0)
        score = sum(values) / n
        scores.append({
            "family": family, "substrate": cell["substrate"], "role": cell["role"],
            "scale": scale, "n": n, "official_combined": round(score, 3),
        })
        frozen.append({
            "family": family, "scale": scale, "n": n,
            "prediction_sha256": sha256_file(cell["prediction_path"]),
            "judgment_path": os.path.abspath(judge_path),
            "judgment_sha256": sha256_file(judge_path),
        })
        print(f"{family:<16} N{scale:<7} {score:.3f}", flush=True)

    score_path = os.path.join(JUDGE_ROOT, "canonical_table3_scores.csv")
    with open(score_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(scores[0]))
        writer.writeheader()
        writer.writerows(sorted(scores, key=lambda r: (r["scale"], r["substrate"], r["role"])))

    by_key = {(r["substrate"], r["role"], r["scale"]): r["official_combined"]
              for r in scores}
    rows = []
    scales = [1144, 1434, 1798, 2254, 6980, 21614, 42587]
    for scale in scales:
        row = {"scale": scale}
        row["Files"] = by_key.get(("Files", "agent", scale), "")
        for substrate in ("HippoRAG2", "LightRAG", "MS-GraphRAG"):
            agent = by_key.get((substrate, "agent", scale))
            native = by_key.get((substrate, "native", scale))
            row[substrate] = (
                f"{agent:.1f} ({native:.1f})"
                if agent is not None and native is not None else "")
        rows.append(row)
    table_path = os.path.join(JUDGE_ROOT, "canonical_table3_display.csv")
    with open(table_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["scale", "Files", "HippoRAG2", "LightRAG", "MS-GraphRAG"])
        writer.writeheader()
        writer.writerows(rows)
    final = {
        "schema_version": 1,
        "input_manifest_path": os.path.abspath(MANIFEST_IN),
        "input_manifest_sha256": sha256_file(MANIFEST_IN),
        "judge_model": "deepseek-v4-flash",
        "aggregation": "mean(completeness_pct if aligned else 0) on exact paired IDs",
        "scores_path": os.path.abspath(score_path),
        "scores_sha256": sha256_file(score_path),
        "display_path": os.path.abspath(table_path),
        "display_sha256": sha256_file(table_path),
        "cells": frozen,
    }
    final_path = os.path.join(JUDGE_ROOT, "final_manifest.json")
    with open(final_path, "w") as f:
        json.dump(final, f, indent=2)
        f.write("\n")
    print(f"[done] {final_path}", flush=True)


if __name__ == "__main__":
    main()
