#!/usr/bin/env python3
"""Validate and freeze the completed 68-cell canonical official judgment."""

import csv
import hashlib
import json
import os
import subprocess

import numpy as np


HERE = os.path.dirname(os.path.abspath(__file__))
PRED_ROOT = os.path.join(HERE, "results", "canonical_main_20260724")
JUDGE_ROOT = os.path.join(HERE, "results", "judge_official_canonical_20260724")
INPUT_MANIFEST = os.path.join(PRED_ROOT, "manifest.json")
FINAL_MANIFEST = os.path.join(JUDGE_ROOT, "final_manifest.json")
MATRIX_PATH = os.path.join(JUDGE_ROOT, "canonical_matrix.csv")
CI_PATH = os.path.join(JUDGE_ROOT, "canonical_ci.csv")
DISPLAY = {
    "bm25": "BM25",
    "naiverag": "DenseRAG",
    "sandbox": "File-System Agent",
    "hipporag": "HippoRAG2",
    "linearrag": "LinearRAG",
    "graphrag": "MS-GraphRAG",
    "lightrag": "LightRAG",
}
BOOTSTRAP_SEED = 20260724
BOOTSTRAP_REPS = 10000


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def latest_jsonl(path):
    rows = {}
    with open(path) as f:
        for line_no, line in enumerate(f, 1):
            row = json.loads(line)
            if row.get("id"):
                rows[row["id"]] = (row, line_no)
    return rows


def prediction_hash(row):
    payload = {
        "predicted_answer": row.get("predicted_answer") or "",
        "retrieved_chunk_ids": row.get("retrieved_chunk_ids") or [],
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True,
                     separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def bootstrap_ci(values, seed):
    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype=np.float64)
    means = np.empty(BOOTSTRAP_REPS, dtype=np.float64)
    chunk = 1000
    for i in range(0, BOOTSTRAP_REPS, chunk):
        take = min(chunk, BOOTSTRAP_REPS - i)
        idx = rng.integers(0, len(values), size=(take, len(values)))
        means[i:i + take] = values[idx].mean(axis=1)
    lo, hi = np.quantile(means, [0.025, 0.975])
    return round(float(lo), 3), round(float(hi), 3)


def main():
    input_manifest = json.load(open(INPUT_MANIFEST))
    if len(input_manifest["cells"]) != 68:
        raise RuntimeError("canonical input manifest does not contain 68 cells")
    matrix = []
    cis = []
    frozen_cells = []

    for cell_idx, cell in enumerate(input_manifest["cells"]):
        family = cell["family"]
        scale = int(cell["scale"])
        predictions = latest_jsonl(cell["prediction_path"])
        judge_path = os.path.join(JUDGE_ROOT, f"{family}_N{scale}.jsonl")
        if not os.path.exists(judge_path):
            raise FileNotFoundError(judge_path)
        judgments = latest_jsonl(judge_path)
        expected_ids = set(predictions)
        if len(expected_ids) != 500:
            raise RuntimeError(f"{family} N{scale}: prediction IDs={len(expected_ids)}")

        selected = []
        for qid in sorted(expected_ids):
            pred, _ = predictions[qid]
            expected_hash = prediction_hash(pred)
            item = judgments.get(qid)
            if item is None:
                raise RuntimeError(f"{family} N{scale}: missing judgment {qid}")
            row, _ = item
            if row.get("prediction_sha256") != expected_hash:
                raise RuntimeError(f"{family} N{scale}: stale judgment hash {qid}")
            if row.get("aligned") is None or row.get("completeness_pct") is None:
                raise RuntimeError(f"{family} N{scale}: incomplete judgment {qid}")
            selected.append(row)
        if set(judgments) != expected_ids:
            extra = sorted(set(judgments) - expected_ids)[:5]
            raise RuntimeError(f"{family} N{scale}: unexpected judgment IDs {extra}")

        scores = [
            float(r["completeness_pct"]) if int(r["aligned"]) else 0.0
            for r in selected
        ]
        correctness = 100.0 * sum(int(r["aligned"]) for r in selected) / 500
        completeness = sum(float(r["completeness_pct"]) for r in selected) / 500
        combined = sum(scores) / 500
        recalls = [float(r["doc_recall_pct"]) for r in selected
                   if r.get("doc_recall_pct") is not None]
        doc_recall = sum(recalls) / len(recalls) if recalls else None
        lo, hi = bootstrap_ci(scores, BOOTSTRAP_SEED + cell_idx)
        matrix.append({
            "family": family,
            "method": DISPLAY[family],
            "scale": scale,
            "n": 500,
            "combined": round(combined, 3),
            "correctness": round(correctness, 3),
            "completeness": round(completeness, 3),
            "doc_recall": round(doc_recall, 3) if doc_recall is not None else "",
            "doc_recall_n": len(recalls),
        })
        cis.append({
            "family": family, "method": DISPLAY[family], "scale": scale,
            "metric": "official_combined", "n": 500,
            "bootstrap_reps": BOOTSTRAP_REPS, "seed": BOOTSTRAP_SEED + cell_idx,
            "estimate": round(combined, 3), "ci95_low": lo, "ci95_high": hi,
        })
        frozen_cells.append({
            "family": family, "scale": scale,
            "prediction_path": cell["prediction_path"],
            "prediction_sha256": sha256_file(cell["prediction_path"]),
            "judgment_path": os.path.abspath(judge_path),
            "judgment_sha256": sha256_file(judge_path),
            "n": 500,
        })
        print(f"{family:<10} N{scale:<7} combined={combined:.3f} "
              f"CI=[{lo:.3f},{hi:.3f}]", flush=True)

    matrix.sort(key=lambda x: (x["family"], x["scale"]))
    cis.sort(key=lambda x: (x["family"], x["scale"]))
    with open(MATRIX_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(matrix[0]))
        writer.writeheader()
        writer.writerows(matrix)
    with open(CI_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(cis[0]))
        writer.writeheader()
        writer.writerows(cis)

    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=HERE, text=True).strip()
    except Exception:
        git_commit = None
    manifest = {
        "schema_version": 1,
        "input_manifest_path": os.path.abspath(INPUT_MANIFEST),
        "input_manifest_sha256": sha256_file(INPUT_MANIFEST),
        "judge_model": "deepseek-v4-flash",
        "judge_script_path": os.path.join(HERE, "judge_official.py"),
        "judge_script_sha256": sha256_file(os.path.join(HERE, "judge_official.py")),
        "finalizer_path": os.path.abspath(__file__),
        "finalizer_sha256": sha256_file(__file__),
        "aggregation": {
            "combined": "mean(completeness_pct if aligned else 0) over exactly 500",
            "doc_recall": "mean over answerable questions; null excluded",
            "bootstrap": {
                "method": "question-level nonparametric percentile",
                "reps": BOOTSTRAP_REPS,
                "base_seed": BOOTSTRAP_SEED,
            },
        },
        "git_commit": git_commit,
        "matrix_path": os.path.abspath(MATRIX_PATH),
        "matrix_sha256": sha256_file(MATRIX_PATH),
        "ci_path": os.path.abspath(CI_PATH),
        "ci_sha256": sha256_file(CI_PATH),
        "cells": frozen_cells,
    }
    with open(FINAL_MANIFEST, "w") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")
    print(f"[done] {FINAL_MANIFEST}", flush=True)


if __name__ == "__main__":
    main()
