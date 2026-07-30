#!/usr/bin/env python3
"""Six-tier fixed-sample cost--accuracy teaser.

The retriever-variant control (BM25, Qwen Dense, BGE-M3 Dense, and
BM25+Dense RRF) was run on one preregistered 150-question sample at six
scales.  The remaining systems are restricted to the same question IDs and
scales from the canonical official judge.  Unsupported graph tiers score
zero, while token cost is averaged over completed tiers.
"""

import csv
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter, LogLocator


HERE = os.path.dirname(os.path.abspath(__file__))
ARCHIVE_ROOT = os.path.dirname(HERE)
U = os.environ.get("URAG_ROOT", ARCHIVE_ROOT)
RESULTS = f"{U}/results"
TIERS = [1144, 2254, 6980, 42587, 131876, 511959]
CANONICAL_SOURCE = {
    "sandbox": "sandbox",
    "hipporag": "hipporag",
    "msgraphrag": "graphrag",
    "lightrag": "lightrag",
    "linearrag": "linearrag",
}
GRAPH_METHODS = ["hipporag", "msgraphrag", "lightrag", "linearrag"]
ORDER = [
    "bm25", "hybrid_rrf", "sandbox", "dense_bgem3", "denserag",
    *GRAPH_METHODS,
]

COLOR = {
    "bm25": "#92400e",
    "denserag": "#a21caf",
    "dense_bgem3": "#db2777",
    "hybrid_rrf": "#6d5a7d",
    "sandbox": "#0891b2",
    "hipporag": "#2563eb",
    "msgraphrag": "#1d4ed8",
    "lightrag": "#0ea5e9",
    "linearrag": "#6366f1",
}
NAME = {
    "bm25": "BM25",
    "denserag": "DenseRAG (Qwen)",
    "dense_bgem3": "DenseRAG (BGE-M3)",
    "hybrid_rrf": "BM25+Dense RRF",
    "sandbox": "File-System Agent",
    "hipporag": "HippoRAG 2",
    "msgraphrag": "MS-GraphRAG",
    "lightrag": "LightRAG",
    "linearrag": "LinearRAG",
}
METHOD_BY_NAME = {display_name: method for method, display_name in NAME.items()}
MARKER = {
    "bm25": "*",
    "denserag": "o",
    "dense_bgem3": "o",
    "hybrid_rrf": "o",
    "sandbox": "o",
    **{method: "o" for method in GRAPH_METHODS},
}
HALO = [pe.withStroke(linewidth=2.5, foreground="white")]


def _load_last_wins(path):
    seen = {}
    for line in open(path):
        row = json.loads(line)
        if row.get("aligned") is not None:
            seen[row["id"]] = row
    return seen


def load_accuracy():
    sample_manifest = json.load(open(f"{RESULTS}/query_sample_150.json"))
    sample_ids = set(sample_manifest["question_ids"])
    if sample_manifest["n"] != 150 or len(sample_ids) != 150:
        raise RuntimeError("fixed control sample is not exactly 150 unique IDs")

    control_rows = json.load(open(
        f"{RESULTS}/judge_retrieval_controls_20260726/summary.json"
    ))
    control = {}
    for row in control_rows:
        if row["scale"] not in TIERS:
            continue
        if row["n"] != 150 or row["fail"] != 0:
            raise RuntimeError(f"incomplete retrieval-control cell: {row}")
        control[(row["baseline"], row["scale"])] = float(row["combined"])
    control_source = {
        "bm25": "bm25",
        "denserag": "naiverag",
        "dense_bgem3": "dense_bgem3",
        "hybrid_rrf": "hybrid_rrf",
    }

    acc = {}
    coverage = {}
    sources = {}
    for method, source in control_source.items():
        values = [control[(source, n)] for n in TIERS]
        acc[method] = float(np.mean(values))
        coverage[method] = len(values)
        sources[method] = "matched six-scale retrieval-control rejudge"

    canonical_dir = (
        f"{RESULTS}/judge_official_canonical_20260724"
    )
    for method, source in CANONICAL_SOURCE.items():
        values = []
        completed = 0
        for n in TIERS:
            path = f"{canonical_dir}/{source}_N{n}.jsonl"
            if not os.path.exists(path):
                values.append(0.0)
                continue
            rows = _load_last_wins(path)
            selected = [
                rows[qid] for qid in sample_ids
                if qid in rows and rows[qid].get("completeness_pct") is not None
            ]
            if len(selected) != 150:
                raise RuntimeError(
                    f"{method} N={n}: expected 150 canonical judgments, "
                    f"found {len(selected)}"
                )
            scores = [
                row["completeness_pct"] if row["aligned"] == 1 else 0.0
                for row in selected
            ]
            values.append(float(np.mean(scores)))
            completed += 1
        acc[method] = float(np.mean(values))
        coverage[method] = completed
        sources[method] = "canonical official judge restricted to fixed 150 IDs"
    return acc, coverage, sources


def load_costs():
    costs = {method: [] for method in ORDER}
    dense_embed = {
        int(row["scale"]): float(row["document_embedding_input_tokens"])
        for row in csv.DictReader(open(
            f"{RESULTS}/dense_embedding_token_backfill_20260724/"
            "dense_embedding_tokens.csv"
        ))
    }
    linear_embed = {
        int(row["scale"]): float(row["embed_tokens"])
        for row in csv.DictReader(open(f"{RESULTS}/embed_token_backfill.csv"))
        if row["family"] == "linearrag"
    }
    linear_embed.setdefault(1144, dense_embed[1144] * 2.06)
    embedding_weight = 0.6 / 27.0

    for n in TIERS:
        for method, ledger in (("bm25", "BM25"), ("denserag", "NaiveRAG")):
            path = f"{RESULTS}/token_usage/{ledger}_enterprise_N{n}.json"
            total = json.load(open(path))["build_qa_total_tokens"]
            if method == "denserag":
                total += dense_embed[n] * embedding_weight
            costs[method].append(total)
        costs["sandbox"].append(sum(
            json.loads(line)["total_tok"]
            for line in open(
                f"{RESULTS}/sandbox/enterprise_N{n}/predictions.jsonl"
            )
        ))

    # The BGE cache was built once at the largest strict prefix.  Allocate its
    # audited full-cache embedding tokens to smaller prefixes in proportion to
    # the exact Qwen tokenizer backfill.  This preserves the observed BGE/Qwen
    # full-corpus tokenizer ratio without pretending that unlogged prefix
    # counters were directly measured.
    bge_progress = json.load(open(
        f"{RESULTS}/retrieval_controls_20260726/remote_progress/progress.json"
    ))
    if bge_progress["completed_rows"] != bge_progress["rows"]:
        raise RuntimeError("BGE embedding cache is incomplete")
    bge_ratio = (
        float(bge_progress["embedding_total_tokens"])
        / dense_embed[511959]
    )
    qa_scale = 500.0 / 150.0
    for n in TIERS:
        for method in ("dense_bgem3", "hybrid_rrf"):
            ledger = json.load(open(
                f"{RESULTS}/token_usage/{method}_enterprise_N{n}.json"
            ))
            if ledger["qa"]["calls"] != 150:
                raise RuntimeError(f"{method} N={n} does not have 150 calls")
            qa = ledger["qa"]["total_tokens"] * qa_scale
            if method == "dense_bgem3":
                embed = dense_embed[n] * bge_ratio
            else:
                embed = dense_embed[n]
            costs[method].append(qa + embed * embedding_weight)

    hippo_build = {
        int(row["N_docs"]): int(row["build_total_tok"])
        for row in csv.DictReader(open(f"{RESULTS}/hipporag_scaling_ledger.csv"))
        if row["build_total_tok"].isdigit()
    }
    hippo_qa_tiers = [1144, 2254, 6980, 21614, 42587, 131876]
    hippo_qa = {
        n: int(json.load(open(
            f"{RESULTS}/token_usage/HippoRAG_enterprise_N{n}.json"
        ))["qa"]["total_tokens"])
        for n in hippo_qa_tiers
        if os.path.exists(
            f"{RESULTS}/token_usage/HippoRAG_enterprise_N{n}.json"
        )
    }
    hippo_complete_qa = [
        value for n, value in hippo_qa.items() if n != 131876
    ]
    hippo_qa[131876] = int(float(np.median(hippo_complete_qa)))
    light_build = {
        int(row["N_docs"]): int(row["build_total_tok"])
        for row in csv.DictReader(open(f"{RESULTS}/lightrag_scaling_ledger.csv"))
        if row["build_total_tok"].isdigit()
    }
    for method in GRAPH_METHODS:
        source = CANONICAL_SOURCE[method]
        supported = [
            n for n in TIERS
            if os.path.exists(
                f"{RESULTS}/judge_official_canonical_20260724/"
                f"{source}_N{n}.jsonl"
            )
        ]
        for n in supported:
            if method == "hipporag":
                total = hippo_build[n] + hippo_qa[n]
            elif method == "msgraphrag":
                usage = json.load(open(
                    f"{RESULTS}/token_usage/graphrag_enterprise_N{n}.json"
                ))
                total = usage["build_qa_total_tokens"]
            elif method == "lightrag":
                usage = json.load(open(
                    f"{RESULTS}/token_usage/LightRAG_enterprise_N{n}.json"
                ))
                total = light_build[n] + usage["qa"]["total_tokens"]
            else:
                usage = json.load(open(
                    f"{RESULTS}/token_usage/LinearRAG_enterprise_N{n}.json"
                ))
                total = usage["qa"]["total_tokens"]
                if n == 1144:
                    total *= 5
                total += linear_embed[n] * embedding_weight
            costs[method].append(total)
    cost_sources = {
        "bm25": "six full-500 reader ledgers; no model-based construction",
        "denserag": (
            "six full-500 reader ledgers + exact Qwen embedding-token "
            "backfill at 0.6/27 weight"
        ),
        "dense_bgem3": (
            "six fixed-150 reader ledgers normalized to 500 + BGE full-cache "
            "tokens allocated by exact Qwen prefix ratios at 0.6/27 weight"
        ),
        "hybrid_rrf": (
            "six fixed-150 reader ledgers normalized to 500 + exact Qwen "
            "embedding-token backfill at 0.6/27 weight"
        ),
        "sandbox": "sum(total_tok) over six full-500 prediction ledgers",
        "hipporag": (
            "five completed build+QA ledgers; N=131876 QA median-imputed"
        ),
        "msgraphrag": "three completed build+QA ledgers",
        "lightrag": "two completed build+QA ledgers",
        "linearrag": (
            "five completed QA ledgers + audited embedding-token accounting"
        ),
    }
    return (
        {method: float(np.mean(values)) for method, values in costs.items()},
        cost_sources,
    )


def load_metrics():
    acc, coverage, sources = load_accuracy()
    costs, cost_sources = load_costs()
    return acc, costs, coverage, sources, cost_sources


def load_metrics_from_ledger(path):
    """Load the publication point ledger shipped in the review archive."""
    acc, cost, coverage, sources, cost_sources = {}, {}, {}, {}, {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            method = METHOD_BY_NAME[row["method"]]
            acc[method] = float(row["accuracy"])
            cost[method] = float(row["token_cost"])
            coverage[method] = int(row["covered_tiers"].split("/", 1)[0])
            sources[method] = row["accuracy_source"]
            cost_sources[method] = row["cost_source"]
    missing = set(ORDER) - set(acc)
    if missing:
        raise ValueError(f"Point ledger is missing methods: {sorted(missing)}")
    return acc, cost, coverage, sources, cost_sources


def main():
    ledger_path = os.path.join(
        ARCHIVE_ROOT, "figures", "teaser_crossscale_points.csv"
    )
    if (
        os.environ.get("TEASER_RECOMPUTE_FROM_RAW") != "1"
        and os.path.exists(ledger_path)
    ):
        acc, cost, coverage, sources, cost_sources = (
            load_metrics_from_ledger(ledger_path)
        )
    else:
        acc, cost, coverage, sources, cost_sources = load_metrics()
    plt.rcParams.update({
        "font.size": 7.5,
        "axes.labelsize": 7.5,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.2,
        "ytick.major.size": 2.2,
        "grid.linewidth": 0.4,
        "grid.alpha": 0.30,
        "grid.color": "#9aa0a6",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    })

    # Keep the teaser compact enough for AAAI's first-page title/abstract
    # block without shrinking its 7pt-scale text.
    fig, ax = plt.subplots(figsize=(3.35, 2.30))
    ax.set_xscale("log")
    ax.set_xlim(1.45e6, 5.0e8)
    ax.set_ylim(10, 80.5)
    ax.grid(True, axis="y")
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # A light halo singles out the best balance without introducing a rank formula.
    ax.scatter(
        [cost["bm25"]], [acc["bm25"]], s=230, marker="o",
        facecolor="#fef3c7", edgecolor="none", alpha=0.75, zorder=2,
    )

    for method in ORDER:
        size = 92 if method == "bm25" else 44
        ax.scatter(
            cost[method], acc[method], s=size, marker=MARKER[method],
            facecolor=COLOR[method], edgecolor="white",
            linewidth=0.65, zorder=4,
        )

    # Keep labels close to their points while separating the dense upper-left
    # cluster and the two low-accuracy GraphRAG points.
    offsets = {
    "bm25": (0, 10, "center", "bottom"),
    "hybrid_rrf": (10, 5, "center", "bottom"),
    "dense_bgem3": (0, 6, "center", "bottom"),
    "denserag": (0, 6, "center", "bottom"),
    "sandbox": (0, 6, "center", "bottom"),
    "hipporag": (0, 6, "center", "bottom"),
    "msgraphrag": (0, 10, "center", "bottom"),
    "lightrag": (-7, 5, "center", "bottom"),
    "linearrag": (0, 6, "center", "bottom"),
}
    for method in ORDER:
        dx, dy, ha, va = offsets[method]
        ax.annotate(
            NAME[method], (cost[method], acc[method]), xytext=(dx, dy),
            textcoords="offset points", ha=ha, va=va,
            fontsize=6.1,
            color=COLOR[method], fontweight="bold",
            path_effects=HALO, zorder=6,
        )

    ax.set_xlabel("Token Cost")
    ax.set_ylabel("Accuracy (%)")
    ax.xaxis.set_major_locator(LogLocator(base=10, numticks=4))
    ax.xaxis.set_major_formatter(
        FuncFormatter(lambda x, _: f"{x/1e6:g}M" if x >= 1e6 else f"{x:g}")
    )

    stem = os.environ.get("TEASER_STEM", "teaser_crossscale")
    out_dir = os.environ.get(
        "FIGURE_OUTPUT_DIR", os.path.join(ARCHIVE_ROOT, "figures")
    )
    os.makedirs(out_dir, exist_ok=True)
    out_pdf = f"{out_dir}/{stem}.pdf"
    out_png = f"{out_dir}/{stem}_preview.png"
    out_csv = f"{out_dir}/{stem}_points.csv"
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=200)

    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=[
                "method", "accuracy", "token_cost", "covered_tiers",
                "accuracy_source", "cost_source",
            ],
            lineterminator="\n",
        )
        writer.writeheader()
        for method in ORDER:
            writer.writerow({
                "method": NAME[method],
                "accuracy": f"{acc[method]:.6f}",
                "token_cost": f"{cost[method]:.0f}",
                "covered_tiers": f"{coverage[method]}/{len(TIERS)}",
                "accuracy_source": sources[method],
                "cost_source": cost_sources[method],
            })
    print("saved", out_pdf)
    print({
        method: {
            "accuracy": round(acc[method], 2),
            "cost_M": round(cost[method] / 1e6, 2),
            "coverage": f"{coverage[method]}/{len(TIERS)}",
        }
        for method in ORDER
    })


if __name__ == "__main__":
    main()
