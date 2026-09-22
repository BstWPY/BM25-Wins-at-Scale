# BM25 Wins at Scale — Experiment Artifact

This repository contains the sanitized core experiment code and aggregate
result ledgers for **"BM25 Wins at Scale: A Scaling Study of
Retrieval-Augmented Generation Paradigms."** It supports result auditing and
reproduction of the corpus ladder, retrieval baselines, agent controls, token
accounting, judging, and paper figures.

The repository intentionally excludes model checkpoints, raw benchmark
documents, local caches, private endpoints, machine logs, and cluster-specific
orchestration scripts.

## External resources

For the paper's exact corpus document sets, start with
[`docs/reproduction.md`](docs/reproduction.md). Frozen bedrock/background IDs
now reproduce all 28 historical set checksums, and the 42,587-document tier
has been rebuilt and cross-checked against the unchanged original chunking
functions. Run `python scripts/frozen_ladder.py` for an offline ID audit.
The original experiment scripts below are preserved; the new tools avoid
re-mining or re-sampling the paper's ladder. See [`SECURITY.md`](SECURITY.md)
before running optional shell-agent controls.

Obtain the public EnterpriseRAG-Bench corpus and question set from the
release cited in the main paper.

The experiments used:

- Reader/build LLM: `Qwen3.6-27B`, served through an OpenAI-compatible API
- Embedding model: `Qwen3-Embedding-0.6B`, served through an OpenAI-compatible API
- Judge: the benchmark official judging prompt, run in one shared scoring
  session per experiment family/control

## Environment variables

Copy `.env.example` and set the paths and endpoints for your environment:

```bash
cp .env.example .env
# Edit .env, then:
set -a
source .env
set +a
```

The principal variables are:

```bash
export PROJECT_ROOT="<project-root>"
export URAG_ROOT="<experiment-root>"
export U_DATA_ROOT="<tier-data-root>"
export U_RESULTS_ROOT="<result-root>"
export ENTERPRISE_RAG_BENCH_ROOT="<benchmark-root>"
export MODEL_ROOT="<model-root>"
export U_LLM_ENDPOINT="<openai-compatible-llm-base-url>"
export U_EMBED_ENDPOINT="<openai-compatible-embedding-base-url>"
export GW_BASE="<judge-api-base-url>"
export GW_KEY="<judge-api-key>"
```

`GW_KEY` is required only for scripts that call an external judge.  The
included aggregate ledgers can be audited without a judge key.

## Core pipeline

The corpus-scaling experiment is reconstructed in the following order:

```bash
# 1. Render the public benchmark documents and build the corpus metadata table.
python scripts/build_corpus_meta.py

# 2. Mine hard traps/lures and audit answerable not-found candidates.
python scripts/mine_traps.py

# 3. Build the 28 strictly nested source×noise-stratified tiers.
python scripts/pack_corpus_enterprise.py --npoints 28

# 4. Compute corpus-token scale for every tier.
python scripts/compute_corpus_tokens.py
```

The native retrieval adapters are under `scripts/adapters/`. Each adapter uses
the shared contract in `scripts/unified_config.py`, including chunking,
retrieval budget, reader prompt, model endpoints, and deterministic generation
settings. Evaluation and official-judge aggregation are implemented in
`scripts/evaluate.py`, `scripts/judge_official.py`, and the corresponding
`finalize_*` scripts.

The GraphRAG-family packages and LinearRAG implementation should be installed
from the releases cited by the paper. They are not vendored here. The adapter
code records the integration points used in the experiments. Optional
method-specific and harness variables are documented in `.env.example`.

## Directory layout

- `scripts/`: sanitized pipeline, metering, judging, analysis, and plotting
  scripts.
- `scripts/mine_traps.py`: hard-negative and not-found-lure construction.
- `scripts/pack_corpus_enterprise.py`: nested 28-tier corpus construction and
  source/noise composition checks.
- `scripts/adapters/`: native retrieval and GraphRAG-family adapters.
- `scripts/sandbox_agent/`: raw-file agent harness and harness-control scripts.
- `scripts/graph_tools/`: read-only graph-substrate servers and graph-agent
  harness.
- `results/`: aggregate CSV/JSON outputs used by the paper tables, figures,
  token accounting, and matched controls.
- `data_manifests/`: aggregate tier-size and corpus-token manifests without
  raw chunk text.
- `figures/`: final paper figure PDFs, including the crossover schematic,
  and the cross-scale point ledger.

## Suggested audit order

1. Inspect `data_manifests/scale_corpus_tokens.json` and
   `data_manifests/tier_manifest_summary.json` to verify the ladder sizes.
2. Inspect `results/judge_official_canonical_20260724/canonical_matrix.csv`
   and `canonical_ci.csv` for the native-ladder accuracy tables and Figure 3.
   This canonical directory is the sole authoritative native-ladder matrix;
   superseded root-level accuracy matrices are intentionally omitted.
3. Inspect `results/build_fit_summary.csv` and the method-specific
   `results/*scaling_ledger.csv` files for construction scaling and Table 2.
4. Inspect `results/agent_query_cost_summary.csv` for the File-System Agent
   query-cost narrative and `results/query_latency_plot_summary.csv` for
   Figure 5. The phase-separated files under `results/token_usage/` and the
   token-accounting audit provide additional method-level checks.
5. Inspect `results/per_type_official_combined_N42587.csv` for Figure 6 and
   `results/harness_control_summary.json` for Table 5.
6. Inspect `results/judge_official_table3_20260724/` for the graph-substrate
   agent comparison.
7. Inspect `results/judge_retrieval_controls_20260726/summary.json`,
   `results/judge_agent_bm25_control/summary.json`, and
   `results/agent_bm25_control_report.md` for the retrieval-primitive controls.
8. Recreate the cross-scale accuracy--cost summary directly with
   `python scripts/teaser_crossscale.py`; it reads the included point ledger.
   Recomputing the ledger from omitted per-question outputs requires setting
   `TEASER_RECOMPUTE_FROM_RAW=1` and `URAG_ROOT` to a complete experiment tree.

The repository includes the final figure PDFs and the exact aggregate ledgers
used for the paper's quantitative plots. The cross-scale summary can also
be regenerated directly from its included point ledger. Raw benchmark
text, per-question free-form outputs, and full latency traces are omitted
for privacy and size; regenerating those aggregates requires the public
benchmark plus a complete experiment tree.

## Privacy and anonymization

The repository was scanned for:

- private tokens and API keys,
- local absolute paths,
- SSH host/user strings,
- private gateway domains,
- model/cache paths tied to a specific machine.

Only aggregate results and opaque question identifiers are included. Raw
benchmark documents, question/answer text, retrieved document paths, and
per-question free-form model outputs are excluded.
