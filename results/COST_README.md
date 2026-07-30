# Construction and Query Cost Sources

The paper uses the following aggregate ledgers as authoritative sources:

- Graph-family construction scaling:
  `graphrag_scaling_ledger.csv`, `hipporag_scaling_ledger.csv`,
  `lightrag_scaling_ledger.csv`, and `linearrag_scaling_ledger.csv`.
- Per-method phase-separated token usage: `token_usage/`.
- Token-accounting cross-check:
  `token_accounting_audit_20260725/token_accounting_audit.csv`.
- Dense embedding-token backfill:
  `dense_embedding_token_backfill_20260724/dense_embedding_tokens.csv`.
- Table 2 fitted exponents and rounded feasibility estimates:
  `build_fit_summary.csv`.
- File-System Agent query-cost aggregates:
  `agent_query_cost_summary.csv`.
- Figure 5 plotted latency aggregates:
  `query_latency_plot_summary.csv`.

The largest completed native-ladder tiers are:

- MS-GraphRAG: `N=8,750`;
- LightRAG: `N=2,254`;
- HippoRAG 2: `N=131,876`;
- LinearRAG: `N=131,876`.

LinearRAG has zero **generative-LLM** construction tokens because its
builder uses local NER and embeddings. This does not mean zero construction
compute or zero embedding tokens.

Build tokens and query tokens are reported separately. Graph-family
construction ledgers capture indexing calls, whereas files in
`token_usage/` capture phase-separated usage available to each adapter.
The paper's construction plots use the scaling ledgers. The File-System
Agent query-cost statements use `agent_query_cost_summary.csv`, aggregated
from the completed per-question run ledgers; this file supersedes partial
or zero-valued intermediate query records for that method.

Wall-clock extrapolations are order-of-magnitude feasibility estimates.
Token scaling, rather than wall time, is the primary cross-method
construction-cost comparison.
