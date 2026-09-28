# Running the research scripts

Corpus tier construction uses `scripts/frozen_ladder.py` and
`scripts/materialize_frozen_tier.py` and does not invoke a model or shell agent.

## Optional agent controls

The Codex control in `scripts/sandbox_agent/run_codex_fs.py` disables its
internal sandbox and approval mechanism. The Codex and Pi launchers inherit
the parent environment. Run these controls in a disposable VM or container
with corpus/output mounts, dedicated credentials, and restricted network
access.

## Services and logs

Graph and shim servers are local research services. Keep them on local
interfaces. Setting `CODEX_SHIM_DUMP` records complete requests and responses;
keep those logs outside public source releases.

## Index files

HippoRAG loads Python pickle indexes. Load only indexes from a trusted source
inside the isolated experiment environment.
