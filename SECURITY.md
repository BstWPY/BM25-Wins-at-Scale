# Research artifact security

For document-set reproduction, use `scripts/frozen_ladder.py` and
`scripts/materialize_frozen_tier.py`. They do not invoke a model or shell agent.
The original experiment scripts are retained unchanged for provenance.

The optional Codex control in `scripts/sandbox_agent/run_codex_fs.py` disables
its internal sandbox/approval mechanism, and the Codex/Pi launchers inherit the
parent environment. These are historical research controls, not safe default
programs for an everyday workstation. If reproducing them, provision a
disposable VM/container with only corpus/output mounts, dedicated credentials,
and restricted network access. A prompt and read-only corpus permissions do
not restrict access to other host files. This release does not certify live
agent execution or silently change the conditions of reported experiments.

The shim can dump complete requests and responses when `CODEX_SHIM_DUMP` is
set. Session logs and model outputs may include full benchmark content. Keep
them outside public source releases. Graph/shim servers are local research
services and should not be exposed as public authenticated APIs.

HippoRAG graph code loads Python pickle indexes. Only load a verified artifact
from a trusted source in an isolated environment; no indexes are bundled here.

The source tree and locally available Git history were scanned with Gitleaks
v8.30.1. No credentials were detected. The seven shipped figure PDFs had no
nonempty Author/Subject metadata fields. These checks do not prove that all
possible secrets, binary metadata, or dependency vulnerabilities are absent.

The private HF core archive has ten absolute-path links; it should not be
published unchanged as a portable supplement. Twelve generic-key matches in
its regular files were verified to be `token_usage_sha256` checksums. Its raw
outputs and large query indexes are not included in this source release and
are not covered by a blanket publication approval. The complete 70 GB archive
has not been audited.

`requirements-release.txt` pins the reconstruction environment tested for this
supplement. The historical optional GPU packages in `requirements.txt` are not
a recovered/version-locked environment and were not fully vulnerability-audited.
