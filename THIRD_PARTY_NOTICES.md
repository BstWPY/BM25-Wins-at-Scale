# Third-party materials

The root MIT license covers this repository's original code and documentation.
It does not replace licenses on benchmark data, models, third-party packages,
or externally downloaded artifacts.

The document IDs refer to EnterpriseRAG-Bench by onyx-dot-app. Its repository
at commit `d36685e273713975ee20299bbf1ab64165575b3c` carries the MIT license,
Copyright (c) 2026 DanswerAI, Inc. A copy is included in
`licenses/EnterpriseRAG-Bench-MIT.txt`. This candidate contains identifier
manifests, not benchmark document text, questions, or model outputs. Input
source hashes and the exact upstream commit are recorded in
`data_manifests/frozen/benchmark_source.json`.

Qwen model weights/tokenizers and optional GraphRAG, LightRAG, HippoRAG,
LinearRAG, Qwen-Agent, Pi, and Codex components must be obtained separately
under their own licenses. No model weights, executable tools, prebuilt indexes,
or third-party package binaries are included in this release candidate.

The private Hugging Face archival repository currently declares `license:
other`. This code release does not relicense or authorize wholesale publication
of that archive. The archive contains raw predictions, judgments, and indexes
outside the scope of the selected code/ID release.
