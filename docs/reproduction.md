# Exact corpus-set reproduction

The original experiment scripts are unchanged. This supplement adds frozen
identifiers and independent verification/materialization tools so readers can
reproduce the paper's ladder without rerunning LLM-based trap mining.

## Verified release inputs

- Original experiment-code commit: `a6206e5e37e37fa474dc9394d86b10b08cce19a7`.
- HF archive revision: `e565526a73f1195e59a4055599f3a9fbe50eb67d`.
- Archived corpus: `reconstruction_seed/enterprise_corpus_meta.parquet`.
- Corpus size: 1,382,220,647 bytes; complete file checksum verified.
- Corpus SHA-256: `41d8191e44616227a9be173a6e4a09340b873a07a8f03e3f283ecbde85547cd6`.
- The 1,144 bedrock IDs and recovered background order of 510,815 IDs reproduce
  all 28 historical document-set SHA-256 values.

N=42,587 uses the bedrock plus the first 41,443 background IDs. Its historical
set hash is `cc70d16d9754ff127b2c6b3f8d5a2b08dab725e400e54ff9b51ecca7e2138e2e`.

## Verify or export IDs without downloads

Python 3.12 and the standard library are sufficient:

```bash
python scripts/frozen_ladder.py
python scripts/frozen_ladder.py --tier 42587 --out outputs/ids_42587
python -m unittest discover -s tests -v
```

The export refuses to overwrite an existing directory. The compressed
`ocean_order.txt.gz` is UTF-8, one ID per line, in recovered order. Historical
set hashes use `sha256("\n".join(sorted(ids)).encode("utf-8"))` without a final
newline. Order hashes use the same convention without sorting.

## Rebuild text/chunks using the archived metadata

The pinned reconstruction environment is in `requirements-release.txt`. It
was tested on Python 3.12.2 on Windows. These versions describe the release
validation environment, not a recovered historical GPU environment.

The corpus metadata is currently in a private archival HF repository, so an
authorized account or an author-provided copy is required. Do not put tokens
in command lines or send them with a supplement. The original 70 GB archive
does not need to be downloaded. Download the single file with an authenticated
HF CLI, or use an independently shared copy with the exact checksum above:

```bash
hf download Bstwpy/rag-scaling-rebuttal-artifacts reconstruction_seed/enterprise_corpus_meta.parquet --repo-type dataset --revision e565526a73f1195e59a4055599f3a9fbe50eb67d --local-dir /path/to/artifact-inputs
git clone https://github.com/onyx-dot-app/EnterpriseRAG-Bench.git /path/to/benchmark
git -C /path/to/benchmark checkout d36685e273713975ee20299bbf1ab64165575b3c
python -m pip install -r requirements-release.txt
python scripts/materialize_frozen_tier.py --tier 42587 --corpus-meta /path/to/artifact-inputs/reconstruction_seed/enterprise_corpus_meta.parquet --benchmark-root /path/to/benchmark --out /path/to/new-output/enterprise_N42587
```

Use a new output directory. The first tokenizer use may download the public
`o200k_base` encoding. No LLM, model checkpoint, GPU, or model API is required.
The selected tier's document text is held in memory; larger tiers require more
memory. Partial output retains an `INCOMPLETE` marker and must not be used.

The tool checks the full Parquet hash, pinned question/scaffold hashes, frozen
ID sets and order, and each selected document's token length. Outputs include
compatible `chunks.json`, `chunks_meta.jsonl`, `questions.jsonl`, `manifest.json`,
and `scale_report.json`, plus ordered IDs, document hashes, and file checksums.

## What was actually validated

On 2026-09-22, N=42,587 was materialized from the complete hash-verified
archive: 42,587 documents, 61,615 chunks, and 470 questions with annotated gold.
All original source proportions, noise proportion, and manifest hash match.

For an independent implementation check, the unchanged original
`chunk_body`/`dump_point` functions were executed on the same selected text and
ordered IDs. The three files below were byte-identical under the original
Linux UTF-8/LF convention (normalized for this Windows run):

| Output | SHA-256 shared by original functions and new materializer |
|---|---|
| chunks.json | `5613466000f47f8d7f573060284e03bc8e8b66d6f8db77415bf2a18b48f7c89a` |
| chunks_meta.jsonl | `f95b7067090e379201cf995b5b766da4f559362e7207c8534849a95e3ac4e7bc` |
| questions.jsonl | `2bf0dd6e5f1f5c1675cad5a8061703f35f572f7c9e85c00227a302df3cc31c8a` |

Manifest objects are identical; their JSON indentation differs. All historical
fields in `scale_report.json` are identical; the new report adds provenance and
content hashes. Machine-readable evidence is in `docs/validation/`.

## Evidence boundaries

The background order was recovered by replaying the original seed=42 algorithm
against the archived Parquet row order, retaining bucket insertion order.
All 28 historical set hashes match. No independent saved original order hash
or original `chunks_meta.jsonl` was found in the inspected archive. The newly
frozen order hashes verify future consistency, not independent historic order.

The upstream benchmark commit was pinned during validation; its gold IDs
reproduce the historical bedrock. It is not claimed to be the original
experiment checkout. New output equality with the original functions is not
the same as comparison with unavailable historical saved chunk bytes.

Do not rerun metadata scanning/trap mining to recover the exact paper ladder:
the historical scanner uses unordered worker completion, and seeded shuffling
depends on input/bucket order. Those original scripts are preserved for
inspection. Use the frozen IDs for the paper's document sets.

This validation does not rerun model predictions or change historical scores.
Original model/package revisions beyond those in the archive remain unknown;
the unpinned research `requirements.txt` is preserved without inventing a
historical lockfile.
