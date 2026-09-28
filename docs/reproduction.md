# Constructing the corpus ladder

The study uses 28 nested corpus tiers. Each tier contains 1,144 bedrock
documents plus a prefix of the 510,815-document background ordering. The
42,587-document tier uses the first 41,443 background documents.

## Inputs

- Corpus, questions, and scaffold documents:
  [BM25 Wins at Scale — Corpus](https://huggingface.co/datasets/Bstwpy/BM25-Wins-at-Scale-Corpus)
  (approximately 1.38 GB).
- Bedrock document IDs: `data_manifests/frozen/bedrock_dsids.txt`.
- Background ordering: `data_manifests/frozen/ocean_order.txt.gz`.
- Available tier sizes: `data_manifests/frozen/tier_sizes.json`.

The corpus package contains 511,957 document rows and two separate scaffold
documents, giving 511,959 documents at the largest tier.

## Export document IDs

Run from the code repository using Python 3.12:

```bash
python scripts/frozen_ladder.py --tier 42587 --out ../bm25-tiers/ids_42587
```

This writes `ordered_dsids.txt` and `manifest.json`. Replace `42587` with
another size from `tier_sizes.json`. Use a new output directory.

## Build document chunks and questions

Download the corpus and install the dependencies:

```bash
python -m pip install huggingface_hub
hf download Bstwpy/BM25-Wins-at-Scale-Corpus --repo-type dataset --local-dir ../bm25-corpus
python -m pip install -r requirements-release.txt
```

Build the 42,587-document tier:

```bash
python scripts/materialize_frozen_tier.py --tier 42587 --corpus-meta ../bm25-corpus/reconstruction_seed/enterprise_corpus_meta.parquet --benchmark-root ../bm25-corpus/benchmark --out ../bm25-tiers/enterprise_N42587
```

Use a new output directory. The tokenizer is downloaded on first use. Corpus
tier construction does not require a GPU, LLM, or model API. Larger tiers use
more memory because selected document text is held in memory.

## Outputs

| File | Contents |
|---|---|
| `chunks.json` | Document chunks for indexing and retrieval. |
| `chunks_meta.jsonl` | Chunk IDs, document IDs, source categories, and noise indicators. |
| `questions.jsonl` | Benchmark questions with supporting documents mapped to the selected tier's chunks. |
| `manifest.json` | Selected document IDs and tier metadata. |
| `ordered_dsids.txt` | Document IDs in tier order. |
| `scale_report.json` | Document/chunk counts and source/noise composition. |

The 42,587-document tier produces 61,615 chunks. Feed `chunks.json` and
`chunks_meta.jsonl` into your retrieval index, and use `questions.jsonl` for
the corresponding evaluation questions. The experiment adapters are in
`scripts/adapters/`.

A directory containing `INCOMPLETE` is still being built; use it after the
command completes and the marker is removed.
