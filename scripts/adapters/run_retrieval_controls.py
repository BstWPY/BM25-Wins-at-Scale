#!/usr/bin/env python3
"""Reviewer-facing retrieval controls on the fixed EnterpriseRAG sample.

This driver intentionally keeps the controls small and pre-registered:

* ``dense_bgem3``: BAAI/bge-m3 dense retrieval, native no-instruction query
  encoding, cosine top-5, and the same zero-shot reader as the main baselines.
* ``hybrid_rrf``: unweighted reciprocal-rank fusion of the existing BM25 and
  Qwen3-Embedding-0.6B rankings.  Each retriever contributes its top-100,
  RRF uses k=60, and the fused top-5 is sent to the same reader.

The EnterpriseRAG scale snapshots are strict prefixes.  BGE-M3 therefore
embeds the largest snapshot once into a resumable memmap and smaller cells
reuse exact prefixes.  Existing Qwen document embeddings are read-only.
Every prediction cell is written to an isolated result family and finalized
only after the requested sample has exactly one successful row per ID.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from openai import OpenAI
from rank_bm25 import BM25Okapi

UNIFIED = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(UNIFIED))
import unified_config as U  # noqa: E402


_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "is", "are", "was", "were",
    "in", "on", "at", "to", "for", "of", "with", "it", "this", "that",
    "these", "those", "i", "you", "he", "she", "we", "they",
}
_WORD = re.compile(r"[a-z0-9]+")
_WRITE_LOCK = threading.Lock()


def tokenize(text: str) -> list[str]:
    return [
        token for token in _WORD.findall((text or "").lower())
        if token not in _STOPWORDS
    ]


def atomic_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with open(tmp, "w") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalized(array) -> np.ndarray:
    result = np.asarray(array, dtype=np.float32)
    if result.ndim == 1:
        result = result.reshape(1, -1)
    result /= np.linalg.norm(result, axis=1, keepdims=True) + 1e-12
    return result


def usage_dict(response) -> dict[str, int]:
    usage = getattr(response, "usage", None)
    return {
        "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
        "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
    }


def embed_request(
    client: OpenAI,
    model: str,
    texts: list[str],
    instruction: str = "",
) -> tuple[np.ndarray, dict[str, int]]:
    inputs = [
        instruction + ((text or " ").replace("\n", " "))
        for text in texts
    ]
    response = client.embeddings.create(input=inputs, model=model)
    return normalized([row.embedding for row in response.data]), usage_dict(response)


def build_bge_prefix_cache(
    chunks: list[str],
    cache_dir: Path,
    endpoints: list[str],
    model: str,
    batch_size: int,
    workers: int,
    pause_sec: float,
) -> tuple[Path, dict]:
    """Encode the maximum snapshot once, with ordered/resumable writes."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    emb_path = cache_dir / "emb.npy"
    progress_path = cache_dir / "progress.json"
    clients = [
        OpenAI(base_url=url, api_key="EMPTY", timeout=600.0, max_retries=5)
        for url in endpoints
    ]
    progress = {}
    if progress_path.exists():
        progress = json.load(open(progress_path))
    completed = int(progress.get("completed_rows", 0))
    prompt_tokens = int(progress.get("embedding_prompt_tokens", 0))
    total_tokens = int(progress.get("embedding_total_tokens", 0))
    calls = int(progress.get("embedding_calls", 0))

    if emb_path.exists():
        matrix = np.load(emb_path, mmap_mode="r+")
        if matrix.shape[0] != len(chunks):
            raise RuntimeError(
                f"BGE cache shape mismatch: {matrix.shape[0]} != {len(chunks)}")
        if completed > len(chunks):
            raise RuntimeError("BGE progress exceeds cache length")
        dimension = int(matrix.shape[1])
    else:
        # One smoke request discovers the server-returned embedding dimension.
        smoke, usage = embed_request(clients[0], model, [chunks[0]])
        dimension = int(smoke.shape[1])
        matrix = np.lib.format.open_memmap(
            emb_path, mode="w+", dtype=np.float32,
            shape=(len(chunks), dimension))
        matrix[0:1] = smoke
        matrix.flush()
        completed = 1
        prompt_tokens += usage["prompt_tokens"]
        total_tokens += usage["total_tokens"]
        calls += 1
        atomic_json(progress_path, {
            "schema_version": 1,
            "model": model,
            "endpoints": endpoints,
            "rows": len(chunks),
            "dimension": dimension,
            "batch_size": batch_size,
            "workers": workers,
            "pause_sec": pause_sec,
            "completed_rows": completed,
            "embedding_prompt_tokens": prompt_tokens,
            "embedding_total_tokens": total_tokens,
            "embedding_calls": calls,
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        })

    if completed == len(chunks):
        print(f"[bge-cache] complete {matrix.shape} -> {emb_path}", flush=True)
        return emb_path, json.load(open(progress_path))

    starts = list(range(completed, len(chunks), batch_size))

    def one(item):
        order, start = item
        end = min(start + batch_size, len(chunks))
        client = clients[order % len(clients)]
        array, usage = embed_request(client, model, chunks[start:end])
        # Optional duty-cycle limiter.  It lives inside the worker so an eager
        # executor cannot prefetch around the pause.
        if pause_sec > 0:
            time.sleep(pause_sec)
        return start, end, array, usage

    print(
        f"[bge-cache] resume={completed}/{len(chunks)} batch={batch_size} "
        f"workers={workers} endpoints={len(endpoints)}", flush=True)
    # executor.map yields in input order.  This lets completed_rows remain a
    # valid contiguous checkpoint even though requests execute concurrently.
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for start, end, array, usage in executor.map(one, enumerate(starts)):
            if array.shape != (end - start, dimension):
                raise RuntimeError(
                    f"embedding shape mismatch at {start}:{end}: {array.shape}")
            matrix[start:end] = array
            matrix.flush()
            completed = end
            prompt_tokens += usage["prompt_tokens"]
            total_tokens += usage["total_tokens"]
            calls += 1
            atomic_json(progress_path, {
                "schema_version": 1,
                "model": model,
                "endpoints": endpoints,
                "rows": len(chunks),
                "dimension": dimension,
                "batch_size": batch_size,
                "workers": workers,
                "pause_sec": pause_sec,
                "completed_rows": completed,
                "embedding_prompt_tokens": prompt_tokens,
                "embedding_total_tokens": total_tokens,
                "embedding_calls": calls,
                "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            })
            if completed % (batch_size * 100) < batch_size:
                print(f"  [bge-cache {completed}/{len(chunks)}]", flush=True)
    del matrix
    return emb_path, json.load(open(progress_path))


def top_indices(scores: np.ndarray, k: int) -> list[int]:
    k = min(k, int(scores.shape[0]))
    if k <= 0:
        return []
    candidate = np.argpartition(-scores, k - 1)[:k]
    # Stable deterministic ordering: score descending, then chunk ID ascending.
    ordered = sorted(
        (int(index) for index in candidate),
        key=lambda index: (-float(scores[index]), index),
    )
    return ordered


def dense_rankings(matrix: np.ndarray, queries: np.ndarray, k: int) -> list[list[int]]:
    result = []
    for index, query in enumerate(queries):
        result.append(top_indices(matrix @ query, k))
        if (index + 1) % 25 == 0:
            print(f"  [dense-rank {index + 1}/{len(queries)}]", flush=True)
    return result


def rrf_fuse(left: list[int], right: list[int], rrf_k: int, top_k: int) -> list[int]:
    scores: dict[int, float] = {}
    for ranking in (left, right):
        for rank, chunk_id in enumerate(ranking, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (rrf_k + rank)
    return [
        chunk_id for chunk_id, _ in
        sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:top_k]
    ]


def reader_answer(client: OpenAI, context: str, question: str) -> tuple[str, dict[str, int]]:
    response = client.chat.completions.create(
        model=U.READER_MODEL,
        temperature=U.GEN_TEMPERATURE,
        max_tokens=U.GEN_MAX_TOKENS,
        seed=U.SEED,
        extra_body=U.LLM_EXTRA_BODY,
        messages=[
            {"role": "system", "content": U.READER_SYSTEM_PROMPT},
            {"role": "user", "content": U.reader_user_msg(context, question)},
        ],
    )
    usage = getattr(response, "usage", None)
    return (response.choices[0].message.content or "").strip(), {
        "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
        "completion_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
        "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
    }


def run_reader_cell(
    family: str,
    dataset: str,
    chunks: list[str],
    questions: list[dict],
    rankings: list[list[int]],
    endpoints: str,
    concurrency: int,
    parameters: dict,
) -> None:
    out_dir = Path(U.RESULTS_ROOT) / family / dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    partial = out_dir / "predictions.partial.jsonl"
    final = out_dir / "predictions.jsonl"
    manifest_path = out_dir / "run_manifest.json"
    parameter_hash = hashlib.sha256(json.dumps(
        parameters, sort_keys=True, separators=(",", ":")
    ).encode()).hexdigest()
    if manifest_path.exists():
        prior = json.load(open(manifest_path))
        if prior.get("parameter_sha256") != parameter_hash:
            raise RuntimeError(
                f"refusing to mix incompatible partial rows in {out_dir}")
    else:
        atomic_json(manifest_path, {
            "schema_version": 1,
            "family": family,
            "dataset": dataset,
            "sample_n": len(questions),
            "parameters": parameters,
            "parameter_sha256": parameter_hash,
            "driver_sha256": file_sha256(Path(__file__)),
            "status": "running",
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        })

    done = {}
    if partial.exists():
        for line in open(partial):
            try:
                row = json.loads(line)
                if row.get("id") and not row.get("error"):
                    done[row["id"]] = row
            except Exception:
                continue

    endpoint_list = [
        value.strip() for value in endpoints.split(",") if value.strip()]
    if not endpoint_list:
        raise RuntimeError("reader endpoint list is empty")
    clients = [
        OpenAI(base_url=value, api_key="EMPTY", timeout=600.0, max_retries=5)
        for value in endpoint_list
    ]
    jobs = [
        (question, ranking)
        for question, ranking in zip(questions, rankings)
        if question["id"] not in done
    ]
    print(
        f"[reader] {family}/{dataset}: todo={len(jobs)} "
        f"resume={len(done)} concurrency={concurrency}", flush=True)

    def one(job_index, question, ranking):
        context = "\n\n".join(chunks[index] for index in ranking)
        client = clients[job_index % len(clients)]
        answer, usage = reader_answer(client, context, question["question"])
        return {
            "id": question["id"],
            "predicted_answer": answer,
            "retrieved_chunk_ids": ranking,
            "usage": usage,
        }

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = {
            executor.submit(one, job_index, question, ranking): question["id"]
            for job_index, (question, ranking) in enumerate(jobs)
        }
        for count, future in enumerate(as_completed(futures), start=1):
            row = future.result()
            with _WRITE_LOCK:
                with open(partial, "a") as handle:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                    handle.flush()
                    os.fsync(handle.fileno())
            done[row["id"]] = row
            if count % 25 == 0 or count == len(jobs):
                print(f"  [reader {len(done)}/{len(questions)}]", flush=True)

    expected = [question["id"] for question in questions]
    if set(done) != set(expected):
        missing = sorted(set(expected) - set(done))
        raise RuntimeError(f"{family}/{dataset} incomplete: missing={missing[:5]}")
    with open(final.with_suffix(".jsonl.tmp"), "w") as handle:
        for question_id in expected:
            row = dict(done[question_id])
            row.pop("usage", None)
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(final.with_suffix(".jsonl.tmp"), final)

    totals = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    for row in done.values():
        for key in totals:
            totals[key] += int((row.get("usage") or {}).get(key, 0))
    token_path = Path(U.RESULTS_ROOT) / "token_usage" / f"{family}_{dataset}.json"
    atomic_json(token_path, {
        "baseline": family,
        "dataset": dataset,
        "source": "wrapper_usage",
        "build": {
            "prompt_tokens": 0, "completion_tokens": 0,
            "total_tokens": 0, "calls": 0,
        },
        "qa": {**totals, "calls": len(questions)},
        "build_qa_total_tokens": totals["total_tokens"],
        "note": (
            "build=0 API LLM tokens; local/remote embedding compute is "
            "recorded separately in the run manifest"
        ),
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
    })
    manifest = json.load(open(manifest_path))
    manifest.update({
        "status": "complete",
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "prediction_rows": len(questions),
        "predictions_sha256": file_sha256(final),
        "token_usage_sha256": file_sha256(token_path),
    })
    atomic_json(manifest_path, manifest)
    print(f"[done] {family}/{dataset} -> {final}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scales", default="1144,2254,6980,42587,131876,511959")
    parser.add_argument(
        "--modes", default="dense_bgem3,hybrid_rrf",
        help="comma-separated subset of dense_bgem3,hybrid_rrf")
    parser.add_argument("--sample_file", required=True)
    parser.add_argument(
        "--reader_endpoints", "--reader_endpoint", dest="reader_endpoints",
        required=True, help="comma-separated OpenAI chat base URLs")
    parser.add_argument(
        "--bge_endpoints", default="",
        help="comma-separated OpenAI embedding base URLs")
    parser.add_argument("--bge_model", default="bge-m3")
    parser.add_argument("--qwen_embed_endpoint", default="")
    parser.add_argument("--qwen_embed_model", default="qwen3-embed-0.6b")
    parser.add_argument(
        "--max_chunks_file", default="",
        help="optional largest-snapshot chunks.json; smaller scales use prefixes")
    parser.add_argument(
        "--questions_file", default="",
        help="optional shared questions.jsonl (questions are identical by ID)")
    parser.add_argument(
        "--chunk_counts",
        default="1144:2018,2254:3606,6980:10419,42587:61615,"
                "131876:190096,511959:737878",
        help="scale:chunk-count mapping used with --max_chunks_file")
    parser.add_argument(
        "--qwen_cache_root", default=str(UNIFIED / "naiverag_ws"),
        help="root containing enterprise_N*/emb.npy canonical caches")
    parser.add_argument("--bge_batch_size", type=int, default=32)
    parser.add_argument("--bge_workers", type=int, default=2)
    parser.add_argument(
        "--bge_batch_pause_sec", type=float, default=0.0,
        help="per-request duty-cycle pause inside each BGE worker")
    parser.add_argument("--reader_concurrency", type=int, default=12)
    parser.add_argument("--candidate_k", type=int, default=100)
    parser.add_argument("--rrf_k", type=int, default=60)
    parser.add_argument("--top_k", type=int, default=5)
    args = parser.parse_args()

    scales = [int(value) for value in args.scales.split(",") if value.strip()]
    modes = [value.strip() for value in args.modes.split(",") if value.strip()]
    unknown = set(modes) - {"dense_bgem3", "hybrid_rrf"}
    if unknown:
        raise SystemExit(f"unknown modes: {sorted(unknown)}")
    if "dense_bgem3" in modes and not args.bge_endpoints:
        raise SystemExit("--bge_endpoints is required for dense_bgem3")
    if "hybrid_rrf" in modes and not args.qwen_embed_endpoint:
        raise SystemExit("--qwen_embed_endpoint is required for hybrid_rrf")

    sample_ids = set(json.load(open(args.sample_file))["question_ids"])
    max_scale = max(scales)
    max_dataset = f"enterprise_N{max_scale}"
    chunk_counts = {}
    for item in args.chunk_counts.split(","):
        scale, count = item.split(":", 1)
        chunk_counts[int(scale)] = int(count)
    if args.max_chunks_file:
        max_chunks = json.load(open(args.max_chunks_file))
        if chunk_counts.get(max_scale) != len(max_chunks):
            raise RuntimeError(
                f"maximum chunk count mismatch: mapping={chunk_counts.get(max_scale)} "
                f"file={len(max_chunks)}")
    else:
        max_chunks = json.load(open(
            Path(U.DATA_ROOT) / max_dataset / "chunks.json"))
        chunk_counts[max_scale] = len(max_chunks)
    shared_questions = None
    if args.questions_file:
        shared_questions = [
            json.loads(line) for line in open(args.questions_file)]

    def scale_payload(scale):
        dataset = f"enterprise_N{scale}"
        if args.max_chunks_file:
            if scale not in chunk_counts:
                raise RuntimeError(f"missing chunk count for scale {scale}")
            chunks = max_chunks[:chunk_counts[scale]]
        else:
            chunks = json.load(open(
                Path(U.DATA_ROOT) / dataset / "chunks.json"))
            if chunks != max_chunks[:len(chunks)]:
                raise RuntimeError(
                    f"{dataset} is not an exact prefix of {max_dataset}")
        source_questions = shared_questions
        if source_questions is None:
            source_questions = [
                json.loads(line) for line in open(
                    Path(U.DATA_ROOT) / dataset / "questions.jsonl")]
        questions = [
            row for row in source_questions if row["id"] in sample_ids]
        if len(questions) != len(sample_ids):
            raise RuntimeError(
                f"{dataset}: fixed sample mismatch "
                f"{len(questions)} != {len(sample_ids)}")
        return dataset, chunks, questions

    # BGE document encoding is independent of the Qwen/BM25 hybrid.  Start it
    # on its dedicated endpoint while the main thread evaluates the hybrid,
    # then join before any BGE retrieval.  This keeps one auditable driver PID
    # while using separate GPUs without cross-method request contention.
    bge_executor = None
    bge_future = None
    if "dense_bgem3" in modes:
        bge_executor = ThreadPoolExecutor(max_workers=1)
        bge_future = bge_executor.submit(
            build_bge_prefix_cache,
            max_chunks,
            UNIFIED / "dense_variant_ws" / "bge_m3" / max_dataset,
            [url.strip() for url in args.bge_endpoints.split(",") if url.strip()],
            args.bge_model,
            args.bge_batch_size,
            args.bge_workers,
            args.bge_batch_pause_sec,
        )

    qwen_client = (
        OpenAI(
            base_url=args.qwen_embed_endpoint,
            api_key="EMPTY", timeout=600.0, max_retries=5)
        if "hybrid_rrf" in modes else None
    )
    if "hybrid_rrf" in modes:
        for scale in scales:
            dataset, chunks, questions = scale_payload(scale)
            query_texts = [row["question"] for row in questions]
            qwen_queries, query_usage = embed_request(
                qwen_client, args.qwen_embed_model, query_texts,
                instruction=U.EMBED_QUERY_INSTRUCTION)
            qwen_path = (
                Path(args.qwen_cache_root) / dataset / "emb.npy")
            qwen_matrix = np.load(qwen_path, mmap_mode="r")
            if qwen_matrix.shape[0] != len(chunks):
                raise RuntimeError(
                    f"{dataset}: Qwen cache rows {qwen_matrix.shape[0]} "
                    f"!= chunks {len(chunks)}")
            dense_top = dense_rankings(
                qwen_matrix, qwen_queries, args.candidate_k)
            print(f"[bm25] build {dataset} ({len(chunks)} chunks)", flush=True)
            bm25 = BM25Okapi([tokenize(chunk) for chunk in chunks])
            bm25_top = []
            for index, question in enumerate(query_texts):
                bm25_top.append(top_indices(
                    bm25.get_scores(tokenize(question)), args.candidate_k))
                if (index + 1) % 25 == 0:
                    print(f"  [bm25-rank {index + 1}/{len(query_texts)}]", flush=True)
            rankings = [
                rrf_fuse(left, right, args.rrf_k, args.top_k)
                for left, right in zip(bm25_top, dense_top)
            ]
            run_reader_cell(
                "hybrid_rrf", dataset, chunks, questions, rankings,
                args.reader_endpoints, args.reader_concurrency,
                parameters={
                    "method": "unweighted reciprocal rank fusion",
                    "components": ["BM25", args.qwen_embed_model],
                    "candidate_k_per_component": args.candidate_k,
                    "rrf_k": args.rrf_k,
                    "top_k": args.top_k,
                    "sample_file_sha256": file_sha256(Path(args.sample_file)),
                    "qwen_document_cache": str(qwen_path),
                    "query_embedding_usage": query_usage,
                    "reader_model": U.READER_MODEL,
                    "reader_temperature": U.GEN_TEMPERATURE,
                    "reader_seed": U.SEED,
                },
            )
            del bm25, qwen_matrix

    if "dense_bgem3" in modes:
        bge_matrix_path, bge_progress = bge_future.result()
        bge_executor.shutdown(wait=True)
        bge_client = OpenAI(
            base_url=args.bge_endpoints.split(",")[0].strip(),
            api_key="EMPTY", timeout=600.0, max_retries=5)
        bge_matrix = np.load(bge_matrix_path, mmap_mode="r")
        for scale in scales:
            dataset, chunks, questions = scale_payload(scale)
            query_texts = [row["question"] for row in questions]
            bge_queries, query_usage = embed_request(
                bge_client, args.bge_model, query_texts)
            rankings = dense_rankings(
                bge_matrix[:len(chunks)], bge_queries, args.top_k)
            run_reader_cell(
                "dense_bgem3", dataset, chunks, questions, rankings,
                args.reader_endpoints, args.reader_concurrency,
                parameters={
                    "method": "dense cosine",
                    "embedding_model": args.bge_model,
                    "query_instruction": "",
                    "top_k": args.top_k,
                    "sample_file_sha256": file_sha256(Path(args.sample_file)),
                    "max_prefix_cache": str(bge_matrix_path),
                    "max_prefix_rows": len(max_chunks),
                    "document_embedding_usage": bge_progress,
                    "query_embedding_usage": query_usage,
                    "reader_model": U.READER_MODEL,
                    "reader_temperature": U.GEN_TEMPERATURE,
                    "reader_seed": U.SEED,
                },
            )


if __name__ == "__main__":
    main()
