#!/usr/bin/env python
"""第1步：全量元数据扫描。源文件 → corpus_meta.parquet。
一次性产物，后续 floor/trap-mining/前缀/dump 全部复用，绝不再逐个读原始 json。

每行：dsid, source(路径首段), is_noise, token_len, text。
is_noise 是 original_location 与 dataset_noise_document 的并集口径。
  python build_corpus_meta.py
"""
import os
import json
from multiprocessing import Pool

import pandas as pd

ENT = os.environ.get(
    "ENTERPRISE_RAG_BENCH_ROOT",
    os.path.join(os.environ.get("PROJECT_ROOT", os.getcwd()), "GraphRAG_fs_mapping", "EnterpriseRAG-Bench"),
)
SOURCES = f"{ENT}/generated_data/sources"
UUID_INDEX = f"{ENT}/generated_data/uuid_index.json"
OUT = os.environ.get(
    "ENTERPRISE_CORPUS_META",
    os.path.join(os.environ.get("URAG_ROOT", os.getcwd()), "data", "enterprise_corpus_meta.parquet"),
)
ENCODING = "o200k_base"
WORKERS = int(os.environ.get("CORPUS_META_WORKERS", "32"))

_enc = None


def _init():
    global _enc
    import tiktoken
    _enc = tiktoken.get_encoding(ENCODING)


def _render(doc):
    tfn = doc.get("title_field_name")
    title = str(doc.get(tfn, "")) if tfn else ""
    cfns = doc.get("content_field_names") or []
    parts = []
    for fn in cfns:
        if fn not in doc:
            continue
        v = doc[fn]
        v = "\n".join(str(x) for x in v) if isinstance(v, list) else str(v)
        parts.append(v if len(cfns) == 1 else f"{fn}:\n{v}")
    return title, "\n\n".join(parts)


def _process(item):
    dsid, rel = item
    try:
        d = json.load(open(f"{SOURCES}/{rel}"))
    except Exception:
        return None
    title, content = _render(d)
    if not content.strip():
        return None
    text = f"{title}\n\n{content}".strip()
    # 实验使用并集口径：被移动/近重复的文档，或 benchmark 显式标记的噪声文档。
    is_noise = ("original_location" in d) or bool(d.get("dataset_noise_document"))
    return {
        "dsid": dsid,
        "source": rel.split("/", 1)[0],
        "is_noise": is_noise,
        "token_len": len(_enc.encode(text, disallowed_special=())),
        "text": text,
    }


def main():
    idx = json.load(open(UUID_INDEX))
    items = list(idx.items())
    print(f"[scan] {len(items)} 文档，{WORKERS} 进程...", flush=True)
    rows = []
    with Pool(WORKERS, initializer=_init) as p:
        for i, r in enumerate(p.imap_unordered(_process, items, chunksize=200)):
            if r:
                rows.append(r)
            if (i + 1) % 50000 == 0:
                print(f"  {i+1}/{len(items)} 已处理，有效 {len(rows)}", flush=True)
    print(f"[write] 有效 {len(rows)} 行 → parquet ...", flush=True)
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    df.to_parquet(OUT)
    # 速报关键全局统计（供 floor/ocean 配额用）
    print(f"[done] {OUT}", flush=True)
    print(f"  noise_frac(union definition) = {df['is_noise'].mean():.4f}", flush=True)
    print(f"  source 分布: {df['source'].value_counts(normalize=True).round(4).to_dict()}", flush=True)
    print(f"  token_len: mean={df['token_len'].mean():.0f} median={df['token_len'].median():.0f}", flush=True)


if __name__ == "__main__":
    main()
