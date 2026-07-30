#!/usr/bin/env python
"""把某规模点的【原始文档(未分块)】落成 sandbox agent 探索的文件树(端点③ sandbox)。

关键: sandbox 用原始文档原文(enterprise_corpus_meta.parquet 的 text 列, 按 dsid),
不做 chunk 分块 —— 分块是 graphrag 那些图 baseline 的事。agent 在 {source}/{dsid}.md 的
自然文档树上 grep/glob/read, 等同 EnterpriseRAG 官方 bash-agent 在真实文档上探索的设定。

chunk 只用来建 filemap_flat.json(文件相对路径 -> 该 doc 覆盖的 chunk_id 列表), 供 agent 读过的
文件解析回 retrieved chunk_ids, 对齐 gold_chunk_ids 评 recall —— chunk 不进 agent 视野。

端点② graph->FS mapping 用另一脚本把图展平成文件树, 复用同一 harness、只换根目录。

跑: zzt/bin/python build_filetree.py --dataset enterprise_N1144
"""
import os
import re
import json
import argparse
from collections import defaultdict, Counter
import pandas as pd

W = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 2026-07-07 迁移
PARQUET = f"{W}/data/enterprise_corpus_meta.parquet"


def safe(name):
    return re.sub(r"[^A-Za-z0-9._-]", "_", str(name))[:120]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    ds = f"{W}/data/{args.dataset}"
    out = args.out or f"{W}/sandbox_agent/trees/{args.dataset}/flat"
    os.makedirs(out, exist_ok=True)

    # 该点包含的 dsid 集合
    dsids = json.load(open(f"{ds}/manifest.json"))["dsids"]
    # dsid -> chunk_ids(评 recall) + source, 从 chunks_meta
    d2c, d2src = defaultdict(list), {}
    for m in (json.loads(l) for l in open(f"{ds}/chunks_meta.jsonl")):
        d2c[m["dsid"]].append(m["chunk_id"])
        d2src[m["dsid"]] = m["source"]
    # 原始文档原文(parquet, chunk 化之前)
    meta = pd.read_parquet(PARQUET, columns=["dsid", "text"])
    d2text = dict(zip(meta["dsid"], meta["text"]))
    # scaffold 等不在 parquet 的, fallback 用 chunks 拼接(无 overlap, 不重复)
    chunks = json.load(open(f"{ds}/chunks.json"))

    filemap = {}
    n_orig = n_fallback = 0
    for dsid in dsids:
        src = d2src.get(dsid, "unknown")
        t = d2text.get(dsid)
        if isinstance(t, str) and t.strip():
            body = t                                            # 原始文档原文
            n_orig += 1
        else:
            body = "\n\n".join(chunks[c] for c in sorted(d2c[dsid]))  # scaffold fallback
            n_fallback += 1
        rel = f"{safe(src)}/{safe(dsid)}.md"
        fp = f"{out}/{rel}"
        os.makedirs(os.path.dirname(fp), exist_ok=True)
        with open(fp, "w") as f:
            f.write(body)
        filemap[rel] = sorted(d2c[dsid])

    mp = f"{W}/sandbox_agent/trees/{args.dataset}/filemap_flat.json"
    json.dump(filemap, open(mp, "w"))
    print(f"[flat] 落 {len(dsids)} 文档(原文{n_orig}/fallback{n_fallback}) -> {out}")
    print(f"[flat] filemap({len(filemap)}) -> {mp}")
    print("  source分布:", dict(Counter(d2src.get(d, '?') for d in dsids)))


if __name__ == "__main__":
    main()
