#!/usr/bin/env python
"""LightRAG 图 index server —— KV/vdb JSON + graphml 常驻内存, HTTP 工具暴露(端点② agentic+graph)。

数据源(lightrag_ws/{dataset}/):
  vdb_entities/relationships/chunks.json  nano-vectordb: base64 float32 matrix, dim=1024(Qwen3-embed)
  graph_chunk_entity_relation.graphml     实体图(节点带 description, 边带关系 description/keywords)
  kv_store_text_chunks.json               lightrag 自切分 chunk 全文
  kv_store_entity_chunks.json             实体 -> chunk_ids
  kv_store_full_docs.json                 doc-id -> 原文(用于映射回数据集 dsid/chunk_ids 评 recall)
recall 映射: chunk 内容 md5 精确对 unified chunks(74% 逐字节同) + full_doc→dsid 文档级兜底。
原生 hybrid 检索的 LLM 关键词抽取不进工具 —— agent 自己给关键词, 语义等价且零 LLM。

跑: envs/vllm_judge/bin/python lightrag_server.py --dataset enterprise_N1144 --port 8802
"""
import os, sys, json, base64, hashlib, argparse
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from fastapi import FastAPI
from fastapi.responses import JSONResponse
import uvicorn
from transformers import AutoTokenizer, AutoModel
import igraph as ig

W = os.environ.get("URAG_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EMBED_MODEL = os.environ.get("EMBED_MODEL") or os.path.join(
    os.path.dirname(os.path.dirname(W)), "models", "Qwen3-Embedding-0.6B")


class Embedder:
    def __init__(self, path):
        self.tok = AutoTokenizer.from_pretrained(path, padding_side="left")
        self.model = AutoModel.from_pretrained(path, dtype=torch.float32).eval()

    @torch.no_grad()
    def __call__(self, texts):
        texts = [(t or " ").replace("\n", " ") for t in texts]
        b = self.tok(texts, padding=True, truncation=True, max_length=2048, return_tensors="pt")
        out = self.model(**b).last_hidden_state
        idx = b["attention_mask"].sum(1) - 1
        v = out[torch.arange(out.size(0)), idx]
        return F.normalize(v, dim=-1).numpy().astype(np.float32)


def load_vdb(path):
    v = json.load(open(path))
    mat = np.frombuffer(base64.b64decode(v["matrix"]), dtype=np.float32).reshape(
        -1, v["embedding_dim"]).copy()
    mat /= np.clip(np.linalg.norm(mat, axis=1, keepdims=True), 1e-9, None)
    return v["data"], mat


class Index:
    def __init__(self, dataset):
        B = f"{W}/lightrag_ws/{dataset}"
        print(f"[load] {B}", flush=True)
        self.ent_rows, self.ent_mat = load_vdb(f"{B}/vdb_entities.json")
        self.rel_rows, self.rel_mat = load_vdb(f"{B}/vdb_relationships.json")
        self.chk_rows, self.chk_mat = load_vdb(f"{B}/vdb_chunks.json")
        self.text_chunks = json.load(open(f"{B}/kv_store_text_chunks.json"))
        self.entity_chunks = json.load(open(f"{B}/kv_store_entity_chunks.json"))
        g = ig.Graph.Read_GraphML(f"{B}/graph_chunk_entity_relation.graphml")
        self.g = g
        self.vid = {v["id"]: v.index for v in g.vs}

        # 数据集映射: chunk 内容 md5 -> unified chunk_ids; doc 内容 md5 -> dsid
        ds = f"{W}/data/{dataset}"
        chunks = json.load(open(f"{ds}/chunks.json"))
        metas = [json.loads(l) for l in open(f"{ds}/chunks_meta.jsonl")]
        cmd5 = defaultdict(list)
        dsid_chunks = defaultdict(list)
        for t, m in zip(chunks, metas):
            cmd5[hashlib.md5(t.encode()).hexdigest()].append(int(m["chunk_id"]))
            dsid_chunks[m["dsid"]].append(int(m["chunk_id"]))
        full_docs = json.load(open(f"{B}/kv_store_full_docs.json"))
        import pandas as pd
        meta = pd.read_parquet(f"{W}/data/enterprise_corpus_meta.parquet", columns=["dsid", "text"])
        dmd5 = {hashlib.md5(t.encode()).hexdigest(): d for d, t in zip(meta["dsid"], meta["text"])}
        self.doc2dsid = {k: dmd5.get(hashlib.md5(v["content"].encode()).hexdigest())
                         for k, v in full_docs.items()}
        self.map_chunk = {}
        for cid, row in self.text_chunks.items():
            ids = cmd5.get(hashlib.md5(row["content"].encode()).hexdigest())
            if ids:
                self.map_chunk[cid] = {"chunk_ids": ids, "exact": True,
                                       "dsid": None}
            else:
                d = self.doc2dsid.get(row.get("full_doc_id"))
                self.map_chunk[cid] = {"chunk_ids": dsid_chunks.get(d, []), "exact": False,
                                       "dsid": d}
        self.dsid_of_doc = self.doc2dsid

    def topk(self, mat, qv, k):
        s = mat @ qv
        idx = np.argpartition(-s, min(k, len(s) - 1))[:k]
        return sorted(((int(i), float(s[i])) for i in idx), key=lambda x: -x[1])


ap = argparse.ArgumentParser()
ap.add_argument("--dataset", default="enterprise_N1144")
ap.add_argument("--port", type=int, default=8802)
ARGS = ap.parse_args()
IX = Index(ARGS.dataset)
EMB = Embedder(EMBED_MODEL)
print("[ready]", flush=True)
app = FastAPI()


def chunk_payload(cid, score=None, preview=0):
    row = IX.text_chunks.get(cid, {})
    m = IX.map_chunk.get(cid, {})
    d = {"chunk": cid, "doc": row.get("full_doc_id"),
         "dsid": m.get("dsid") or IX.dsid_of_doc.get(row.get("full_doc_id")),
         "chunk_ids": m.get("chunk_ids", []), "exact_map": m.get("exact")}
    if score is not None:
        d["score"] = round(score, 6)
    if preview:
        d["preview"] = (row.get("content") or "")[:preview]
    return d


@app.get("/health")
def health():
    return {"dataset": ARGS.dataset, "entities": len(IX.ent_rows), "relations": len(IX.rel_rows),
            "chunks": len(IX.chk_rows), "graph_vertices": IX.g.vcount(), "graph_edges": IX.g.ecount()}


@app.get("/search_entities")
def search_entities(q: str, k: int = 8):
    qv = EMB([q])[0]
    out = []
    for i, s in IX.topk(IX.ent_mat, qv, k):
        r = IX.ent_rows[i]
        name = r.get("entity_name")
        desc = ""
        vid = IX.vid.get(name)
        if vid is not None:
            desc = (IX.g.vs[vid].attributes().get("description") or "")[:300]
        out.append({"entity": name, "score": round(s, 4), "description": desc,
                    "chunks": (IX.entity_chunks.get(name, {}) or {}).get("chunk_ids", [])[:5]})
    return out


@app.get("/search_relations")
def search_relations(q: str, k: int = 8):
    qv = EMB([q])[0]
    out = []
    for i, s in IX.topk(IX.rel_mat, qv, k):
        r = IX.rel_rows[i]
        out.append({"src": r.get("src_id"), "tgt": r.get("tgt_id"), "score": round(s, 4),
                    "content": (r.get("content") or "")[:200], "src_chunk": r.get("source_id")})
    return out


@app.get("/get_entity")
def get_entity(name: str, k: int = 30):
    vid = IX.vid.get(name)
    if vid is None:
        cand = [n for n in IX.vid if n.lower() == name.lower()]
        if cand:
            vid, name = IX.vid[cand[0]], cand[0]
        else:
            return JSONResponse({"error": f"entity not found: {name!r}"}, status_code=404)
    attrs = IX.g.vs[vid].attributes()
    rels = []
    for e in IX.g.incident(vid, mode="all")[:k]:
        edge = IX.g.es[e]
        other = edge.target if edge.source == vid else edge.source
        ea = edge.attributes()
        rels.append({"other": IX.g.vs[other]["id"],
                     "description": (ea.get("description") or "")[:200],
                     "keywords": ea.get("keywords"), "src_chunk": ea.get("source_id")})
    return {"entity": name, "type": attrs.get("entity_type"),
            "description": attrs.get("description"),
            "relations": rels,
            "chunks": (IX.entity_chunks.get(name, {}) or {}).get("chunk_ids", [])[:k]}


@app.get("/read_chunk")
def read_chunk(id: str):
    row = IX.text_chunks.get(id)
    if row is None:
        return JSONResponse({"error": f"chunk not found: {id}"}, status_code=404)
    d = chunk_payload(id)
    d["text"] = row.get("content", "")
    return d


@app.get("/retrieve")
def retrieve(q: str, k: int = 5):
    """hybrid 近似: top 实体+top 关系收集候选 chunk, 与 chunk 稠密分融合排序。"""
    qv = EMB([q])[0]
    votes = defaultdict(float)
    for i, s in IX.topk(IX.ent_mat, qv, 5):
        name = IX.ent_rows[i].get("entity_name")
        for cid in (IX.entity_chunks.get(name, {}) or {}).get("chunk_ids", [])[:5]:
            votes[cid] += s
    for i, s in IX.topk(IX.rel_mat, qv, 5):
        cid = IX.rel_rows[i].get("source_id")
        if cid:
            votes[cid] += s
    chunk_dense = {IX.chk_rows[i]["__id__"]: s for i, s in IX.topk(IX.chk_mat, qv, 20)}
    for cid, s in chunk_dense.items():
        votes[cid] += s
    top = sorted(votes.items(), key=lambda x: -x[1])[:k]
    return {"chunks": [chunk_payload(cid, sc, preview=300) for cid, sc in top]}


if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=ARGS.port, log_level="warning")
