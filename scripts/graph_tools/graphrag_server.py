#!/usr/bin/env python
"""MS GraphRAG index server —— parquet + lancedb 常驻内存, HTTP 工具暴露(端点② agentic+graph)。

数据源(graphrag_ws/{dataset}/output/):
  entities/relationships/communities/community_reports/text_units.parquet
  lancedb: entity_description / community_full_content / text_unit_text 三张向量表(id 对齐)
recall 映射: text_unit.text 内容 md5 -> unified chunk_ids(切分同源, 应≈全命中)。
启动时 parity 自检: 用 CPU Qwen3-embed 复现库内向量, 打印两种拼接口径的 cos, 取高者为查询口径。

跑: envs/vllm_judge/bin/python graphrag_server.py --dataset enterprise_N1144 --port 8803
"""
import os, sys, json, hashlib, argparse
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from fastapi import FastAPI
from fastapi.responses import JSONResponse
import uvicorn
from transformers import AutoTokenizer, AutoModel
import lancedb

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


class Index:
    def __init__(self, dataset):
        B = f"{W}/graphrag_ws/{dataset}/output"
        print(f"[load] {B}", flush=True)
        self.ent = pd.read_parquet(f"{B}/entities.parquet")
        self.rel = pd.read_parquet(f"{B}/relationships.parquet")
        self.rep = pd.read_parquet(f"{B}/community_reports.parquet")
        self.tu = pd.read_parquet(f"{B}/text_units.parquet")
        db = lancedb.connect(f"{B}/lancedb")

        def vecs(table, ids):
            df = db.open_table(table).to_pandas()[["id", "vector"]]
            m = dict(zip(df["id"], df["vector"]))
            mat = np.stack([np.asarray(m[i], dtype=np.float32) for i in ids])
            mat /= np.clip(np.linalg.norm(mat, axis=1, keepdims=True), 1e-9, None)
            return mat
        self.ent_mat = vecs("entity_description", self.ent["id"].tolist())
        self.rep_mat = vecs("community_full_content", self.rep["id"].tolist())
        self.tu_mat = vecs("text_unit_text", self.tu["id"].tolist())

        self.ent_by_title = {t.lower(): i for i, t in enumerate(self.ent["title"])}
        self.adj = defaultdict(list)                     # entity title(lower) -> rel row idx
        for i, r in self.rel.iterrows():
            self.adj[str(r["source"]).lower()].append(i)
            self.adj[str(r["target"]).lower()].append(i)
        self.rep_by_comm = {int(r["community"]): i for i, r in self.rep.iterrows()}
        self.tu_by_id = {r["id"]: i for i, r in self.tu.iterrows()}

        # text_unit -> unified chunk_ids(内容 md5)
        ds = f"{W}/data/{dataset}"
        chunks = json.load(open(f"{ds}/chunks.json"))
        metas = [json.loads(l) for l in open(f"{ds}/chunks_meta.jsonl")]
        cmd5 = defaultdict(list)
        for t, m in zip(chunks, metas):
            cmd5[hashlib.md5(t.encode()).hexdigest()].append(int(m["chunk_id"]))
        self.tu_map = {}
        hit = 0
        for _, r in self.tu.iterrows():
            ids = cmd5.get(hashlib.md5(str(r["text"]).encode()).hexdigest(), [])
            hit += bool(ids)
            self.tu_map[r["id"]] = ids
        print(f"[map] text_units md5 -> unified chunks: {hit}/{len(self.tu)}", flush=True)

    def topk(self, mat, qv, k):
        s = mat @ qv
        idx = np.argpartition(-s, min(k, len(s) - 1))[:k]
        return sorted(((int(i), float(s[i])) for i in idx), key=lambda x: -x[1])


ap = argparse.ArgumentParser()
ap.add_argument("--dataset", default="enterprise_N1144")
ap.add_argument("--port", type=int, default=8803)
ARGS = ap.parse_args()
IX = Index(ARGS.dataset)
EMB = Embedder(EMBED_MODEL)

# parity 自检: 两种拼接口径, 取 cos 高者作为查询空间的既定口径记录
_r = IX.ent.iloc[0]
for lbl, txt in [("title:description", f"{_r['title']}:{_r['description']}"),
                 ("description", str(_r["description"]))]:
    c = float(EMB([txt])[0] @ IX.ent_mat[0])
    print(f"[parity] {lbl}: cos={c:.4f}", flush=True)
print("[ready]", flush=True)
app = FastAPI()


def aslist(x):
    if x is None:
        return []
    return list(x)


def tu_payload(tuid, score=None, preview=0):
    i = IX.tu_by_id.get(tuid)
    if i is None:
        return {"text_unit": tuid, "error": "unknown"}
    r = IX.tu.iloc[i]
    d = {"text_unit": tuid, "chunk_ids": IX.tu_map.get(tuid, [])}
    if score is not None:
        d["score"] = round(score, 6)
    if preview:
        d["preview"] = str(r["text"])[:preview]
    return d


@app.get("/health")
def health():
    return {"dataset": ARGS.dataset, "entities": len(IX.ent), "relationships": len(IX.rel),
            "reports": len(IX.rep), "text_units": len(IX.tu)}


@app.get("/search_entities")
def search_entities(q: str, k: int = 8):
    qv = EMB([q])[0]
    out = []
    for i, s in IX.topk(IX.ent_mat, qv, k):
        r = IX.ent.iloc[i]
        out.append({"entity": r["title"], "type": r["type"], "score": round(s, 4),
                    "degree": int(r["degree"]),
                    "description": str(r["description"])[:250]})
    return out


@app.get("/get_entity")
def get_entity(name: str, k: int = 20):
    i = IX.ent_by_title.get(name.lower())
    if i is None:
        return JSONResponse({"error": f"entity not found: {name!r}"}, status_code=404)
    r = IX.ent.iloc[i]
    rels = []
    for j in IX.adj.get(name.lower(), [])[:k]:
        e = IX.rel.iloc[j]
        rels.append({"source": e["source"], "target": e["target"],
                     "description": str(e["description"])[:200], "weight": float(e["weight"])})
    return {"entity": r["title"], "type": r["type"], "description": str(r["description"]),
            "relations": rels,
            "text_units": [tu_payload(t) for t in aslist(r["text_unit_ids"])[:10]]}


@app.get("/search_reports")
def search_reports(q: str, k: int = 5, level: int = -1):
    qv = EMB([q])[0]
    out = []
    for i, s in IX.topk(IX.rep_mat, qv, k * 3):
        r = IX.rep.iloc[i]
        if level >= 0 and int(r["level"]) != level:
            continue
        out.append({"community": int(r["community"]), "level": int(r["level"]),
                    "title": r["title"], "score": round(s, 4),
                    "summary": str(r["summary"])[:300]})
        if len(out) >= k:
            break
    return out


@app.get("/read_report")
def read_report(community: int):
    i = IX.rep_by_comm.get(int(community))
    if i is None:
        return JSONResponse({"error": f"community not found: {community}"}, status_code=404)
    r = IX.rep.iloc[i]
    return {"community": int(r["community"]), "level": int(r["level"]), "title": r["title"],
            "full_content": str(r["full_content"])}


@app.get("/search_chunks")
def search_chunks(q: str, k: int = 5):
    qv = EMB([q])[0]
    return [tu_payload(IX.tu.iloc[i]["id"], s, preview=200) for i, s in IX.topk(IX.tu_mat, qv, k)]


@app.get("/read_chunk")
def read_chunk(id: str):
    i = IX.tu_by_id.get(id)
    if i is None:
        return JSONResponse({"error": f"text unit not found: {id}"}, status_code=404)
    d = tu_payload(id)
    d["text"] = str(IX.tu.iloc[i]["text"])
    return d


@app.get("/retrieve")
def retrieve(q: str, k: int = 5):
    """local-search 近似: top 实体投票其 text_units + chunk 稠密分融合; 附最相关社区报告。"""
    qv = EMB([q])[0]
    votes = defaultdict(float)
    ent_hits = IX.topk(IX.ent_mat, qv, 5)
    for i, s in ent_hits:
        for t in aslist(IX.ent.iloc[i]["text_unit_ids"])[:8]:
            votes[t] += s
    for i, s in IX.topk(IX.tu_mat, qv, 20):
        votes[IX.tu.iloc[i]["id"]] += s
    top = sorted(votes.items(), key=lambda x: -x[1])[:k]
    rep = IX.topk(IX.rep_mat, qv, 1)
    rr = IX.rep.iloc[rep[0][0]] if rep else None
    return {"entities": [{"entity": IX.ent.iloc[i]["title"], "score": round(s, 4)} for i, s in ent_hits],
            "chunks": [tu_payload(t, sc, preview=300) for t, sc in top],
            "top_report": ({"community": int(rr["community"]), "title": rr["title"],
                            "summary": str(rr["summary"])[:200]} if rr is not None else None)}


if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=ARGS.port, log_level="warning")
