#!/usr/bin/env python
"""HippoRAG 图 index server —— 把已建索引(graph.pickle + vdb parquet + openie)常驻内存,
以 HTTP 工具形式暴露给 agent(端点② agentic+graph)。纯只读、确定性、CPU 可跑。

与原 pipeline 同口径的关键点:
  - 查询嵌入 = 裸文本(该 fork 的 OpenAI.batch_encode 中 instruction 是死代码, 实测库内向量
    与裸文本 cos=0.9999), \n→空格、空串→" ", L2 归一;
  - PPR: igraph personalized_pagerank, damping=0.5(config 默认), 边权 float 化;
  - retrieve = 原生一轮式排序器(query→top facts→种子实体加权→PPR(+passage 稠密混权)→top-k chunk),
    agent 调它一次≈一轮式 HippoRAG, 也可用细粒度工具自行组合迭代。

跑: envs/vllm_judge/bin/python hippo_server.py --dataset enterprise_N1144 --port 8801
"""
import os, sys, json, ast, pickle, hashlib, argparse, re
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from fastapi import FastAPI
from fastapi.responses import JSONResponse
import uvicorn
from transformers import AutoTokenizer, AutoModel

W = os.environ.get("URAG_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EMBED_MODEL = os.environ.get("EMBED_MODEL") or os.path.join(
    os.path.dirname(os.path.dirname(W)), "models", "Qwen3-Embedding-0.6B")


def norm_text(t):  # HippoRAG misc_utils.text_processing
    return re.sub("[^A-Za-z0-9 ]", " ", str(t).lower()).strip()


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
        n = dataset.split("_N")[-1]
        ws = next(f"{W}/hipporag_ws/{c}" for c in (f"qa_scale_N{n}", f"enterprise_N{n}", dataset)
                  if os.path.isdir(f"{W}/hipporag_ws/{c}"))
        sub = next(d for d in os.listdir(ws) if os.path.isdir(f"{ws}/{d}") and d != "llm_cache")
        B = f"{ws}/{sub}"
        print(f"[load] {B}", flush=True)

        g = pickle.load(open(f"{B}/graph.pickle", "rb"))
        g.es["weight"] = [float(w) for w in g.es["weight"]]
        g.simplify(combine_edges={"weight": "sum"})       # multigraph 成对边合并
        self.g = g
        self.names = g.vs["name"]
        self.content = g.vs["content"]
        self.is_chunk = np.array([x.startswith("chunk-") for x in self.names])
        self.vid_by_content = {self.content[i]: i for i in range(g.vcount()) if not self.is_chunk[i]}
        self.vid_by_name = {self.names[i]: i for i in range(g.vcount())}
        self.chunk_vids = np.where(self.is_chunk)[0]

        def load_vdb(kind):
            df = pd.read_parquet(f"{B}/{kind}_embeddings/vdb_{kind}.parquet")
            mat = np.stack(df["embedding"].to_numpy()).astype(np.float32)
            mat /= np.linalg.norm(mat, axis=1, keepdims=True)
            return df["content"].tolist(), df["hash_id"].tolist(), mat
        self.ent_txt, _, self.ent_mat = load_vdb("entity")
        self.fact_txt, _, self.fact_mat = load_vdb("fact")
        self.chk_txt, self.chk_ids, self.chk_mat = load_vdb("chunk")
        self.chk_vid = {h: self.vid_by_name.get(h) for h in self.chk_ids}

        # 三元组解析 + openie 溯源(标准化三元组 -> chunk hashes)
        self.facts = []                                    # (s,p,o) 原文
        for t in self.fact_txt:
            try:
                s, p, o = ast.literal_eval(t)
            except Exception:
                s, p, o = str(t), "", ""
            self.facts.append((str(s), str(p), str(o)))
        src = defaultdict(set)
        openie = json.load(open(f"{ws}/openie_results_ner_qwen3.6-27b.json"))["docs"]
        for doc in openie:
            for tri in doc.get("extracted_triples", []):
                if len(tri) == 3:
                    src[tuple(norm_text(x) for x in tri)].add(doc["idx"])
        self.fact_src = [sorted(src.get(tuple(norm_text(x) for x in f), []))[:3] for f in self.facts]
        # 实体 -> 出边/入边三元组下标
        self.out_facts, self.in_facts = defaultdict(list), defaultdict(list)
        for i, (s, p, o) in enumerate(self.facts):
            self.out_facts[norm_text(s)].append(i)
            self.in_facts[norm_text(o)].append(i)

        # chunk 元信息(md5(text) 对回数据集)
        ds = f"{W}/data/{dataset}"
        chunks = json.load(open(f"{ds}/chunks.json"))
        metas = [json.loads(l) for l in open(f"{ds}/chunks_meta.jsonl")]
        self.chk_meta = {}
        for t, m in zip(chunks, metas):
            h = "chunk-" + hashlib.md5((t if isinstance(t, str) else t.get("text", "")).encode()).hexdigest()
            self.chk_meta.setdefault(h, {"dsid": m["dsid"], "source": m["source"],
                                         "cids": []})["cids"].append(int(m["chunk_id"]))
        self.chk_text = {h: t for h, t in zip(self.chk_ids, self.chk_txt)}

    # ---------- 检索原语 ----------
    def topk(self, mat, qv, k):
        s = mat @ qv
        idx = np.argpartition(-s, min(k, len(s) - 1))[:k]
        return sorted(((int(i), float(s[i])) for i in idx), key=lambda x: -x[1])

    def ppr(self, seed_weights, damping=0.5, k=5, passage_reset=None):
        reset = np.zeros(self.g.vcount())
        for vid, w in seed_weights.items():
            reset[vid] += w
        if passage_reset is not None:
            for vid, w in passage_reset.items():
                reset[vid] += w
        if reset.sum() <= 0:
            return []
        pr = self.g.personalized_pagerank(damping=damping, directed=False,
                                          weights="weight", reset=reset.tolist(),
                                          implementation="prpack")
        pr = np.asarray(pr)
        cv = self.chunk_vids
        order = cv[np.argsort(-pr[cv])][:k]
        return [(self.names[v], float(pr[v])) for v in order]


ap = argparse.ArgumentParser()
ap.add_argument("--dataset", default="enterprise_N1144")
ap.add_argument("--port", type=int, default=8801)
ARGS = ap.parse_args()
IX = Index(ARGS.dataset)
EMB = Embedder(EMBED_MODEL)
print("[ready]", flush=True)
app = FastAPI()


def chunk_payload(h, score=None, preview=0):
    m = IX.chk_meta.get(h, {})
    d = {"chunk": h, "dsid": m.get("dsid"), "source": m.get("source"),
         "chunk_ids": m.get("cids", [])}
    if score is not None:
        d["score"] = round(score, 6)
    if preview:
        d["preview"] = (IX.chk_text.get(h) or "")[:preview]
    return d


@app.get("/health")
def health():
    return {"dataset": ARGS.dataset, "vertices": IX.g.vcount(), "edges": IX.g.ecount(),
            "entities": int((~IX.is_chunk).sum()), "chunks": int(IX.is_chunk.sum()),
            "facts": len(IX.facts)}


@app.get("/search_entities")
def search_entities(q: str, k: int = 8):
    qv = EMB([q])[0]
    deg = IX.g.degree()
    out = []
    for i, s in IX.topk(IX.ent_mat, qv, k):
        ent = IX.ent_txt[i]
        vid = IX.vid_by_content.get(ent)
        out.append({"entity": ent, "score": round(s, 4),
                    "degree": deg[vid] if vid is not None else 0,
                    "n_facts": len(IX.out_facts.get(norm_text(ent), [])) +
                               len(IX.in_facts.get(norm_text(ent), []))})
    return out


@app.get("/search_facts")
def search_facts(q: str, k: int = 8):
    qv = EMB([q])[0]
    out = []
    for i, s in IX.topk(IX.fact_mat, qv, k):
        su, p, o = IX.facts[i]
        out.append({"subject": su, "predicate": p, "object": o, "score": round(s, 4),
                    "src_chunks": IX.fact_src[i]})
    return out


@app.get("/neighbors")
def neighbors(entity: str, k: int = 30):
    key = norm_text(entity)
    vid = IX.vid_by_content.get(key)
    if vid is None:
        return JSONResponse({"error": f"entity not found: {entity!r} (normalized {key!r})"}, status_code=404)
    fo = [{"predicate": IX.facts[i][1], "object": IX.facts[i][2], "src_chunks": IX.fact_src[i]}
          for i in IX.out_facts.get(key, [])[:k]]
    fi = [{"subject": IX.facts[i][0], "predicate": IX.facts[i][1], "src_chunks": IX.fact_src[i]}
          for i in IX.in_facts.get(key, [])[:k]]
    syn, men = [], []
    for e in IX.g.incident(vid):
        edge = IX.g.es[e]
        other = edge.target if edge.source == vid else edge.source
        w = edge["weight"]
        if IX.is_chunk[other]:
            men.append(IX.names[other])
        elif w != int(w):                       # 小数权重=同义词边
            syn.append((w, IX.content[other]))
    syn = [t for _, t in sorted(syn, reverse=True)[:15]]
    return {"entity": IX.content[vid], "facts_out": fo, "facts_in": fi,
            "synonyms": syn,
            "mentioned_in": [chunk_payload(h) for h in men[:k]]}


@app.get("/ppr")
def ppr(seeds: str, k: int = 5, damping: float = 0.5):
    sw = {}
    missing = []
    for s in seeds.split("|"):
        vid = IX.vid_by_content.get(norm_text(s))
        if vid is None:
            missing.append(s)
        else:
            sw[vid] = sw.get(vid, 0.0) + 1.0
    if not sw:
        return JSONResponse({"error": f"no valid seeds, missing={missing}"}, status_code=404)
    res = [chunk_payload(h, sc, preview=200) for h, sc in IX.ppr(sw, damping, k)]
    return {"seeds_used": [IX.content[v] for v in sw], "missing": missing, "chunks": res}


@app.get("/read_chunk")
def read_chunk(id: str):
    if id not in IX.chk_text:
        return JSONResponse({"error": f"chunk not found: {id}"}, status_code=404)
    d = chunk_payload(id)
    d["text"] = IX.chk_text[id]
    return d


@app.get("/retrieve")
def retrieve(q: str, k: int = 5, link_top_k: int = 5, passage_node_weight: float = 0.05,
             damping: float = 0.5):
    """原生一轮式排序器: query→top facts→种子实体(fact 分数加权)→PPR(+chunk 稠密混权)→top-k。"""
    qv = EMB([q])[0]
    seeds = defaultdict(float)
    top_facts = IX.topk(IX.fact_mat, qv, link_top_k)
    for i, s in top_facts:
        su, _, o = IX.facts[i]
        for e in (su, o):
            vid = IX.vid_by_content.get(norm_text(e))
            if vid is not None:
                seeds[vid] += max(s, 0.0)
        # chunk 稠密分数按 passage_node_weight 混入 reset(近似原生 dpr 混权)
    ch = IX.topk(IX.chk_mat, qv, max(k * 4, 20))
    smax = max((s for _, s in ch), default=1.0) or 1.0
    preset = {}
    for i, s in ch:
        vid = IX.chk_vid.get(IX.chk_ids[i])
        if vid is not None and s > 0:
            preset[vid] = passage_node_weight * s / smax
    chunks = [chunk_payload(h, sc, preview=300) for h, sc in IX.ppr(dict(seeds), damping, k, preset)]
    return {"linked_facts": [{"fact": IX.facts[i], "score": round(s, 4)} for i, s in top_facts],
            "chunks": chunks}


if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=ARGS.port, log_level="warning")
