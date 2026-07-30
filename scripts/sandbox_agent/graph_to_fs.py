#!/usr/bin/env python
"""端点② mapping: 把 HippoRAG 建的图物化成 agent 可探索的文件树(graphfs)。

映射关系(图元素 -> 文件系统元素):
  chunk 节点         -> {source}/docs/{dsid}.md 内的一段(带 <!-- chunk-id --> 锚)
  entity 节点        -> {home}/entities/{slug}.md 一页; home=提及最多的信源(多数票>=2/3),
                        跨源无主归属的实体上浮 _company/(全公司级概念)
  三元组关系边       -> 实体页 "Facts" 行: 谓词写回(图边只有 weight, 谓词从 openie 恢复) +
                        宾语 wikilink + 溯源 chunk 链接; 宾语页对称生成 "Facts (as object)"
  同义词边(小数权重) -> "See also" 别名链接(按权重取 top), 不物化 59 万条边本身
  entity-chunk 提及边-> 实体页 "Mentioned in" 反链 + doc 段尾 "Entities here" 正链

原则: 树管住址(路径有语义、与 sandbox flat 树同构), 链接管关系(图结构全在 wikilink)。
纯确定性转换, 无 LLM 成本。filemap_graphfs.json: doc 文件相对路径 -> chunk_ids(评 recall,
实体页不算 evidence)。

跑: python3 graph_to_fs.py --dataset enterprise_N1144 \
      [--graph_ws qa_scale_N1144] [--max_alias 15] [--home_ratio 0.667]
"""
import os
import re
import json
import pickle
import hashlib
import argparse
from collections import defaultdict, Counter

W = os.environ.get("URAG_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def norm(text):  # HippoRAG misc_utils.text_processing 同款归一化
    return re.sub("[^A-Za-z0-9 ]", " ", str(text).lower()).strip()


def slug(text, used):
    s = re.sub(r"[^a-z0-9]+", "-", norm(text)).strip("-")[:80]
    if not s:
        s = "e-" + hashlib.md5(text.encode()).hexdigest()[:8]
    if s in used:
        s = s[:71] + "-" + hashlib.md5(text.encode()).hexdigest()[:8]
    used.add(s)
    return s


def safe(name):
    return re.sub(r"[^A-Za-z0-9._-]", "_", str(name))[:120]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--graph_ws", default="", help="hipporag_ws 下的工作区名, 默认 qa_scale_N 或 enterprise_N 自动探测")
    ap.add_argument("--out", default="")
    ap.add_argument("--max_alias", type=int, default=15)
    ap.add_argument("--home_ratio", type=float, default=0.667, help="归属信源票数占比阈值, 不足则上浮 _company")
    args = ap.parse_args()

    n = args.dataset.split("_N")[-1]
    ws = args.graph_ws
    if not ws:
        for cand in (f"qa_scale_N{n}", f"enterprise_N{n}", args.dataset):
            if os.path.isdir(f"{W}/hipporag_ws/{cand}"):
                ws = cand
                break
    B = f"{W}/hipporag_ws/{ws}"
    sub = [d for d in os.listdir(B) if os.path.isdir(f"{B}/{d}") and d != "llm_cache"][0]
    ds = f"{W}/data/{args.dataset}"
    out = args.out or f"{W}/sandbox_agent/trees/{args.dataset}/graphfs"
    os.makedirs(out, exist_ok=True)

    # ---------- 载入 ----------
    g = pickle.load(open(f"{B}/{sub}/graph.pickle", "rb"))
    names = g.vs["name"]
    content = g.vs["content"]
    is_chunk = [x.startswith("chunk-") for x in names]
    openie = json.load(open(f"{B}/openie_results_ner_qwen3.6-27b.json"))["docs"]
    chunks = json.load(open(f"{ds}/chunks.json"))
    metas = [json.loads(l) for l in open(f"{ds}/chunks_meta.jsonl")]
    texts = [c if isinstance(c, str) else (c.get("text") or c.get("content")) for c in chunks]

    # chunk 哈希 -> 数据集元信息(md5(text) 即 HippoRAG 的 chunk id)
    c2meta = defaultdict(lambda: {"dsid": None, "source": None, "cids": []})
    doc_chunks = defaultdict(list)  # dsid -> [(顺序, hash, text)]
    for i, (t, m) in enumerate(zip(texts, metas)):
        h = "chunk-" + hashlib.md5(t.encode()).hexdigest()
        c2meta[h]["dsid"], c2meta[h]["source"] = m["dsid"], m["source"]
        c2meta[h]["cids"].append(m["chunk_id"])
        if not any(h == x[1] for x in doc_chunks[m["dsid"]]):
            doc_chunks[m["dsid"]].append((i, h, t))

    # 实体名(归一化 content) -> 顶点号
    ent_by_content = {content[i]: i for i in range(len(names)) if not is_chunk[i]}

    # ---------- 图边分拣 ----------
    ent_src = defaultdict(Counter)      # 实体 -> 提及信源票数
    mentions = defaultdict(set)         # 实体 -> {chunk hash}
    syn = defaultdict(list)             # 实体 -> [(权重, 邻居)] 小数权重=同义词边
    for e in g.es:
        s, t = e.source, e.target
        if is_chunk[s] != is_chunk[t]:
            ch, en = (s, t) if is_chunk[s] else (t, s)
            m = c2meta.get(names[ch])
            if m and m["source"]:
                ent_src[en][m["source"]] += 1
                mentions[en].add(names[ch])
        elif not is_chunk[s]:
            w = float(e["weight"])
            if w != int(w):
                syn[s].append((w, t))
                syn[t].append((w, s))

    # ---------- 实体归属 + 路径 ----------
    used, epath = set(), {}
    home_stat = Counter()
    for i in range(len(names)):
        if is_chunk[i]:
            continue
        cnt = ent_src.get(i)
        if cnt:
            top_src, top_n = cnt.most_common(1)[0]
            home = top_src if top_n >= sum(cnt.values()) * args.home_ratio else "_company"
        else:
            home = "_orphan"  # 图里无提及边的实体(理论上没有, 兜底)
        epath[i] = f"{home}/entities/{slug(content[i], used)}.md"
        home_stat[home] += 1

    def dpath(dsid):
        src = next((c2meta[h]["source"] for _, h, _ in doc_chunks[dsid]), "unknown")
        return f"{src}/docs/{safe(dsid)}.md"

    # ---------- 三元组: openie 恢复谓词, 溯源到 chunk ----------
    facts = defaultdict(dict)   # subj 顶点 -> {(pred, obj_key): {"obj": 顶点或原文, "src": set}}
    rfacts = defaultdict(dict)  # obj 顶点 -> 同上(反向)
    n_tri = n_linked = 0
    for doc in openie:
        h = doc["idx"]
        if h not in c2meta:
            continue
        for tri in doc.get("extracted_triples", []):
            if len(tri) != 3:
                continue
            s, p, o = (str(x) for x in tri)
            si, oi = ent_by_content.get(norm(s)), ent_by_content.get(norm(o))
            n_tri += 1
            if si is None:
                continue
            n_linked += oi is not None
            k = (p, oi if oi is not None else norm(o))
            f = facts[si].setdefault(k, {"obj": oi if oi is not None else o, "src": set()})
            f["src"].add(h)
            if oi is not None:
                rfacts[oi].setdefault((p, si), {"subj": si, "src": set()})["src"].add(h)

    # ---------- 写 doc 页(chunk 节点 -> 文档内锚点段) ----------
    filemap = {}
    for dsid, lst in doc_chunks.items():
        rel = dpath(dsid)
        os.makedirs(f"{out}/{os.path.dirname(rel)}", exist_ok=True)
        parts, cids = [], []
        for _, h, t in sorted(lst):
            cids += c2meta[h]["cids"]
            ents = sorted(
                (i for i in range(len(names)) if not is_chunk[i] and h in mentions.get(i, ()))
            ) if False else None  # 逐 chunk 扫全实体太慢, 下面用倒排
            parts.append(f"<!-- chunk-id: {h} -->\n{t}")
        filemap[rel] = sorted(set(cids))
        open(f"{out}/{rel}", "w").write("\n\n".join(parts) + "\n")

    # doc 页尾补 "Entities here"(用 mentions 倒排, 一次遍历)
    doc_ents = defaultdict(set)
    for en, hs in mentions.items():
        for h in hs:
            doc_ents[c2meta[h]["dsid"]].add(en)
    for dsid, ens in doc_ents.items():
        rel = dpath(dsid)
        links = " · ".join(f"[[{epath[i]}]]" for i in sorted(ens, key=lambda x: content[x])[:60])
        with open(f"{out}/{rel}", "a") as f:
            f.write(f"\n---\nEntities here: {links}\n")

    # ---------- 写实体页 ----------
    for i in range(len(names)):
        if is_chunk[i]:
            continue
        rel = epath[i]
        os.makedirs(f"{out}/{os.path.dirname(rel)}", exist_ok=True)
        L = [f"# {content[i] or '(empty entity)'}\n"]
        if facts.get(i):
            L.append("## Facts")
            for (p, _), f in sorted(facts[i].items(), key=lambda kv: kv[0][0]):
                o = f["obj"]
                otxt = f"[[{epath[o]}|{content[o]}]]" if isinstance(o, int) else str(o)
                srcs = " ".join(f"[[{dpath(c2meta[h]['dsid'])}#{h}]]" for h in sorted(f["src"])[:3])
                L.append(f"- {p} → {otxt}  (src: {srcs})")
            L.append("")
        if rfacts.get(i):
            L.append("## Facts (as object)")
            for (p, si), f in sorted(rfacts[i].items(), key=lambda kv: kv[0][0]):
                L.append(f"- [[{epath[si]}|{content[si]}]] {p} → this")
            L.append("")
        if syn.get(i):
            uniq = {}  # multigraph 同义边成对出现, 按邻居去重取最大权重
            for w, j in syn[i]:
                uniq[j] = max(w, uniq.get(j, 0))
            best = sorted(uniq.items(), key=lambda kv: -kv[1])[: args.max_alias]
            L.append("## See also (synonyms)")
            L.append(" · ".join(f"[[{epath[j]}|{content[j]}]]" for j, _ in best))
            L.append("")
        if mentions.get(i):
            L.append("## Mentioned in")
            docs = sorted({dpath(c2meta[h]["dsid"]) for h in mentions[i]})
            L.append(" · ".join(f"[[{d}]]" for d in docs))
            L.append("")
        open(f"{out}/{rel}", "w").write("\n".join(L))

    json.dump(filemap, open(f"{out}/../filemap_graphfs.json", "w"))
    n_ent = len(names) - sum(is_chunk)
    print(f"[graphfs] {args.dataset}: docs={len(doc_chunks)} chunks={sum(is_chunk)} entities={n_ent}")
    print(f"[graphfs] homes: {dict(home_stat.most_common())}")
    print(f"[graphfs] triples={n_tri} subj-linked={sum(len(v) for v in facts.values())} obj-also-entity={n_linked}")
    print(f"[graphfs] out={out}")


if __name__ == "__main__":
    main()
