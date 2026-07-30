#!/usr/bin/env python
"""EnterpriseRAG 规模 scaling 语料生成 v2。详见 DESIGN_scaling_enterprise.md。

依赖：
  第1步 data/enterprise_corpus_meta.parquet  (dsid/source/is_noise/token_len/text)
  第2-3步 data/floor_traps.json + data/contamination_blacklist.json

产物：data/enterprise_N{size}/ × 28  (chunks.json + chunks_meta.jsonl + questions.jsonl
       + manifest.json[sorted dsid+sha256] + scale_report.json)

核心：floor(难度基岩,固定) + ocean(单种子双分层前缀,保证嵌套+组成恒定) → 28 log 点。
跑：python pack_corpus_enterprise.py [--npoints 28]
"""
import os
import sys
import json
import random
import hashlib
import argparse
from collections import defaultdict, Counter

import pandas as pd
import tiktoken

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
UNIFIED = os.environ.get("URAG_ROOT", os.path.dirname(SCRIPT_DIR))
sys.path.insert(0, SCRIPT_DIR)
import unified_config as U
ENT = os.environ.get(
    "ENTERPRISE_RAG_BENCH_ROOT",
    os.path.join(os.environ.get("PROJECT_ROOT", os.getcwd()), "EnterpriseRAG-Bench"),
)
QUESTIONS = os.environ.get("ENTERPRISE_QUESTIONS", os.path.join(ENT, "questions.jsonl"))

SEED = 42
CHUNK_SIZE, CHUNK_OVERLAP, ENCODING = 1200, 100, "o200k_base"
HIST_TOL_ABS = 0.02        # 直方图绝对容差
HIST_TOL_REL = 0.10        # m5：相对容差（对稀有 source 的有效 backstop）
enc = tiktoken.get_encoding(ENCODING)


def chunk_body(text):
    toks = enc.encode(text, disallowed_special=())
    if len(toks) <= CHUNK_SIZE:
        return [text]
    out, i = [], 0
    while i < len(toks):
        out.append(enc.decode(toks[i:i + CHUNK_SIZE]))
        if i + CHUNK_SIZE >= len(toks):
            break
        i += CHUNK_SIZE - CHUNK_OVERLAP
    return out


def dump_point(dsids, N, text, source, is_noise, qs, out_root):
    """切 chunk + 映射 gold(doc级) + 写该规模点全部产物。"""
    chunks, meta_rows = [], []
    ds2chunks = defaultdict(list)
    for d in dsids:
        for piece in chunk_body(text[d]):
            cid = len(chunks)
            chunks.append(piece)
            meta_rows.append({"chunk_id": cid, "dsid": d,
                              "source": source[d], "is_noise": bool(is_noise[d])})
            ds2chunks[d].append(cid)
    dset = set(dsids)
    q_out = []
    for q in qs:
        gd = [d for d in (q.get("expected_doc_ids") or []) if d in dset]
        gc = [c for d in gd for c in ds2chunks[d]]
        ans = q.get("gold_answer", "")
        q_out.append({"id": q["question_id"], "question": q["question"],
                      "answer": [ans] if isinstance(ans, str) else ans,
                      "question_type": q.get("question_type"),
                      "answer_facts": q.get("answer_facts", []),
                      "gold_dsids": gd, "gold_chunk_ids": gc,
                      "has_gold": bool(q.get("expected_doc_ids"))})
    od = f"{out_root}/enterprise_N{N}"
    os.makedirs(od, exist_ok=True)
    json.dump(chunks, open(f"{od}/chunks.json", "w"), ensure_ascii=False)
    with open(f"{od}/chunks_meta.jsonl", "w") as f:
        for m in meta_rows:
            f.write(json.dumps(m) + "\n")
    with open(f"{od}/questions.jsonl", "w") as f:
        for q in q_out:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")
    sorted_ds = sorted(dsids)
    sha = hashlib.sha256("\n".join(sorted_ds).encode()).hexdigest()
    json.dump({"N_docs": len(dsids), "sha256": sha, "dsids": sorted_ds},
              open(f"{od}/manifest.json", "w"))
    src_dist = Counter(source[d] for d in dsids)
    report = {"N_docs": len(dsids), "N_chunks": len(chunks),
              "noise_frac": round(sum(bool(is_noise[d]) for d in dsids) / len(dsids), 4),
              "source_dist": {s: round(c / len(dsids), 4) for s, c in src_dist.most_common()},
              "q_with_gold": sum(q["has_gold"] for q in q_out),
              "manifest_sha256": sha}
    json.dump(report, open(f"{od}/scale_report.json", "w"), indent=2)
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npoints", type=int, default=28)
    ap.add_argument("--out_root", default=U.DATA_ROOT)
    args = ap.parse_args()

    print("[load] corpus_meta + questions + traps ...", flush=True)
    meta = pd.read_parquet(os.path.join(U.DATA_ROOT, "enterprise_corpus_meta.parquet"))
    all_dsids = meta["dsid"].tolist()
    text = dict(zip(meta["dsid"], meta["text"]))
    source = dict(zip(meta["dsid"], meta["source"]))
    is_noise = dict(zip(meta["dsid"], meta["is_noise"]))
    all_set = set(all_dsids)
    qs = [json.loads(l) for l in open(QUESTIONS)]

    # ---- FLOOR ----
    gold = set()
    for q in qs:
        gold.update(q.get("expected_doc_ids", []) or [])
    gold &= all_set
    # scaffold 合成 dsid 注入（company_overview/initiatives 无 dsid）
    scaffold = {}
    for syn, path in [("__scaffold_company_overview", f"{ENT}/generated_data/company_overview.md"),
                      ("__scaffold_initiatives", f"{ENT}/generated_data/initiatives.md")]:
        t = open(path).read().strip()
        text[syn] = t
        source[syn] = "scaffold"
        is_noise[syn] = False
        scaffold[syn] = t
    tr = json.load(open(os.path.join(U.DATA_ROOT, "floor_traps.json")))
    trap_ds = ({d for v in tr.get("traps", {}).values() for d in v}
               | {d for v in tr.get("lures", {}).values() for d in v}) & all_set
    contam_path = os.path.join(U.DATA_ROOT, "contamination_blacklist.json")
    contam = {x["dsid"] for x in json.load(open(contam_path))}
    # B1：contam 绝不能删真 gold（某文档可能既被判某 info_not_found 题"可答"、又是另一题 gold）
    assert not (contam & gold), f"contam 误含 {len(contam & gold)} 个真 gold（污染判定有误）"
    contam_ocean = contam - gold                  # 只从 ocean 排除"非 gold"的污染文档
    floor = gold | set(scaffold) | trap_ds        # floor 不减 contam（gold/trap 优先级最高）
    gold_raw = set()
    for q in qs:
        gold_raw.update(q.get("expected_doc_ids", []) or [])
    assert not (gold_raw - all_set), f"{len(gold_raw - all_set)} 个 gold 不在 corpus_meta（扫描丢了文档）"
    print(f"[floor] {len(floor)} = gold {len(gold)} + scaffold {len(scaffold)} + trap {len(trap_ds)}", flush=True)

    # ---- OCEAN: 单种子 source×noise 双分层前缀（任意前缀直方图≈全局边际）----
    ocean = [d for d in all_dsids if d not in floor and d not in contam_ocean]
    rng = random.Random(SEED)
    buckets = defaultdict(list)
    for d in ocean:
        buckets[(source[d], bool(is_noise[d]))].append(d)
    keyed = []
    for b, docs in buckets.items():
        rng.shuffle(docs)
        n = len(docs)
        for i, d in enumerate(docs):
            keyed.append(((i + 0.5) / n, d))       # 归一化分层位置 → 前缀按比例
    keyed.sort()
    ocean_order = [d for _, d in keyed]
    assert set(ocean_order) == set(ocean) and len(ocean_order) == len(set(ocean_order)), \
        "ocean_order 构造错误（漏/重复 doc）"          # m6：独立核验构造正确性，非平凡前缀
    glob_src = Counter(source[d] for d in ocean)
    glob_noise = sum(bool(is_noise[d]) for d in ocean) / len(ocean)

    # ---- 28 log 点 ----
    lo, hi = len(floor), len(floor) + len(ocean)
    k = args.npoints
    sizes = sorted(set(int(round(lo * (hi / lo) ** (i / (k - 1)))) for i in range(k)))
    floor_list = sorted(floor)
    print(f"[points] {len(sizes)} 个: {sizes}", flush=True)

    # ---- 逐点：验证门禁 + dump ----
    prev = set()
    for N in sizes:
        m = max(0, N - len(floor))
        pts = floor_list + ocean_order[:m]
        pset = set(pts)
        assert floor.issubset(pset), f"floor⊄N{N}"
        assert prev.issubset(pset), f"非嵌套 break at N{N}"
        # ocean 直方图门禁（只查 ocean 部分；floor 偏斜是已披露的固定 offset）
        if m > 0:                                          # m6：所有有 ocean 的点都验
            osrc = Counter(source[d] for d in ocean_order[:m])
            bad = []
            for s, gc in glob_src.items():
                target, got = gc / len(ocean), osrc.get(s, 0) / m
                tol = max(HIST_TOL_ABS, HIST_TOL_REL * target)   # m5：绝对/相对取大
                if abs(got - target) > tol:
                    bad.append(f"{s}={got:.4f}≠{target:.4f}")
            onoise = sum(bool(is_noise[d]) for d in ocean_order[:m]) / m
            if abs(onoise - glob_noise) > HIST_TOL_ABS:
                bad.append(f"noise={onoise:.4f}≠{glob_noise:.4f}")
            if N < 2 * len(floor):                          # B2：floor-dominated 低点只警告
                if bad:
                    print(f"  ⚠ N{N}(floor-dominated,禁报crossover) 直方图: {bad}", flush=True)
            else:
                assert not bad, f"N{N} 直方图门禁失败: {bad}"   # B2：其余硬停
        prev = pset
        rep = dump_point(pts, N, text, source, is_noise, qs, args.out_root)
        print(f"  N={N}: {rep['N_docs']}d/{rep['N_chunks']}c noise={rep['noise_frac']} "
              f"qgold={rep['q_with_gold']} top3={list(rep['source_dist'].items())[:3]}", flush=True)
    print("[done] 全部规模点生成 + 嵌套/直方图门禁通过", flush=True)


if __name__ == "__main__":
    main()
