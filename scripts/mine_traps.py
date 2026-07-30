#!/usr/bin/env python
"""第2-3步 v2：trap-mining + 污染扫描 → floor 的难度载体（doc 级 dsid）。
已修审查 BLOCKER/MINOR：
  B3 dense 加 Qwen3-Embed query instruction（query 侧，doc 侧不加，官方非对称）
  B4 embed 有限重试（指数退避）+ judge 重试
  m1 lure 封顶 LURE_CAP 且按 dense 相似度取最像的
  m2 info_not_found 污染扫描用更宽 prefilter（产物注明 top-K-bounded 非全库）
  m3 top_m / prefilter argparse 化（支持 M∈{5,10} 敏感性）
  m7 题级 ThreadPoolExecutor 并发（embed/LLM 是 IO，大幅提速）

输出: data/floor_traps{suffix}.json {traps:{qid:[dsid]}, lures:{qid:[dsid]}}
      data/contamination_blacklist{suffix}.json [{qid,dsid}]
跑: 带 rank_bm25+openai+pandas+numpy 的环境
"""
import sys
import os
import re
import json
import time
import argparse
import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
UNIFIED = os.environ.get("URAG_ROOT", os.path.dirname(SCRIPT_DIR))
sys.path.insert(0, SCRIPT_DIR)
import unified_config as U
ENT = os.environ.get(
    "ENTERPRISE_RAG_BENCH_ROOT",
    os.path.join(os.environ.get("PROJECT_ROOT", os.getcwd()), "EnterpriseRAG-Bench"),
)
QUESTIONS = os.environ.get("ENTERPRISE_QUESTIONS", os.path.join(ENT, "questions.jsonl"))

HARD_TYPES = {"constrained", "conflicting_info", "completeness"}
EMB_BATCH = 32
LURE_CAP = 5


def tok(s):
    return re.findall(r"[a-z0-9]+", s.lower())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top_m", type=int, default=10, help="dense/bm25 各取 top-M 并集（M敏感性）")
    ap.add_argument("--prefilter", type=int, default=200, help="难度型题 BM25 预筛池")
    ap.add_argument("--inf_prefilter", type=int, default=1000, help="info_not_found 污染扫描更宽预筛")
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--out_suffix", default="")
    args = ap.parse_args()

    print("[load] corpus_meta + questions ...", flush=True)
    meta = pd.read_parquet(os.path.join(U.DATA_ROOT, "enterprise_corpus_meta.parquet"))
    dsids = meta["dsid"].tolist()
    texts = meta["text"].tolist()
    qs = [json.loads(l) for l in open(QUESTIONS)]
    targets = [q for q in qs if q["question_type"] in HARD_TYPES or q["question_type"] == "info_not_found"]
    print(f"  corpus {len(dsids)}, 目标题 {len(targets)} (top_m={args.top_m})", flush=True)

    print("[bm25] 索引全 corpus（纯 CPU，约 10-30 分钟）...", flush=True)
    bm25 = BM25Okapi([tok(t) for t in texts])

    emb = OpenAI(base_url=U.EMBED_ENDPOINT, api_key="EMPTY", timeout=60.0, max_retries=2)
    llm = OpenAI(base_url=U.LLM_ENDPOINT, api_key="EMPTY", timeout=60.0, max_retries=2)

    def embed(txt_list, is_query=False):
        if is_query:                                        # B3：query 侧加 instruction
            txt_list = [U.EMBED_QUERY_INSTRUCTION + t for t in txt_list]
        out = []
        for i in range(0, len(txt_list), EMB_BATCH):
            batch = [t[:4000] for t in txt_list[i:i + EMB_BATCH]]
            for attempt in range(5):                        # B4：重试
                try:
                    r = emb.embeddings.create(model=U.EMBED_SERVED_NAME, input=batch)
                    out.extend([d.embedding for d in r.data])
                    break
                except Exception:
                    if attempt == 4:
                        raise
                    time.sleep(2 ** attempt)
        return np.array(out)

    def judge(prompt):
        for attempt in range(5):
            try:
                r = llm.chat.completions.create(
                    model=U.LLM_SERVED_NAME,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0, max_tokens=8, extra_body=U.LLM_EXTRA_BODY)
                return r.choices[0].message.content.strip().lower()
            except Exception:
                if attempt == 4:
                    return "[err]"
                time.sleep(2 ** attempt)

    def process(q):
        qt = q["question_type"]
        gold = set(q.get("expected_doc_ids", []) or [])
        pf = args.inf_prefilter if qt == "info_not_found" else args.prefilter
        scores = bm25.get_scores(tok(q["question"]))
        pre = [i for i in np.argsort(scores)[::-1][:pf] if dsids[i] not in gold]
        if not pre:
            return q["question_id"], [], [], []
        qv = embed([q["question"]], is_query=True)[0]
        cand_emb = embed([texts[i] for i in pre])
        sims = cand_emb @ qv
        sim_of = {pre[j]: float(sims[j]) for j in range(len(pre))}
        dense_top = [pre[j] for j in np.argsort(sims)[::-1][:args.top_m]]
        cands = list(dict.fromkeys(dense_top + pre[:args.top_m]))   # dense ∪ bm25
        facts = q.get("answer_facts", [])
        traps, lures, contam = [], [], []
        for i in cands:
            doc = texts[i][:1500]
            if qt == "info_not_found":
                v = judge(f"问题：{q['question']}\n\n文档：\n{doc}\n\n"
                          f"这份文档能直接回答上面的问题吗？只回答 yes 或 no。")
                if v.startswith("yes"):
                    contam.append({"qid": q["question_id"], "dsid": dsids[i]})
                elif v.startswith("no"):
                    lures.append((sim_of.get(i, 0.0), dsids[i]))    # 带相似度，后面 cap
            else:
                v = judge(f"问题：{q['question']}\n正确答案要点：{facts}\n\n候选文档：\n{doc}\n\n"
                          f"这份文档是否在谈同一主题/实体、但给的是错误/过时/不同版本的信息"
                          f"（即会把人诱导答错的干扰项），而不是正确答案？只回答 yes 或 no。")
                if v.startswith("yes"):
                    traps.append(dsids[i])
        lures = [d for _, d in sorted(lures, reverse=True)[:LURE_CAP]]  # m1：取最像的 top-CAP
        return q["question_id"], traps, lures, contam

    traps, lures, contamination = {}, {}, []
    done = 0
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:    # m7：并发
        for qid, tr, lu, co in ex.map(process, targets):
            if tr:
                traps[qid] = tr
            if lu:
                lures[qid] = lu
            contamination.extend(co)
            done += 1
            if done % 10 == 0:
                print(f"  {done}/{len(targets)} 题完成", flush=True)

    suf = args.out_suffix
    json.dump({"traps": traps, "lures": lures,
               "_meta": {"top_m": args.top_m, "prefilter": args.prefilter,
                         "inf_prefilter": args.inf_prefilter,
                         "contamination_scope": f"top-{args.inf_prefilter}-bounded (非全库)"}},
              open(os.path.join(U.DATA_ROOT, f"floor_traps{suf}.json"), "w"),
              ensure_ascii=False)
    json.dump(contamination,
              open(os.path.join(U.DATA_ROOT, f"contamination_blacklist{suf}.json"), "w"),
              ensure_ascii=False)
    print(f"[done] traps={sum(len(v) for v in traps.values())}(覆盖{len(traps)}题) "
          f"lures={sum(len(v) for v in lures.values())}(覆盖{len(lures)}题) "
          f"contam={len(contamination)}", flush=True)


if __name__ == "__main__":
    main()
