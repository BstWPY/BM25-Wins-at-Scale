#!/usr/bin/env python
"""统一评测 —— 所有 baseline 同一口径，杜绝各自评测标准不一致带来的不公平。

主指标:
  LLM-judge Acc : 本地 Qwen3.6 判语义对错（忽略 citation/表述/冗长），跨范式公平 —— 决策表主依据
辅助指标:
  EM / F1       : HotpotQA/SQuAD 标准 normalize，数据集原生、零成本、可复现
  Recall@k      : 基于统一 gold_chunk_ids 的检索覆盖

预测输入契约（predictions.jsonl 每行）：
  {"id": <qid>, "predicted_answer": "...", "retrieved_chunk_ids": [int, ...]}

用法:
  python evaluate.py --dataset 2wikimultihopqa --pred <path>            # 只算 EM/F1/Recall
  python evaluate.py --dataset 2wikimultihopqa --pred <path> --judge    # 追加 LLM-judge
"""
import re
import os
import json
import string
import argparse
import collections
import statistics
import concurrent.futures as cf

import sys
sys.path.insert(0, os.environ.get("URAG_ROOT", os.getcwd()))
import unified_config as U


def normalize(s):
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def em_score(pred, golds):
    return float(any(normalize(pred) == normalize(g) for g in golds))


def f1_score(pred, golds):
    def _f1(p, g):
        pt, gt = normalize(p).split(), normalize(g).split()
        common = collections.Counter(pt) & collections.Counter(gt)
        ns = sum(common.values())
        if ns == 0:
            return 0.0
        prec, rec = ns / len(pt), ns / len(gt)
        return 2 * prec * rec / (prec + rec)
    return max((_f1(pred, g) for g in golds), default=0.0)


JUDGE_PROMPT = """You are grading a question-answering system. Decide whether the PREDICTED answer is correct, given the GOLD answer(s).
Judge ONLY semantic correctness of the core answer. IGNORE formatting, citations like [Data: ...], extra explanation, verbosity, or paraphrasing. If the predicted answer contains the correct gold answer (or an equivalent), it is correct.

Question: {q}
Gold answer(s): {gold}
Predicted answer: {pred}

Reply with exactly one word: CORRECT or INCORRECT."""


def llm_judge_one(client, model, q, pred, golds):
    msg = [{"role": "user", "content": JUDGE_PROMPT.format(q=q, gold=" | ".join(golds), pred=pred)}]
    try:
        r = client.chat.completions.create(model=model, messages=msg, temperature=0,
                                            max_tokens=8, extra_body=U.LLM_EXTRA_BODY)
        out = r.choices[0].message.content.strip().upper()
        return 1.0 if ("INCORRECT" not in out and "CORRECT" in out) else 0.0
    except Exception:
        return 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--data_root", default=U.DATA_ROOT)
    ap.add_argument("--k", type=int, default=U.TOP_K)
    ap.add_argument("--judge", action="store_true", help="追加 LLM-judge（本地 Qwen3.6）")
    ap.add_argument("--judge_workers", type=int, default=16)
    args = ap.parse_args()

    gold = {q["id"]: q for q in (json.loads(l) for l in open(f"{args.data_root}/{args.dataset}/questions.jsonl"))}
    preds = [json.loads(l) for l in open(args.pred)]

    ems, f1s, recalls, all_hits = [], [], [], []
    judge_items = []
    n_retr = 0
    for p in preds:
        q = gold.get(p["id"])
        if q is None:
            continue
        ems.append(em_score(p["predicted_answer"], q["answer"]))
        f1s.append(f1_score(p["predicted_answer"], q["answer"]))
        judge_items.append((q["question"], p["predicted_answer"], q["answer"]))
        if "retrieved_chunk_ids" in p and q["gold_chunk_ids"]:
            n_retr += 1
            topk = set(p["retrieved_chunk_ids"][:args.k])
            gset = set(q["gold_chunk_ids"])
            recalls.append(len(topk & gset) / len(gset))
            all_hits.append(float(gset <= topk))

    n = len(ems)
    print(f"==== {args.dataset}  ({n} questions) ====")
    judge_acc = None
    if args.judge:
        from openai import OpenAI
        client = OpenAI(base_url=U.LLM_ENDPOINT, api_key="EMPTY", timeout=600.0, max_retries=5)
        with cf.ThreadPoolExecutor(args.judge_workers) as ex:
            verdicts = list(ex.map(lambda it: llm_judge_one(client, U.LLM_SERVED_NAME, *it), judge_items))
        judge_acc = statistics.mean(verdicts) * 100
        print(f"  LLM-judge Acc : {judge_acc:.2f}   ★ 主指标")
    print(f"  EM            : {statistics.mean(ems)*100:.2f}")
    print(f"  F1            : {statistics.mean(f1s)*100:.2f}")
    if n_retr:
        print(f"  Recall@{args.k} (cov) : {statistics.mean(recalls)*100:.2f}   (over {n_retr} q)")
        print(f"  AllGold@{args.k}      : {statistics.mean(all_hits)*100:.2f}")
    return judge_acc


if __name__ == "__main__":
    main()
