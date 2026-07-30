#!/usr/bin/env python
"""预计算每个 scale 点的原始语料 token 总数(各点 manifest 的 dsids 在 corpus_meta 上求和 token_len)。
原始语料 token 是比文档数更本质、跨范式统一的规模度量:建图按 chunk/内容算成本，文档长度跨源漂移(实测小点1473→大点1173 tok/doc)，
用文档数当横轴会扭曲成本曲线。chunked 方法和不切块的 sandbox 都能用原始 token 统一。见 DESIGN 第7节(按chunk数非doc数拟合)。
输出 data/scale_corpus_tokens.json: {str(N_docs): {corpus_tokens, n_docs, n_chunks, mean_doc_token}}。
跑: zzt python compute_corpus_tokens.py
"""
import os, json, glob
import pandas as pd

W = os.environ.get(
    "URAG_ROOT",
    os.path.join(os.environ.get("PROJECT_ROOT", os.getcwd()), "GraphRAG_fs_mapping", "unified"),
)
META = f"{W}/data/enterprise_corpus_meta.parquet"

print("[load] corpus_meta (dsid, token_len) ...", flush=True)
df = pd.read_parquet(META, columns=["dsid", "token_len"])
tok = dict(zip(df["dsid"].astype(str), df["token_len"].astype("int64")))
print(f"[load] {len(tok)} docs", flush=True)

out = {}
for d in sorted(glob.glob(f"{W}/data/enterprise_N*")):
    n = os.path.basename(d).replace("enterprise_N", "")
    mf = f"{d}/manifest.json"
    sr = f"{d}/scale_report.json"
    if not os.path.exists(mf):
        continue
    dsids = json.load(open(mf)).get("dsids", [])
    corpus_tok = sum(int(tok.get(str(x), 0)) for x in dsids)
    nch = None
    if os.path.exists(sr):
        nch = json.load(open(sr)).get("N_chunks")
    out[n] = {
        "n_docs": len(dsids),
        "corpus_tokens": corpus_tok,
        "n_chunks": nch,
        "mean_doc_token": round(corpus_tok / max(len(dsids), 1), 1),
    }
    print(f"N{n}: docs={len(dsids)} corpus_tok={corpus_tok/1e6:.2f}M chunks={nch} mean={out[n]['mean_doc_token']}", flush=True)

json.dump(out, open(f"{W}/data/scale_corpus_tokens.json", "w"), indent=2)
print(f"[saved] data/scale_corpus_tokens.json ({len(out)} points)")
