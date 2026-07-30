#!/usr/bin/env python3
"""fig_pareto 与 fig_pertype 的官方口径版 (2026-07-15, 用户拍板全文换官方 combined)。
数据: judge_official_resweep (bedrock 逐题判分, last-wins 去重 + 滤 None), 成本沿用 chart_data.js。
样式与 paper_figures.py 完全一致; 覆盖 figures/fig_pareto.pdf 与 fig_pertype.pdf。"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patheffects as pe

HALO = [pe.withStroke(linewidth=2.2, foreground="white")]
HERE = os.path.dirname(os.path.abspath(__file__))
U = os.environ.get("URAG_ROOT", os.getcwd())

RAW = open(os.path.join(
    os.environ.get("PROJECT_ROOT", os.getcwd()), "rag_bench_site", "chart_data.js"
)).read()
D = json.loads(RAW[RAW.index("=") + 1:].rstrip(";\n"))

COLOR = {"bm25": "#92400e", "naiverag": "#a21caf", "hipporag": "#2563eb",
         "graphrag": "#dc2626", "lightrag": "#ea580c", "linearrag": "#16a34a",
         "sandbox": "#0891b2"}
NAME = {"bm25": "BM25", "naiverag": "DenseRAG", "hipporag": "HippoRAG 2",
        "graphrag": "MS-GraphRAG", "lightrag": "LightRAG", "linearrag": "LinearRAG",
        "sandbox": "File-System Agent"}
MARK = {"bm25": "^", "naiverag": "P", "hipporag": "s", "graphrag": "o",
        "lightrag": "D", "linearrag": "v", "sandbox": "*"}

plt.rcParams.update({
    "font.size": 7.5, "axes.labelsize": 7.5, "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5, "legend.fontsize": 6.2, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.2, "ytick.major.size": 2.2,
    "grid.linewidth": 0.4, "grid.alpha": 0.30, "grid.color": "#9aa0a6",
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})
W1 = 3.35


def newax(h=2.35):
    fig, ax = plt.subplots(figsize=(W1, h))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return fig, ax


def save(fig, name):
    fig.savefig(f"{HERE}/figures/{name}.pdf")
    fig.savefig(f"{HERE}/figures/{name}_preview.png", dpi=200)
    plt.close(fig)
    print("saved", name)


# ---- 官方 bedrock 逐题 combined ----
QTYPE = {json.loads(l)["id"]: json.loads(l)["question_type"]
         for l in open(f"{U}/data/enterprise_N1144/questions.jsonl")}

JUDGE_FILE = {  # fam -> bedrock 判分文件名
    "bm25": "bm25_N1144", "naiverag": "naiverag_N1144", "hipporag": "hipporag_N1144",
    "graphrag": "graphrag_N1144", "lightrag": "lightrag_N1144",
    "sandbox": "sandbox_N1144", "linearrag": "linearrag_N1144",
}

def load_combined(fam, name=None):
    name = name or JUDGE_FILE[fam]
    canonical_done = os.path.exists(
        f"{U}/results/judge_official_canonical_20260724/final_manifest.json"
    )
    dirs = (
        ("judge_official_canonical_20260724", "judge_official_resweep",
         "judge_official")
        if canonical_done else
        ("judge_official_resweep", "judge_official")
    )
    for d in dirs:
        p = f"{U}/results/{d}/{name}.jsonl"
        if os.path.exists(p):
            seen = {}
            for l in open(p):
                r = json.loads(l)
                if r.get("aligned") is None:
                    continue
                seen[r["id"]] = r
            out = {}
            for qid, r in seen.items():
                if r.get("completeness_pct") is None:
                    continue
                out[qid] = r["completeness_pct"] if r["aligned"] == 1 else 0.0
            return out
    raise FileNotFoundError(name)

per_q = {fam: load_combined(fam) for fam in JUDGE_FILE}
for fam, d in per_q.items():
    print(f"  {fam}: n={len(d)} mean={np.mean(list(d.values())):.1f}")

# ---------- fig_pareto (官方 combined + 95% CI) ----------
fig, ax = newax(2.4)
COST = {p["fam"]: p["cost"] for p in D["pareto"]["n1144"]}
DODGE = {"lightrag": 0.82, "graphrag": 1.22}
OFF = {"bm25": (9, -1), "naiverag": (10, -1), "hipporag": (10, -1), "graphrag": (11, -1),
       "lightrag": (11, -1), "sandbox": (12, -1)}
pts = []
for fam, cost in COST.items():
    vals = np.array(list(per_q[fam].values()))
    m = vals.mean()
    ci = 1.96 * vals.std(ddof=1) / np.sqrt(len(vals))
    pts.append({"fam": fam, "cost": cost, "acc": m, "lo": m - ci, "hi": m + ci})
for p in pts:
    k = p["fam"]; c = COLOR[k]
    px = p["cost"] * DODGE.get(k, 1.0)
    ax.errorbar(px, p["acc"], yerr=[[p["acc"] - p["lo"]], [p["hi"] - p["acc"]]],
                fmt=MARK[k], color=c, ms=6 if MARK[k] != "*" else 9,
                elinewidth=0.7, capsize=1.8, markeredgecolor="white", markeredgewidth=0.4)
    dx, dy = OFF.get(k, (0, 7))
    ha = "right" if k == "lightrag" else "left"
    if k == "lightrag":
        dx = -11
    ax.annotate(NAME[k], (px, p["acc"]), textcoords="offset points",
                xytext=(dx, dy), ha=ha, va="center", fontsize=6.5, color=c,
                path_effects=HALO, zorder=5)
srt = sorted(pts, key=lambda p: p["cost"])
front, best = [], -1
for p in srt:
    if p["acc"] > best:
        front.append(p); best = p["acc"]
ax.plot([p["cost"] for p in front], [p["acc"] for p in front],
        ls="--", lw=0.8, color="#6b7280", zorder=0)
ax.set_xscale("log")
ax.set_xlim(1.3e6, 4e8)
ax.set_ylim(38, 88)
ax.set_xlabel("Total cost, build $+$ 500-question workload (tokens)")
ax.set_ylabel("Official combined score (%)")
ax.grid(True, axis="y")
save(fig, "fig_pareto")

# ---------- fig_pertype (官方 combined 分题型) ----------
# 2026-07-22 用户拍板: bedrock 太易(8格100分), 换 N42587 代表性中档
# (仍有5系统在场; MS-GraphRAG/LightRAG 该档已建不动, caption 说明)。
TYPE_ORDER = ["basic", "semantic", "intra_document_reasoning", "project_related",
              "completeness", "conflicting_info", "constrained", "high_level",
              "info_not_found", "miscellaneous"]
TYPES_SHORT = ["basic", "semantic", "intra-doc", "project", "complete.",
               "conflict.", "constr.", "high-level", "not-found", "misc."]
TYPE_LABELS = [f"{short} ({sum(v == kind for v in QTYPE.values())})"
               for short, kind in zip(TYPES_SHORT, TYPE_ORDER)]
FAM_ROWS = ["bm25", "naiverag", "hipporag", "linearrag", "sandbox"]
per_q42 = {fam: load_combined(fam, f"{fam if fam != 'sandbox' else 'sandbox'}_N42587")
           for fam in FAM_ROWS}
for fam, d in per_q42.items():
    print(f"  [pertype42] {fam}: n={len(d)} mean={np.mean(list(d.values())):.1f}")
fams_disp = [NAME[f] for f in FAM_ROWS]
z = np.full((len(FAM_ROWS), len(TYPE_ORDER)), np.nan)
for i, fam in enumerate(FAM_ROWS):
    by_type = {}
    for qid, v in per_q42[fam].items():
        t = QTYPE.get(qid)
        if t:
            by_type.setdefault(t, []).append(v)
    for j, t in enumerate(TYPE_ORDER):
        if by_type.get(t):
            z[i, j] = float(np.mean(by_type[t]))
cmap = LinearSegmentedColormap.from_list("seqblue", ["#eff6ff", "#1e40af"])
fig, ax = plt.subplots(figsize=(W1, 2.0))
mesh = ax.pcolormesh(z, cmap=cmap, vmin=20, vmax=100, edgecolors="white", linewidth=1.0)
for i in range(z.shape[0]):
    for j in range(z.shape[1]):
        v = z[i, j]
        if np.isnan(v):
            continue
        ink = "white" if v >= 72 else "#1f2937"
        ax.text(j + 0.5, i + 0.5, f"{v:.0f}", ha="center", va="center",
                fontsize=5.6, color=ink)
ax.set_xticks(np.arange(10) + 0.5, TYPE_LABELS, rotation=50, ha="right", fontsize=5.4)
ax.set_yticks(np.arange(len(fams_disp)) + 0.5, fams_disp, fontsize=6.3)
ax.invert_yaxis()
ax.tick_params(length=0)
for s in ax.spines.values():
    s.set_visible(False)
cb = fig.colorbar(mesh, ax=ax, fraction=0.035, pad=0.02)
cb.ax.tick_params(labelsize=5.8, length=1.5, width=0.5)
cb.outline.set_linewidth(0.4)
cb.set_label("Official combined score (%)", fontsize=6.2)
save(fig, "fig_pertype")

# ---- 正文引用数字一并打印 ----
for fam in ("bm25", "sandbox"):
    by_type = {}
    for qid, v in per_q[fam].items():
        by_type.setdefault(QTYPE.get(qid), []).append(v)
    print(fam, {t: round(float(np.mean(vs)), 1) for t, vs in sorted(by_type.items())})
print("[done]")
