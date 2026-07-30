#!/usr/bin/env python3
"""Teaser(图1)候选: 成本-精度轨迹图, 全真数据。
x = query tokens per answer (log), y = official combined score。
每个范式一条轨迹, 沿语料 1,144→511,959 docs 演化; 与实验图(单指标vs规模)不重复。
数据: canonical_matrix.csv (combined) + sandbox predictions total_tok + Table1 tok/q。"""
import csv, json, glob, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

HALO = [pe.withStroke(linewidth=2.2, foreground="white")]
U = os.environ.get("URAG_ROOT", os.getcwd())
HERE = os.path.dirname(os.path.abspath(__file__))

COLOR = {"bm25": "#92400e", "naiverag": "#a21caf", "graphrag": "#dc2626", "sandbox": "#0891b2"}
NAME = {"bm25": "BM25", "naiverag": "DenseRAG", "graphrag": "MS-GraphRAG", "sandbox": "File-System Agent"}
MARK = {"bm25": "^", "naiverag": "P", "graphrag": "o", "sandbox": "*"}

plt.rcParams.update({
    "font.size": 7.5, "axes.labelsize": 7.5, "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5, "legend.fontsize": 6.2, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.2, "ytick.major.size": 2.2,
    "grid.linewidth": 0.4, "grid.alpha": 0.30, "grid.color": "#9aa0a6",
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})

TIERS = [1144, 2254, 6980, 21614, 42587, 131876, 511959]

# combined 分数
acc = {}
canonical = (
    f"{U}/results/judge_official_canonical_20260724/"
    "canonical_matrix.csv"
)
for r in csv.DictReader(open(canonical)):
    acc[(r["family"], int(r["scale"]))] = float(r["combined"])

# sandbox 每tier实测 tok/q
sb_tok = {}
for d in glob.glob(f"{U}/results/sandbox/enterprise_N*[0-9]"):
    n = int(d.split("_N")[1])
    if n not in TIERS:
        continue
    toks = [json.loads(l).get("total_tok", 0) for l in open(d + "/predictions.jsonl")]
    sb_tok[n] = sum(toks) / len(toks)

# 各家轨迹 (x=tok/q, y=combined)
traj = {}
for fam in ["bm25", "naiverag", "graphrag", "sandbox"]:
    xs, ys, ns = [], [], []
    for n in TIERS:
        key = (fam, n)
        if key not in acc:
            continue
        if fam == "sandbox":
            x = sb_tok[n]
        elif fam == "bm25":
            x = 5.8e3
        elif fam == "naiverag":
            x = 4.9e3
        else:
            x = 10.4e3
        xs.append(x); ys.append(acc[key]); ns.append(n)
    traj[fam] = (np.array(xs, float), np.array(ys, float), ns)

fig, ax = plt.subplots(figsize=(3.35, 2.5))
ax.set_xscale("log")

for fam in ["naiverag", "graphrag", "bm25", "sandbox"]:
    x, y, ns = traj[fam]
    lw = 1.6 if fam == "bm25" else 1.1
    ax.plot(x, y, "-", color=COLOR[fam], lw=lw, alpha=0.9, zorder=3,
            marker=MARK[fam], ms=4.5 if fam == "sandbox" else 3.0,
            markerfacecolor=COLOR[fam], markeredgecolor="white", markeredgewidth=0.4)
    # 方向箭头: 中段
    for i in [len(x) // 2]:
        if fam in ("bm25", "naiverag") and len(x) > 1:
            # 纯竖直轨迹: 在中点画向下箭头
            ax.annotate("", xy=(x[i], y[i] - 1.5), xytext=(x[i], y[i] + 1.5),
                        arrowprops=dict(arrowstyle="-|>", color=COLOR[fam], lw=0.9), zorder=4)
        elif len(x) > i + 1:
            ax.annotate("", xy=(x[i + 1], y[i + 1]), xytext=(x[i], y[i]),
                        arrowprops=dict(arrowstyle="-|>", color=COLOR[fam], lw=0.9), zorder=4)

# 起终点标注
for fam in ["bm25", "naiverag", "graphrag", "sandbox"]:
    x, y, ns = traj[fam]
    ax.plot(x[0], y[0], "o", ms=2.6, color=COLOR[fam], zorder=5,
            markerfacecolor="white", markeredgecolor=COLOR[fam], markeredgewidth=0.8)

# graphrag 终点 ✗ (builder 死亡)
xg, yg, _ = traj["graphrag"]
ax.text(xg[-1] * 1.35, yg[-1] - 1.0, r"$\times$ build fails", fontsize=6.0,
        color=COLOR["graphrag"], ha="left", va="top", path_effects=HALO)

# 家族名标注
xb, yb, _ = traj["bm25"]
ax.text(xb[-1] * 1.28, yb[0] + 0.5, "BM25", fontsize=7.0, fontweight="bold",
        color=COLOR["bm25"], ha="left", va="center", path_effects=HALO)
xn, yn, _ = traj["naiverag"]
ax.text(xn[-1] * 0.82, yn[-1] - 1.6, "DenseRAG", fontsize=6.5,
        color=COLOR["naiverag"], ha="right", va="top", path_effects=HALO)
ax.text(xg[-1] * 1.35, yg[0] + 0.6, "MS-GraphRAG", fontsize=6.5,
        color=COLOR["graphrag"], ha="left", va="bottom", path_effects=HALO)
xs_, ys_, _ = traj["sandbox"]
ax.text(xs_[0], ys_[0] + 2.0, "File-System Agent", fontsize=6.5,
        color=COLOR["sandbox"], ha="center", va="bottom", path_effects=HALO)

# 关键事实标注
ax.annotate("ties BM25 at 1,144 docs,\nat 39$\\times$ the cost",
            xy=(xs_[0]*0.88, ys_[0]+0.5), xytext=(3.5e4, 90.5),
            fontsize=6.0, color="#374151", ha="center",
            arrowprops=dict(arrowstyle="-", color="#9ca3af", lw=0.6))
ax.annotate("collapses at\n511,959 docs", xy=(xs_[-1], ys_[-1]-1.0),
            xytext=(1.1e5, 20), fontsize=6.0, color="#374151", ha="center",
            arrowprops=dict(arrowstyle="-", color="#9ca3af", lw=0.6))
ax.text(xb[0] * 1.30, 71.5, "no LLM indexing,\nnear-flat cost",
        fontsize=6.0, color="#374151", ha="left", va="top")

# 理想角
ax.annotate("ideal", xy=(2.62e3, 97.0), fontsize=6.5, color="#6b7280",
            fontstyle="italic", ha="left", va="top")

ax.set_xlabel("Query tokens per answer (log scale)")
ax.set_ylabel("Official combined score (%)")
ax.set_xlim(2.4e3, 2.6e6)
ax.set_ylim(12, 100)
ax.grid(True, which="major", axis="both")
ax.set_axisbelow(True)
for s in ["top", "right"]:
    ax.spines[s].set_visible(False)

fig.savefig(f"{HERE}/figures/teaser_traj.pdf")
fig.savefig(f"{HERE}/figures/teaser_traj_preview.png", dpi=200)
print("saved teaser_traj: 每家轨迹点数", {NAME[f]: len(traj[f][0]) for f in traj})
