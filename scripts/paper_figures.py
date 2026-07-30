#!/usr/bin/env python3
"""AAAI 论文用 4 张出版级矢量图(单栏宽 3.35in, 无图内标题, caption 由 LaTeX 提供)。
数据源: rag_bench_site/chart_data.js(已修正口径的唯一真相源)。
色板经 dataviz 六项检查(CVD/对比度)全过: naive 紫→品红, sandbox 青加深。
同时输出 200dpi PNG 预览供目检。"""
# ⚠ 注意: fig_pareto/fig_pertype 已换官方口径, 由 official_figs.py 生成并覆盖 —— 跑完本脚本必须再跑 official_figs.py!
import json, os, re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patheffects as pe

HALO = [pe.withStroke(linewidth=2.2, foreground="white")]

HERE = os.path.dirname(os.path.abspath(__file__))
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
W1 = 3.35  # AAAI 单栏英寸


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


# ---------- fig_acc: 准确率 vs 规模(7 家, CI 色带) ----------
# 官方判官口径(alignment-gated completeness); 数据来自 canonical_ci.csv，
# 横轴为 corpus tokens。缺少 canonical manifest/CI 时直接报错，避免回退到旧口径。
import csv as _csv
_U = os.environ.get("URAG_ROOT", os.getcwd())
_CT = {int(k): v["corpus_tokens"] for k, v in
       json.load(open(f"{_U}/data/scale_corpus_tokens.json")).items() if k.isdigit()}
ACC_OFF = {}
_canonical_ci = f"{_U}/results/judge_official_canonical_20260724/canonical_ci.csv"
_canonical_main_done = f"{_U}/results/judge_official_canonical_20260724/final_manifest.json"
if not (os.path.exists(_canonical_main_done) and os.path.exists(_canonical_ci)):
    raise FileNotFoundError(
        "Canonical accuracy artifacts are required: "
        "results/judge_official_canonical_20260724/"
        "{final_manifest.json,canonical_ci.csv}"
    )
for _r in _csv.DictReader(open(_canonical_ci)):
    _s = int(_r["scale"])
    if _s not in _CT:
        continue
    ACC_OFF.setdefault(_r["family"], []).append(
        (_CT[_s], float(_r["estimate"]),
         float(_r["ci95_low"]), float(_r["ci95_high"])))
# 2026-07-15 用户拍板: 最初版式(7家同权重, CI色带)。唯一改动: linearrag 最后画,
# 否则其首点(N1434)被 graphrag/lightrag 的实心标记盖住; 图例顺序保持原样。
fig, ax = newax(2.5)
draw_order = ["bm25", "sandbox", "hipporag", "naiverag", "graphrag", "lightrag", "linearrag"]
legend_order = ["bm25", "sandbox", "hipporag", "naiverag", "linearrag", "graphrag", "lightrag"]
handles = {}
GAP = 6.0   # 相邻实测点 x 比值超过此值 → 虚线示意未实测区间(仅 sandbox 42587→511959 的12×跳档触发; 常规5×稀疏采样网格保持实线)
for k in draw_order:
    pts = sorted(ACC_OFF.get(k, []))
    if not pts:
        continue
    c = COLOR[k]
    runs, cur = [], [pts[0]]
    for p in pts[1:]:
        if p[0] / cur[-1][0] > GAP:
            runs.append(cur); cur = [p]
        else:
            cur.append(p)
    runs.append(cur)
    for i, run in enumerate(runs):
        xs, ys, los, his = zip(*run)
        if len(run) > 1:
            ax.fill_between(xs, los, his, color=c, alpha=0.10, lw=0)
        ln, = ax.plot(xs, ys, color=c, lw=1.3, marker=MARK[k], ms=3.4,
                      markeredgewidth=0, label=NAME[k] if i == 0 else None)
        if i == 0:
            handles[k] = ln
        if i > 0:   # 断档连接段: 细虚线, 不带色带
            ax.plot([runs[i-1][-1][0], run[0][0]], [runs[i-1][-1][1], run[0][1]],
                    color=c, lw=0.9, ls=(0, (2.5, 2.5)), alpha=0.65)
ax.set_xscale("log")
ax.set_xlabel("Corpus size (tokens)")
ax.set_ylabel("Official combined score (%)")
ax.set_ylim(15, 85)
ax.grid(True, axis="y")
ax.legend([handles[k] for k in legend_order if k in handles],
          [NAME[k] for k in legend_order if k in handles],
          ncol=4, frameon=False, loc="lower left", bbox_to_anchor=(-0.02, 1.0),
          borderpad=0.1, columnspacing=0.8, handlelength=1.3, handletextpad=0.4,
          borderaxespad=0.0)
save(fig, "fig_acc")

# ---------- fig_build: 建图 token log-log + 幂律拟合(直接标注指数) ----------
fig, ax = newax(2.35)
for k in ("lightrag", "graphrag", "hipporag"):
    d = D["build"][k]
    c = COLOR[k]
    ax.plot(d["fitx"], d["fity"], color=c, lw=0.9, ls="--", alpha=0.75)
    ax.scatter(d["x"], d["y"], s=9, color=c, zorder=3, edgecolors="white",
               linewidths=0.3, label=f"{NAME[k]} ($b={d['b']:.2f}$)")
# LinearRAG 嵌入建图(零生成LLM), 来自旧的逐档重建账本。
import math as _math
_lin = []
for _r in _csv.DictReader(open(f"{_U}/results/embed_token_backfill.csv")):
    if _r["family"] == "linearrag" and int(_r["scale"]) in _CT:
        _lin.append((_CT[int(_r["scale"])], float(_r["embed_tokens"])))
_lin.sort()
_lx = [_math.log(a) for a, _ in _lin]; _ly = [_math.log(b) for _, b in _lin]
_n = len(_lin); _mx = sum(_lx)/_n; _my = sum(_ly)/_n
_b = sum((x-_mx)*(y-_my) for x, y in zip(_lx, _ly)) / sum((x-_mx)**2 for x in _lx)
_c = COLOR["linearrag"]
ax.plot([a for a, _ in _lin], [b for _, b in _lin], color=_c, lw=0.9, ls=":", alpha=0.9)
ax.scatter([a for a, _ in _lin], [b for _, b in _lin], s=9, marker="v",
           facecolors="none", edgecolors=_c, linewidths=0.6, zorder=3,
           label=f"LinearRAG emb. ($b={_b:.2f}$)")
# DenseRAG 嵌入建库 = Qwen3-Embedding-0.6B 对每个共享 chunk 精确单遍。
# 2026-07-24 完整回填覆盖全部 28 档，并逐档验证为 full-corpus chunk 前缀。
_dense_embed_csv = (
    f"{_U}/results/dense_embedding_token_backfill_20260724/"
    "dense_embedding_tokens.csv"
)
_nv = {
    int(_r["scale"]): float(_r["document_embedding_input_tokens"])
    for _r in _csv.DictReader(open(_dense_embed_csv))
    if int(_r["scale"]) in _CT
}
_nv = sorted((_CT[s], v) for s, v in _nv.items())
_nx = [_math.log(a) for a, _ in _nv]; _ny = [_math.log(b) for _, b in _nv]
_m = len(_nv); _mnx = sum(_nx)/_m; _mny = sum(_ny)/_m
_bn = sum((x-_mnx)*(y-_mny) for x, y in zip(_nx, _ny)) / sum((x-_mnx)**2 for x in _nx)
_cn = COLOR["naiverag"]
ax.plot([a for a, _ in _nv], [b for _, b in _nv], color=_cn, lw=0.9, ls=":", alpha=0.9)
ax.scatter([a for a, _ in _nv], [b for _, b in _nv], s=10, marker="P",
           facecolors="none", edgecolors=_cn, linewidths=0.6, zorder=3,
           label=f"DenseRAG emb. ($b={_bn:.2f}$)")
ax.legend(loc="lower right", frameon=False, borderpad=0.2, handletextpad=0.3,
          labelspacing=0.35)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Corpus size (tokens)")
ax.set_ylabel("Index-construction tokens")
ax.grid(True, which="major")
save(fig, "fig_build")

# ---------- fig_pareto: N1144 成本-准确率(直接标注, 前沿虚线) ----------
fig, ax = newax(2.4)
pts = D["pareto"]["n1144"]
DODGE = {"lightrag": 0.82, "graphrag": 1.22}   # 两点成本仅差3.5%, 显示层水平微移防误差棒互遮(caption 注明)
OFF = {"bm25": (9, -1), "naiverag": (10, -1), "hipporag": (10, -1), "graphrag": (11, -1),
       "lightrag": (11, -1), "sandbox": (12, -1)}   # 一律放点右侧垂直居中, 不跨误差棒
for p in pts:
    k = p["fam"]; c = COLOR[k]
    px = p["cost"] * DODGE.get(k, 1.0)
    ax.errorbar(px, p["acc"], yerr=[[p["acc"] - p["lo"]], [p["hi"] - p["acc"]]],
                fmt=MARK[k], color=c, ms=6 if MARK[k] != "*" else 9,
                elinewidth=0.7, capsize=1.8, markeredgecolor="white", markeredgewidth=0.4)
    dx, dy = OFF.get(k, (0, 7))
    ha = "right" if k == "lightrag" else "left"          # lightrag 标注放左侧, 避让 graphrag 误差棒
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
ax.set_ylim(44, 97)
ax.set_xlabel("Total cost, build $+$ 500-question workload (tokens)")
ax.set_ylabel("Accuracy (%)")
ax.grid(True, axis="y")
save(fig, "fig_pareto")

# ---------- fig_pertype: 分题型热力图(单色渐变+白缝+自适应墨色) ----------
h = D["pertype"]["n1144"]
TYPES_SHORT = ["basic", "semantic", "intra-doc", "project", "complete.",
               "conflict.", "constr.", "high-level", "not-found", "misc."]
fams_disp = [n.replace("SandboxAgent", "FS Agent").replace("MS GraphRAG", "GraphRAG")
             .replace("HippoRAG", "HippoRAG 2")
             for n in h["fams"]]
z = np.array([[np.nan if v is None else v for v in row] for row in h["z"]], float)
cmap = LinearSegmentedColormap.from_list("seqblue", ["#eff6ff", "#1e40af"])
fig, ax = plt.subplots(figsize=(W1, 2.05))
mesh = ax.pcolormesh(z, cmap=cmap, vmin=20, vmax=100, edgecolors="white", linewidth=1.0)
for i in range(z.shape[0]):
    for j in range(z.shape[1]):
        v = z[i, j]
        if np.isnan(v):
            continue
        ink = "white" if v >= 72 else "#1f2937"
        ax.text(j + 0.5, i + 0.5, f"{v:.0f}", ha="center", va="center",
                fontsize=5.6, color=ink)
ax.set_xticks(np.arange(10) + 0.5, TYPES_SHORT, rotation=38, ha="right", fontsize=6)
ax.set_yticks(np.arange(len(fams_disp)) + 0.5, fams_disp, fontsize=6.3)
ax.invert_yaxis()
ax.tick_params(length=0)
for s in ax.spines.values():
    s.set_visible(False)
cb = fig.colorbar(mesh, ax=ax, fraction=0.035, pad=0.02)
cb.ax.tick_params(labelsize=5.8, length=1.5, width=0.5)
cb.outline.set_linewidth(0.4)
cb.set_label("Accuracy (%)", fontsize=6.2)
save(fig, "fig_pertype")
print("[done]")

# ---------- fig_scaling: 跨双栏大图 (a)准确率 (b)建图成本 ----------
fig, (axA, axB) = plt.subplots(1, 2, figsize=(6.9, 2.55))
for ax in (axA, axB):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
# (a) 准确率
for k in ["bm25", "sandbox", "hipporag", "naiverag", "linearrag", "graphrag", "lightrag"]:
    d = D["acc"].get(k)
    if not d:
        continue
    c = COLOR[k]
    axA.fill_between(d["x"], d["lo"], d["hi"], color=c, alpha=0.10, lw=0)
    axA.plot(d["x"], d["y"], color=c, lw=1.3, marker=MARK[k], ms=3.4,
             markeredgewidth=0, label=NAME[k])
axA.set_xscale("log")
axA.set_xlabel("Corpus size (tokens)")
axA.set_ylabel("Accuracy (%)")
axA.set_ylim(30, 95)
axA.grid(True, axis="y")
axA.legend(ncol=4, frameon=False, loc="lower right", bbox_to_anchor=(1.02, 1.0),
           borderpad=0.1, columnspacing=0.7, handlelength=1.3, handletextpad=0.35,
           borderaxespad=0.0, fontsize=5.8)
axA.set_title("(a)", loc="left", fontsize=8)
# (b) 建图成本
for k in ("lightrag", "graphrag", "hipporag"):
    d = D["build"][k]
    c = COLOR[k]
    axB.plot(d["fitx"], d["fity"], color=c, lw=0.9, ls="--", alpha=0.75)
    axB.scatter(d["x"], d["y"], s=9, color=c, zorder=3, edgecolors="white",
                linewidths=0.3, label=f"{NAME[k]} ($b={d['b']:.2f}$)")
axB.set_xscale("log")
axB.set_yscale("log")
axB.set_xlabel("Corpus size (tokens)")
axB.set_ylabel("Index-construction LLM tokens")
axB.grid(True, which="major")
axB.legend(loc="lower right", frameon=False, borderpad=0.2, handletextpad=0.3,
           labelspacing=0.35)
axB.set_title("(b)", loc="left", fontsize=8)
fig.tight_layout(w_pad=2.0)
save(fig, "fig_scaling")

# ---------- fig_latency: 查询延时 vs 规模 (log-log, 2026-07-16 批注: latency需图表支撑) ----------
fig, ax = newax(2.35)
import glob as _glob, statistics as _st
# 三家单流延时读 results/latency/ 同后端探针中位数, 不再用 chart_data 旧混批数据 (2026-07-21)
_FAMFILE = {"bm25": "BM25", "naiverag": "NaiveRAG", "hipporag": "HippoRAG"}
for k in ("bm25", "naiverag", "hipporag"):
    _xs, _ys = [], []
    for _f in sorted(_glob.glob(f"{_U}/results/latency/{_FAMFILE[k]}_enterprise_N*.jsonl"),
                     key=lambda p: int(p.split("_N")[1].split(".")[0])):
        _n = int(_f.split("_N")[1].split(".")[0])
        if _n not in _CT:
            continue
        if k == "hipporag" and _n == 66932:
            continue  # 该档探针是6/16旧硬件遗留(索引已不存, 无法重测), 混入会压平131876段; 5档新探针已单调

        _l = [json.loads(x)["latency_sec"] for x in open(_f) if x.strip()]
        if not _l:
            continue
        _xs.append(_CT[_n]); _ys.append(_st.median(_l))
    if _xs:
        ax.plot(_xs, _ys, color=COLOR[k], lw=1.2, marker=MARK[k], ms=3.2,
                markeredgewidth=0, label=NAME[k])
# sandbox: 重尾分布, mean 抖动大 → median 线 + 线性插值 p10-p90 带。
# 新的固定配对重测只有在 COMPLETE.json 存在且三档均通过完整性校验时
# 才覆盖旧点；优先采用100题，保底采用分层50题。未完成实验绝不进论文图。
_sandbox_files = {
    int(_f.split("_N")[1].split(".")[0]): _f
    for _f in _glob.glob(f"{_U}/results/latency/sandbox_enterprise_N*.jsonl")
}
_rerun_scales = set()
for _rerun_name in ("latency_fixed100", "latency_fixed50"):
    _rerun_dir = f"{_U}/results/{_rerun_name}"
    _complete_path = f"{_rerun_dir}/COMPLETE.json"
    if not os.path.exists(_complete_path):
        continue
    _complete = json.load(open(_complete_path))
    _expected_ids = set(_complete.get("paired_question_ids", []))
    _expected_n = int(_complete.get("sample_n", len(_expected_ids)))
    _candidate = {}
    for _n in (42587, 66932, 511959):
        _f = f"{_rerun_dir}/sandbox_enterprise_N{_n}.jsonl"
        _rows = [json.loads(x) for x in open(_f) if x.strip()] if os.path.exists(_f) else []
        if (len(_rows) == _expected_n
                and len({r["id"] for r in _rows}) == _expected_n
                and {r["id"] for r in _rows} == _expected_ids
                and not any(r.get("error") for r in _rows)):
            _candidate[_n] = _f
    if len(_candidate) == 3:
        _sandbox_files.update(_candidate)
        _rerun_scales = set(_candidate)
        break
_sx, _smed, _slo, _shi = [], [], [], []
for _n, _f in sorted(_sandbox_files.items()):
    if _n not in _CT or (_n == 511959 and _n not in _rerun_scales):
        # 旧 N511959 探针受退化后端污染；只有健康后端固定100题重测完成后才恢复该点。
        continue
    _l = np.asarray([json.loads(x)["latency_sec"] for x in open(_f) if x.strip()], dtype=float)
    _p10, _p90 = np.quantile(_l, [0.10, 0.90], method="linear")
    _sx.append(_CT[_n]); _smed.append(_st.median(_l))
    _slo.append(float(_p10)); _shi.append(float(_p90))
_c = COLOR["sandbox"]
ax.fill_between(_sx, _slo, _shi, color=_c, alpha=0.12, lw=0)
ax.plot(_sx, _smed, color=_c, lw=1.2, marker=MARK["sandbox"], ms=4.5,
        markeredgewidth=0, label=NAME["sandbox"] + " (median)")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Corpus size (tokens)")
ax.set_ylabel("Query latency (s)")
ax.grid(True, which="major")
ax.legend(ncol=4, frameon=False, loc="lower left", bbox_to_anchor=(-0.02, 1.0),
          borderpad=0.1, columnspacing=0.8, handlelength=1.3, handletextpad=0.4,
          borderaxespad=0.0, fontsize=5.8)
save(fig, "fig_latency")

print("[all done]")
