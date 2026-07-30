#!/usr/bin/env python3
"""两张概念图(代码绘制矢量, 扁平学术风): teaser(3.35x2.2in) + overview(7.0x2.6in)。
风格: 白底、圆角卡片、细线、论文同款色板、少量清晰短标签。输出 PDF + PNG 预览。"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Polygon, Rectangle, Ellipse
import os

HERE = os.path.dirname(os.path.abspath(__file__))
INK = "#374151"; MUT = "#9ca3af"
BM = "#92400e"; GR = "#dc2626"; AG = "#0891b2"; GOLD = "#d4a017"; ORG = "#ea580c"
BLU = "#2563eb"; GRN = "#16a34a"; MAG = "#a21caf"
plt.rcParams.update({"font.size": 6.5, "text.color": INK, "font.family": "DejaVu Sans"})


def canvas(w, h, xmax, ymax):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, xmax); ax.set_ylim(0, ymax)
    ax.axis("off")
    return fig, ax


def rbox(ax, x, y, w, h, fc, ec=None, lw=0.8, r=1.2, alpha=1.0, z=2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                fc=fc, ec=ec or fc, lw=lw, alpha=alpha, zorder=z))


def arrow(ax, x1, y1, x2, y2, color=MUT, lw=1.0, style="-|>", ms=6, z=1, con="arc3,rad=0"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=ms,
                                 color=color, lw=lw, zorder=z, connectionstyle=con,
                                 shrinkA=0, shrinkB=0))


def doc(ax, x, y, w=4.2, h=5.4, fc="#f8fafc", ec=INK, lw=0.6, z=3):
    ax.add_patch(Rectangle((x, y), w, h, fc=fc, ec=ec, lw=lw, zorder=z))
    for i in range(3):
        yy = y + h * (0.68 - 0.18 * i)
        ax.plot([x + w*0.18, x + w*0.82], [yy, yy], color=MUT, lw=0.5, zorder=z+1)


def stack(ax, x, y, n, scale=1.0, fc="#f8fafc", ec=INK):
    for i in range(n):
        doc(ax, x + i*0.9*scale, y + i*0.7*scale, 4.2*scale, 5.4*scale, fc=fc, ec=ec)


def coin(ax, x, y, r=1.05, z=4):
    ax.add_patch(Circle((x, y), r, fc="#fbbf24", ec="#b45309", lw=0.6, zorder=z))
    ax.text(x, y - 0.08, "\\$", ha="center", va="center", fontsize=4.2, color="#7c2d12", zorder=z+1)


def coins(ax, x, y, n):
    for i in range(n):
        coin(ax, x + (i % 3) * 1.6, y + (i // 3) * 1.7)


def flame(ax, x, y, s=1.0, z=4):
    ax.add_patch(Polygon([(x, y), (x - 1.1*s, y + 1.6*s), (x - 0.3*s, y + 1.3*s),
                          (x, y + 3.0*s), (x + 0.3*s, y + 1.3*s), (x + 1.1*s, y + 1.6*s)],
                         closed=True, fc=ORG, ec="none", zorder=z))


def graphball(ax, cx, cy, r=3.4, color=GR, z=4):
    import math
    pts = [(cx + r*math.cos(a), cy + r*math.sin(a)) for a in [0.4, 1.5, 2.6, 3.7, 4.8, 5.9]]
    pts.append((cx, cy))
    for i in range(len(pts)):
        for j in range(i+1, len(pts)):
            if (i + j) % 2 == 0 or j == len(pts)-1:
                ax.plot([pts[i][0], pts[j][0]], [pts[i][1], pts[j][1]],
                        color=color, lw=0.45, alpha=0.55, zorder=z)
    for p in pts:
        ax.add_patch(Circle(p, 0.62, fc=color, ec="white", lw=0.4, zorder=z+1))


def robot(ax, cx, cy, s=1.0, color=AG, z=4):
    rbox(ax, cx-2.0*s, cy-1.7*s, 4.0*s, 3.2*s, fc=color, r=0.7, z=z)
    for dx in (-0.9*s, 0.9*s):
        ax.add_patch(Circle((cx+dx, cy+0.15*s), 0.42*s, fc="white", ec="none", zorder=z+1))
        ax.add_patch(Circle((cx+dx, cy+0.15*s), 0.18*s, fc=INK, ec="none", zorder=z+2))
    ax.plot([cx, cx], [cy+1.5*s, cy+2.4*s], color=color, lw=0.9*s, zorder=z)
    ax.add_patch(Circle((cx, cy+2.6*s), 0.4*s, fc=color, ec="none", zorder=z))
    ax.plot([cx-1.1*s, cx+1.1*s], [cy-0.9*s, cy-0.9*s], color="white", lw=0.55*s, zorder=z+1)


def gauge(ax, cx, cy, r=2.3, z=4):
    th = np.linspace(0.15*np.pi, 0.85*np.pi, 40)
    ax.plot(cx + r*np.cos(th), cy + r*np.sin(th), color=INK, lw=0.9, zorder=z)
    ax.plot([cx, cx + 0.72*r*np.cos(0.32*np.pi)], [cy, cy + 0.72*r*np.sin(0.32*np.pi)],
            color=GR, lw=1.0, zorder=z+1)
    ax.add_patch(Circle((cx, cy), 0.28, fc=INK, ec="none", zorder=z+2))


# ================= 图1 teaser =================
fig, ax = canvas(3.35, 2.2, 100, 66)

# 左: 语料阶梯(垂直居中) + 扇形分流
stack(ax, 3, 28, 2, 0.65)
stack(ax, 8, 27, 3, 0.8)
stack(ax, 14, 26, 4, 0.95)
ax.text(11.5, 21.0, "corpus", ha="center", fontsize=5.6, color=INK, fontweight="bold")
ax.text(11.5, 17.6, "1K $\\rightarrow$ 512K docs", ha="center", fontsize=4.9, color=MUT)
LANE = {"bm25": 54, "graph": 35, "agent": 15}
for k, ycol in LANE.items():
    arrow(ax, 22.5, 33, 26, ycol, color="#d1d5db", lw=0.9, ms=5, con="arc3,rad=-0.15")

# lane 1: BM25
y = LANE["bm25"]
ax.text(27, y + 6.0, "Lexical (BM25)", fontsize=5.7, color=BM, fontweight="bold")
arrow(ax, 26, y, 44.5, y, color=BM, lw=1.0)
rbox(ax, 44.5, y - 3.5, 14.5, 7, fc="#fdf6ec", ec=BM, lw=0.7)
ax.text(51.75, y, "inverted\nindex", ha="center", va="center", fontsize=4.6, color=BM)
arrow(ax, 59, y, 69.5, y, color=BM, lw=1.0)
coin(ax, 62.5, y + 4.2)
ax.text(64.6, y + 4.0, "cheap", fontsize=4.3, color=MUT, va="center")

# lane 2: LLM graph
y = LANE["graph"]
ax.text(27, y + 6.6, "LLM-built graph", fontsize=5.7, color=GR, fontweight="bold")
arrow(ax, 26, y, 33.5, y, color=GR, lw=2.2)
rbox(ax, 34, y - 3.6, 8.5, 7.2, fc="#fdeaea", ec=GR, lw=0.7)
ax.text(38.2, y, "LLM", ha="center", va="center", fontsize=5.5, color=GR, fontweight="bold")
arrow(ax, 42.8, y, 47.8, y, color=GR, lw=2.2)
graphball(ax, 52.6, y, 3.3, color=GR)
arrow(ax, 57.5, y, 69.5, y, color=GR, lw=1.0)
coins(ax, 60, y + 4.4, 5)
flame(ax, 67.6, y + 3.7, 0.85)
ax.text(48, y - 7.4, "build cost explodes", ha="center", fontsize=4.4, color=GR)

# lane 3: agent
y = LANE["agent"]
ax.text(27, y + 6.4, "Agent (no index)", fontsize=5.7, color=AG, fontweight="bold")
robot(ax, 30.5, y, 0.95, color=AG)
th = np.linspace(-0.4*np.pi, 1.15*np.pi, 40)
ax.plot(39.5 + 3.0*np.cos(th), y + 3.0*np.sin(th), color=AG, lw=0.8, zorder=3)
ax.annotate("", xy=(42.2, y-1.6), xytext=(41.5, y-2.3),
            arrowprops=dict(arrowstyle="-|>", color=AG, lw=0.8, mutation_scale=5))
doc(ax, 37.7, y - 2.4, 3.3, 4.3)
arrow(ax, 45.5, y, 69.5, y, color=AG, lw=1.0)
coins(ax, 58.5, y + 4.4, 3)
ax.text(48, y - 7.4, "iterative search at query time", ha="center", fontsize=4.4, color=AG)

# 右: mini pareto
rbox(ax, 72, 8, 26, 52, fc="#fafafa", ec="#e5e7eb", lw=0.7, r=1.6, z=1)
ax.text(85, 53.6, "who wins, at\nwhat price?", ha="center", fontsize=5.2, color=INK, style="italic")
ox, oy, aw, ah = 77.5, 14, 17.5, 32
ax.plot([ox, ox], [oy, oy+ah], color=INK, lw=0.8, zorder=3)
ax.plot([ox, ox+aw], [oy, oy], color=INK, lw=0.8, zorder=3)
pts = [(0.14, 0.90, BM, "*"), (0.30, 0.62, BLU, "s"), (0.10, 0.42, MAG, "P"),
       (0.62, 0.72, AG, "*"), (0.68, 0.20, GR, "o"), (0.62, 0.10, ORG, "D")]
for fx, fy, c, m in pts:
    ax.scatter(ox + fx*aw, oy + fy*ah, s=16 if m == "*" else 9, marker=m,
               color=c, edgecolors="white", linewidths=0.3, zorder=4)
ax.add_patch(Circle((ox + 0.14*aw, oy + 0.90*ah), 2.0, fc="none", ec=GOLD, lw=1.0, zorder=5))
ax.text(ox + aw/2, oy - 3.8, "cost (tokens)", ha="center", fontsize=4.7)
ax.text(ox - 3.4, oy + ah/2, "accuracy", va="center", rotation=90, fontsize=4.7)

fig.savefig(f"{HERE}/figures/teaser.pdf")
fig.savefig(f"{HERE}/figures/teaser.svg")            # Figma 可直接导入编辑
fig.savefig(f"{HERE}/figures/teaser_preview.png", dpi=220)
plt.close(fig)
print("teaser saved")

# ================= 图2 overview =================
fig, ax = canvas(7.0, 2.6, 210, 66)
PY, PH = 10, 46


def panel(x, w, title, num):
    rbox(ax, x, PY, w, PH, fc="#fafafa", ec="#e5e7eb", lw=0.8, r=2.0, z=1)
    ax.text(x + 2.2, PY + PH + 3.2, f"{num}  {title}", fontsize=6.4, fontweight="bold", color=INK)


def parrow(x):
    arrow(ax, x, PY + PH/2, x + 6, PY + PH/2, color=MUT, lw=1.4, ms=9)


# P1 难度基岩
panel(4, 44, "Difficulty floor", "1")
for i, (lbl, c, xx) in enumerate([("722 gold", GOLD, 8), ("326 traps", GR, 22), ("99 lures", ORG, 36)]):
    stack(ax, xx, 38, 2, 0.62, fc="white", ec=c)
    ax.text(xx + 2.6, 34.2, lbl, ha="center", fontsize=4.9, color=c)
    arrow(ax, xx + 2.6, 33, 24 + (i-1)*3.5, 24.5, color=c, lw=0.8, ms=5)
rbox(ax, 10, 16, 32, 7.5, fc="#eef2f7", ec=INK, lw=0.8, r=1.2)
ax.text(26, 19.7, "floor = 1,144 docs", ha="center", va="center", fontsize=5.6, color=INK, fontweight="bold")
ax.text(26, 12.6, "every trap sits in the base tier", ha="center", fontsize=4.6, color=MUT)
parrow(49.5)

# P2 嵌套阶梯
panel(57, 44, "Nested scaling ladder", "2")
cx2, cy2 = 79, PY + PH/2 + 2
for i, rr in enumerate([4.5, 8.5, 12.5, 16.5]):
    rbox(ax, cx2 - rr, cy2 - rr*0.58, 2*rr, 2*rr*0.58, fc="none",
         ec=[INK, BLU, BLU, BLU][i], lw=[1.1, 0.7, 0.7, 0.7][i], r=1.4, z=3)
ax.text(cx2, cy2, "floor", ha="center", va="center", fontsize=4.8, color=INK)
ax.text(cx2, cy2 + 12.6, "$\\times$1.25 / tier", ha="center", fontsize=5.0, color=BLU)
ax.text(79, 17.4, "28 tiers:  $T_1\\subset T_2\\subset\\cdots\\subset T_{28}$",
        fontsize=4.9, color=INK, ha="center")
ax.text(79, 13.2, "source mix & noise rate held fixed", ha="center", fontsize=4.6, color=MUT)
parrow(102.5)

# P3 统一协议
panel(110, 44, "Unified protocol", "3")
chip_c = [BM, MAG, BLU, GR, ORG, GRN, AG]
for i, c in enumerate(chip_c):
    rbox(ax, 114 + i*5.4, 42, 4.2, 6, fc=c, r=0.8, alpha=0.9)
    arrow(ax, 116.1 + i*5.4, 41.5, 128 + (i-3)*1.2, 33.5, color="#d1d5db", lw=0.6, ms=4)
rbox(ax, 112.6, 26, 38.8, 7, fc="#eef2f7", ec=INK, lw=0.8, r=1.2)
ax.text(132, 29.4, "same reader · top-5 · temp 0", ha="center", va="center", fontsize=4.6, color=INK)
gauge(ax, 121, 17.5)
ax.text(121, 12.6, "token ledger", ha="center", fontsize=4.7, color=MUT)
rbox(ax, 135, 15, 6.4, 5.6, fc="white", ec=INK, lw=0.7, r=0.8)
rbox(ax, 142.6, 15, 6.4, 5.6, fc="white", ec=INK, lw=0.7, r=0.8)
ax.text(138.2, 17.8, "J1", ha="center", va="center", fontsize=4.8, color=INK)
ax.text(145.8, 17.8, "J2", ha="center", va="center", fontsize=4.8, color=INK)
ax.text(142, 12.6, "dual judges", ha="center", fontsize=4.7, color=MUT)
parrow(155.5)

# P4 agentic 访问层
panel(163, 43, "Agentic access layer", "4")
cyl_x, cyl_y = 172, 30
ax.add_patch(Rectangle((cyl_x-5, cyl_y-6), 10, 12, fc="#eef2f7", ec=INK, lw=0.8, zorder=3))
ax.add_patch(Ellipse((cyl_x, cyl_y+6), 10, 3.4, fc="#eef2f7", ec=INK, lw=0.8, zorder=4))
ax.add_patch(Ellipse((cyl_x, cyl_y-6), 10, 3.4, fc="#eef2f7", ec=INK, lw=0.8, zorder=2))
ax.text(cyl_x, cyl_y-0.4, "graph\nindex", ha="center", va="center", fontsize=4.9, color=INK, zorder=5)
for dy, lbl in [(11, "function calls"), (0, "CLI tools"), (-11, "$\\cdots$")]:
    arrow(ax, cyl_x+5.6, cyl_y + dy*0.55, 188, cyl_y + dy, color=MUT, lw=0.8, ms=5)
    robot(ax, 193, cyl_y + dy, 0.78, color=AG)
    ax.text(196.5, cyl_y + dy, lbl, fontsize=4.6, va="center", color=INK)
ax.text(184, 12.6, "same budget, same judges", ha="center", fontsize=4.6, color=MUT)

fig.savefig(f"{HERE}/figures/overview.pdf")
fig.savefig(f"{HERE}/figures/overview.svg")          # Figma 可直接导入编辑
fig.savefig(f"{HERE}/figures/overview_preview.png", dpi=220)
plt.close(fig)
print("overview saved")
