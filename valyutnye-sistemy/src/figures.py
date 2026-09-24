"""Рисунки для доклада «Парижская, Генуэзская, Бреттон-Вудская и Ямайская валютные системы».

Запуск: python3 figures.py <папка_для_png>
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
OUT.mkdir(parents=True, exist_ok=True)

# Цвета эпох (проверены на различимость, в т.ч. при дальтонизме)
PARIS, GENOA, BW, JAM = "#a67a06", "#8a4fc0", "#1a8a5a", "#2f62c0"
INK, INK2, MUTED, GRID = "#13241d", "#4b5a52", "#7d8a82", "#dfe3da"

plt.rcParams.update({
    "font.family": "Liberation Sans",
    "font.size": 10,
    "axes.edgecolor": MUTED,
    "axes.labelcolor": INK2,
    "xtick.color": INK2,
    "ytick.color": INK2,
    "text.color": INK,
})


def rounded_bar(ax, x0, x1, y, h, color, alpha=1.0):
    ax.add_patch(FancyBboxPatch((x0, y - h / 2), x1 - x0, h,
                                boxstyle="round,pad=0,rounding_size=0.9",
                                linewidth=0, facecolor=color, alpha=alpha,
                                mutation_aspect=0.12))


# ---------- Рисунок 1. Эволюция мировой валютной системы ----------
fig, ax = plt.subplots(figsize=(8.2, 3.5), dpi=220)
rows = [
    ("Парижская\n(золотомонетный стандарт)", 1867, 1914, PARIS, "1867–1914"),
    ("Генуэзская\n(золотодевизный стандарт)", 1922, 1939, GENOA, "1922–1939"),
    ("Бреттон-Вудская\n(золотодолларовый стандарт)", 1944, 1976, BW, "1944–1976"),
    ("Ямайская\n(стандарт СДР, многовалютный)", 1976, 2026, JAM, "с 1976 г. по н. в."),
]
ys = []
for i, (name, a, b, c, lab) in enumerate(rows):
    y = len(rows) - 1 - i
    ys.append(y)
    rounded_bar(ax, a, b, y, 0.46, c)
    if b - a >= 25:
        ax.text(a + 1.4, y, lab, va="center", ha="left", color="white", fontsize=9, fontweight="bold")
    else:
        ax.text(b + 1.4, y, lab, va="center", ha="left", color=INK, fontsize=9, fontweight="bold")
ax.set_yticks(ys)
ax.set_yticklabels([r[0] for r in rows], fontsize=8.6, color=INK, linespacing=1.15)
ax.tick_params(axis="y", length=0, pad=6)

# Ранний этап Парижской системы: фактический золотой стандарт Великобритании
rounded_bar(ax, 1821, 1867, 3, 0.46, PARIS, alpha=0.25)
ax.text(1844, 3, "Великобритания с 1821 г.", va="center", ha="center", fontsize=7.6, color=INK2)
# Фактический распад Бреттон-Вудса (1971–1976)
rounded_bar(ax, 1971, 1976, 1, 0.46, "white", alpha=0.5)

events = [(1914, "1914 · Первая\nмировая война", "right"), (1929, "1929–1933 ·\nВеликая депрессия", "left"),
          (1971, "1971 · «никсоновский\nшок»", "center"), (1999, "1999 ·\nевро", "right"),
          (2016, "2016 · юань\nв корзине СДР", "center")]
for x, t, ha in events:
    ax.axvline(x, color=GRID, lw=1, zorder=0)
    dx = {"right": -0.8, "left": 0.8, "center": 0}[ha]
    ax.text(x + dx, -0.62, t, ha=ha, va="top", fontsize=7.3, color=INK2, linespacing=1.1)

ax.set_xlim(1818, 2030)
ax.set_ylim(-1.45, 3.5)
ax.set_xticks([1840, 1870, 1900, 1930, 1960, 1990, 2020])
ax.tick_params(axis="x", length=0, pad=2, labelsize=8)
ax.xaxis.set_ticks_position("top")
for s_ in ("left", "right", "bottom"):
    ax.spines[s_].set_visible(False)
ax.spines["top"].set_color(GRID)
fig.tight_layout()
fig.savefig(OUT / "fig1_timeline.png", facecolor="white")
plt.close(fig)

# ---------- Рисунок 2. Механизм «золотых точек» ----------
import math

fig, ax = plt.subplots(figsize=(7.6, 3.3), dpi=220)
parity = 4.8665
up, lo = parity * 1.005, parity * 0.995
t = [i / 10 for i in range(0, 121)]
rate = [parity + 0.0205 * math.sin(x * 0.9) * math.cos(x * 0.23) + 0.004 * math.sin(x * 3.1) for x in t]
ax.axhspan(lo, up, color=PARIS, alpha=0.08, lw=0)
ax.axhline(parity, color=PARIS, lw=1.6)
ax.axhline(up, color=INK2, lw=1, ls=(0, (4, 3)))
ax.axhline(lo, color=INK2, lw=1, ls=(0, (4, 3)))
ax.plot(t, rate, color=INK, lw=1.8, solid_capstyle="round")
ax.text(12.2, parity, "монетный паритет\n1 £ = 4,8665 $", va="center", fontsize=8.4, color=INK)
ax.text(12.2, up, "верхняя «золотая точка»:\nвыгоднее вывезти золото", va="center", fontsize=8.0, color=INK2)
ax.text(12.2, lo, "нижняя «золотая точка»:\nвыгоднее ввезти золото", va="center", fontsize=8.0, color=INK2)
ax.set_xlim(0, 17.6)
ax.set_ylim(4.826, 4.907)
ax.set_xticks([])
ax.set_ylabel("долл. США за 1 ф. ст.", fontsize=8.6)
ax.tick_params(axis="y", labelsize=8)
ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.2f}".replace(".", ",")))
ax.set_xlabel("время → (условная динамика рыночного курса)", fontsize=8.4, color=INK2)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.spines["bottom"].set_color(GRID)
ax.spines["left"].set_color(GRID)
fig.tight_layout()
fig.savefig(OUT / "fig2_gold_points.png", facecolor="white")
plt.close(fig)

# ---------- Рисунок 3. Золотой запас США, 1949–1971 ----------
years = ["1949", "1957", "1960", "1965", "1968", "1971"]
vals = [24.6, 22.9, 17.8, 13.8, 10.9, 10.2]
fig, ax = plt.subplots(figsize=(7.2, 3.2), dpi=220)
xs = range(len(years))
ax.bar(list(xs), vals, width=0.44, color=BW, linewidth=0)
for x, v in zip(xs, vals):
    ax.text(x, v + 0.6, f"{v:.1f}".replace(".", ","), ha="center", va="bottom", fontsize=9, color=INK, fontweight="bold")
ax.set_xticks(list(xs))
ax.set_xticklabels(years)
ax.set_xlim(-0.6, len(years) - 0.4)
ax.set_ylim(0, 28)
ax.set_yticks([0, 5, 10, 15, 20, 25])
ax.grid(axis="y", color=GRID, lw=0.8)
ax.set_axisbelow(True)
ax.set_ylabel("млрд долл. (по цене 35 $ за унцию)", fontsize=8.6)
ax.tick_params(length=0, labelsize=8.6)
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
ax.spines["bottom"].set_color(MUTED)
fig.tight_layout()
fig.savefig(OUT / "fig3_us_gold.png", facecolor="white")
plt.close(fig)

print("ok:", sorted(p.name for p in OUT.glob("*.png")))
