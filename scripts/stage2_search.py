"""Этап 2: поиск аналогов, веер, живое сужение — на одной минуте и на выборке.

База аналогов = все минуты до начала последнего месяца (минус горизонт), запросы —
из последнего месяца. Пишет reports/stage2_search.md и reports/stage2_fan.png.
"""
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analog_fan.config import Config
from analog_fan import search as S

cfg = Config()
feat = pd.read_parquet("data/features.parquet")
blocks = {k: v.split(",") for k, v in json.loads(Path("data/blocks.json").read_text()).items()}
mins = S.minutes_of(feat.index)
cutoff = feat.index[-1].normalize().replace(day=1)          # начало последнего месяца
db_mask = feat["valid_y"].to_numpy() & (feat.index < cutoff - pd.Timedelta(minutes=cfg.horizon))

t0 = time.time()
idx = S.AnalogIndex(feat, blocks, cfg, db_mask)
t_build = time.time() - t0

# --- скорость и выход после прореживания на 2000 случайных минутах тестового месяца
test_rows = np.flatnonzero(feat["valid_y"].to_numpy() & (feat.index >= cutoff))
rng = np.random.default_rng(1)
sample = np.sort(rng.choice(test_rows, 2000, replace=False))
t0 = time.time()
rows, dist, found = idx.search(sample, mins[sample])
t_q = (time.time() - t0) / len(sample)
gaps = []
for r in rows[:200]:
    tm = np.sort(mins[r[r >= 0]])
    gaps.append(np.diff(tm).min())

# --- одна минута: веер, аналоги, сужение
q = int(sample[len(sample) // 2])
r1, d1, _ = idx.search(np.array([q]), mins[[q]])
r1, d1 = r1[0], d1[0]
Y, T = S.gather_paths(feat, r1, cfg)
w0 = S.kernel_weights(d1)
fan0 = S.make_fan(Y, T, w0)
real = feat.loc[feat.index[q], [f"yc{j:02d}" for j in range(1, cfg.horizon + 1)]].to_numpy(float)

L = ["# Этап 2. Поиск аналогов и веер\n",
     f"* База аналогов: {idx.index.ntotal:,} минут до {cutoff:%Y-%m-%d} (исход каждой известен до начала теста)",
     f"* Индекс FAISS IndexFlatL2 (точный), размерность {len(idx.cols)}; построение {t_build:.1f} с",
     f"* Запрос: {t_q * 1000:.1f} мс на минуту (поиск {cfg.k_candidates} кандидатов + прореживание до K={cfg.k})",
     f"* Найдено аналогов после прореживания: медиана {np.median(found):.0f}, минимум {found.min()} из {cfg.k}",
     f"* Минимальный разрыв между аналогами одного запроса: {min(gaps)} мин (требование >= {cfg.episode_gap})",
     f"* Эффективный размер выборки (ESS) по весам: медиана "
     f"{np.median([S.make_fan(*S.gather_paths(feat, rr, cfg), S.kernel_weights(dd)).ess for rr, dd in zip(rows[:100], dist[:100])]):.0f}",
     "", f"## Пример: {feat.index[q]:%Y-%m-%d %H:%M} UTC", "",
     f"* P(+1 ATR раньше −1 ATR) = **{fan0.p_up:.1%}**, P(−1 раньше) = {fan0.p_dn:.1%}",
     f"* Ожидаемый ход к +15 мин: {fan0.exp_move:+.2f} ATR; медиана {fan0.quant[-1, 2]:+.2f}; "
     f"80% полоса [{fan0.quant[-1, 0]:+.2f}, {fan0.quant[-1, 4]:+.2f}] ATR",
     f"* Фактически: {real[-1]:+.2f} ATR к +15 мин", "",
     "5 ближайших аналогов:", "", "| дата (UTC) | расстояние | ход к +15, ATR |", "|---|---:|---:|"]
for i in range(5):
    L.append(f"| {feat.index[r1[i]]:%Y-%m-%d %H:%M} | {d1[i]:.3f} | {Y[i, -1]:+.2f} |")
L += ["", "Живое сужение (якорь зафиксирован, проходят минуты):", "",
      "| прошло мин | живых (норм. RMSE<=0.5) | ESS | медиана к +15 | 80% полоса к +15 | P(+1 раньше −1)* |",
      "|---:|---:|---:|---:|---:|---:|"]
fig, axes = plt.subplots(1, 4, figsize=(15, 3.8), sharey=True)
past = -feat.loc[feat.index[q], [f"p{k:02d}" for k in range(1, 16)]].to_numpy(float)[::-1]
for ax, k in zip(axes, (0, 3, 6, 10)):
    w, err, alive = S.live_reweight(Y, w0, real[:k], cfg)
    f = S.make_fan(Y, T, w, n_alive=int(alive.sum()))
    L.append(f"| {k} | {f.n_alive} | {f.ess:.0f} | {f.quant[-1, 2]:+.2f} | "
             f"[{f.quant[-1, 0]:+.2f}, {f.quant[-1, 4]:+.2f}] | {f.p_up:.1%} |")
    x = np.arange(1, cfg.horizon + 1)
    ax.fill_between(x, f.quant[:, 0], f.quant[:, 4], color="#4f8cff", alpha=.25, lw=0, label="80%")
    ax.fill_between(x, f.quant[:, 1], f.quant[:, 3], color="#4f8cff", alpha=.45, lw=0, label="50%")
    ax.plot(x, f.quant[:, 2], color="#1d4ed8", lw=1.5, label="медиана")
    ax.plot(np.arange(-14, 1), np.r_[past[1:], 0], color="k", lw=1)
    ax.plot(np.r_[0, x[:k]], np.r_[0, real[:k]], color="#e11d48", lw=2, label="реальный путь")
    ax.axvline(k, color="grey", lw=.5, ls=":")
    ax.set_title(f"прошло {k} мин · живых {f.n_alive}")
    ax.set_xlabel("минуты от якоря")
axes[0].set_ylabel("ход, ATR"); axes[0].legend(fontsize=7, loc="upper left")
fig.tight_layout(); fig.savefig("reports/stage2_fan.png", dpi=110)
L += ["", "\\* P по аналогам с их полным исходом (для k>0 это не «с текущей цены», а от якоря).",
      "", "![](stage2_fan.png)"]
Path("reports/stage2_search.md").write_text("\n".join(L) + "\n")
print("\n".join(L))
