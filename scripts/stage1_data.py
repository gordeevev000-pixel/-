"""Этап 1: загрузка, проверка данных, построение отпечатков.

python scripts/stage1_data.py <csv> [src_tz] [market_tz]
Пишет reports/stage1_data.md, reports/stage1_*.png и data/features.parquet.
"""
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
from analog_fan import data, features

csv = sys.argv[1] if len(sys.argv) > 1 else "data/standin_BTCUSDT_1m.csv"
src_tz = sys.argv[2] if len(sys.argv) > 2 else "UTC"
cfg = Config(market_tz=sys.argv[3] if len(sys.argv) > 3 else "UTC")

t0 = time.time()
raw = data.load_csv(csv, src_tz=src_tz)
clean, rep = data.validate_and_clean(raw, max_fill=cfg.max_fill, market_tz=cfg.market_tz)
feat, blocks = features.build(clean, cfg)
Path("data").mkdir(exist_ok=True)
feat.to_parquet("data/features.parquet")
pd.Series({b: ",".join(v) for b, v in blocks.items()}).to_json("data/blocks.json")
clean[["open", "high", "low", "close", "volume", "seg"]].to_parquet("data/clean.parquet")

vx, vy = feat["valid_x"], feat["valid_y"]
atr_bps = (feat["atr"] / feat["close"] * 1e4)[vx]
t = feat.loc[vy, "touch"]
xcols = [c for b in blocks.values() for c in b]

L = [f"# Этап 1. Данные и отпечатки\n", f"Файл: `{csv}`, время в файле: {src_tz}, время рынка: {cfg.market_tz}\n",
     "## Проверка данных\n", rep.text(), "",
     "## Отпечатки\n",
     f"* Размерность отпечатка: **{len(xcols)}** ({', '.join(f'{b}: {len(v)}' for b, v in blocks.items())})",
     f"* Минут с валидным отпечатком: {vx.sum():,} ({vx.mean():.1%}); с валидным отпечатком и исходом на {cfg.horizon} мин: {vy.sum():,}",
     f"* ATR({cfg.atr_n}) 1m: медиана {atr_bps.median():.1f} б.п. цены "
     f"(10–90%: {atr_bps.quantile(.1):.1f}–{atr_bps.quantile(.9):.1f}); в цене: медиана {feat.loc[vx,'atr'].median():.4g}",
     f"* Ход за {cfg.horizon} мин в ATR: σ = {feat.loc[vy, f'yc{cfg.horizon:02d}'].std():.2f}, "
     f"|ход| медиана = {feat.loc[vy, f'yc{cfg.horizon:02d}'].abs().median():.2f}",
     f"* Барьер ±{cfg.barrier_atr} ATR за {cfg.horizon} мин (безусловно): +1 первым {np.mean(t==1):.1%}, "
     f"−1 первым {np.mean(t==0):.1%}, оба в одной минуте {np.mean(t==0.5):.1%}, ни один {t.isna().mean():.1%}",
     "", "Сводка признаков (валидные минуты):", "", "| признак | среднее | σ | 1% | 99% |", "|---|---:|---:|---:|---:|"]
d = feat.loc[vx, xcols]
for col in [c for c in xcols if not c.startswith("p") or c in ("p01", "p05", "p15", "p30", "p60")]:
    s = d[col]
    L.append(f"| {col} | {s.mean():.3f} | {s.std():.3f} | {s.quantile(.01):.3f} | {s.quantile(.99):.3f} |")
L.append(f"\nВремя работы: {time.time() - t0:.0f} с")

# --- графики: профиль по часам и ATR во времени
fig, ax = plt.subplots(1, 2, figsize=(12, 3.6))
prof = rep.stats["hour_profile"]
ax[0].bar(prof.index - 0.2, prof["absret"], width=0.4, label="|доходность|")
ax[0].bar(prof.index + 0.2, prof["vol"], width=0.4, label="объём")
ax[0].set_title("Активность по часам UTC (1 = среднее)"); ax[0].legend(); ax[0].set_xticks(range(0, 24, 2))
atr_bps.resample("1D").median().plot(ax=ax[1], lw=1)
ax[1].set_title(f"ATR({cfg.atr_n}) 1m, б.п. цены (медиана за день)")
fig.tight_layout(); fig.savefig("reports/stage1_profile.png", dpi=110)

Path("reports/stage1_data.md").write_text("\n".join(L) + "\n\n![](stage1_profile.png)\n")
print("\n".join(L))
