"""Проверка SCENARIOS.pine (через копию scenario_backtest.py) на минутках histdata 2018–2026.

Запуск:  python signals/fetch_histdata.py data/hd 2018
         python signals/intraday_evaluate.py data/hd

Деление: 2018–2022 — данные, на которых выбирались правила; 2023–2026 — новые данные,
на которых после этого ничего не менялось.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import intraday_lab as L  # noqa: E402
import scenario_backtest as SB  # noqa: E402


def run_grid(folder: Path, tfs=(1, 5, 15), windows=(15, 30), scores=(2, 3, 4), cost_mult=1.0) -> pd.DataFrame:
    rows = []
    for name, mk in L.MARKETS.items():
        base = L.load(folder, name)
        for tf in tfs:
            df = L.resample(base, tf)
            for W in windows:
                for ms in scores:
                    tr = SB.run(df, mk, SB.Cfg(W=W, min_score=ms, cost=mk.cost * cost_mult))
                    if len(tr):
                        tr["market"], tr["tf"], tr["W"], tr["ms"] = name, tf, W, ms
                        rows.append(tr)
    return pd.concat(rows, ignore_index=True)


def table(t: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    out = []
    for key, g in t.groupby(by):
        key = key if isinstance(key, tuple) else (key,)
        for oos in (False, True):
            s = SB.stats(g[g.oos == oos].r_net.to_numpy())
            out.append({**dict(zip(by, key)), "выборка": "2023–26" if oos else "2018–22", **s})
    return pd.DataFrame(out)


def repaint_check(df: pd.DataFrame, mk: L.Market, cuts: int = 60) -> tuple[int, int]:
    """История обрезается по свече сигнала и в случайных местах: сделки, открытые до обрезки,
    должны совпасть с полным прогоном (вход, стоп, направление)."""
    cols = ["i", "dir", "entry", "stop", "score"]
    full = SB.run(df, mk, SB.Cfg())
    rng = np.random.default_rng(7)
    ks = np.r_[rng.choice(full.i.to_numpy() + 1, min(cuts, len(full)), replace=False),
               rng.integers(len(df) // 3, len(df), 20)]
    bad = 0
    for k in ks:
        part = SB.run(df.iloc[:k], mk, SB.Cfg())
        a = full[full.exit_i < k][cols + ["exit_i", "r"]].reset_index(drop=True)
        b = part[cols + ["exit_i", "r"]].reset_index(drop=True) if len(part) else a.iloc[:0]
        bad += int(not a.equals(b))
    return bad, len(ks)


if __name__ == "__main__":
    folder = Path(sys.argv[1])
    pd.set_option("display.width", 220)
    t = run_grid(folder)
    print("=== Все рынки: окно диапазона × строгость × таймфрейм ===")
    print(table(t, ["W", "ms", "tf"]).to_string(index=False))
    d = t[(t.W == 30) & (t.ms == 3)]
    print("\n=== По умолчанию (30 минут, 3 из 4): по рынкам ===")
    print(table(d, ["market"]).to_string(index=False))
    print("\n=== По умолчанию: по рынкам и таймфреймам ===")
    print(table(d, ["market", "tf"]).to_string(index=False))
    print("\n=== По умолчанию: по годам (средний R после издержек) ===")
    d = d.assign(year=d.time.dt.year)
    print(d.pivot_table(index="market", columns="year", values="r_net", aggfunc="mean").round(2).to_string())
    for k in (0.5, 1.5):
        tk = run_grid(folder, windows=(30,), scores=(3,), cost_mult=k)
        print(f"\n=== Издержки ×{k} (30 минут, 3 из 4) ===")
        print(table(tk, ["tf"]).to_string(index=False))
    for name in ("SP500", "GOLD", "DAX"):
        mk = L.MARKETS[name]
        bad, n = repaint_check(L.resample(L.load(folder, name), 5), mk)
        print(f"\nперерисовка {name} 5м: расхождений {bad} из {n} обрезок")
