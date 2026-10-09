"""Проверка SCENARIOS.pine (через копию scenario_backtest.py) на минутках histdata 2018–2026.

Запуск:  python signals/fetch_histdata.py data/hd 2018
         python signals/intraday_evaluate.py data/hd

2018–2022 — данные, на которых выбирались правила. 2023–2026 — проверочные: на них только
считали результат, но по ним тоже принимались решения (какие семейства сценариев отбросить,
окно 30 минут по умолчанию), поэтому это не «нетронутый экзамен».
Дни с дырами в самих данных (бары есть меньше чем в 90 % 30-минутных слотов сессии) из статистики
исключены, кусок файла DAX с котировками другого индекса — тоже (intraday_lab.BAD_RANGES).
Интервалы — бутстрэп по торговым дням: рынки в один день сильно связаны.
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
        cov = SB.coverage(base, mk)
        for tf in tfs:
            df = L.resample(base, tf)
            for W in windows:
                for ms in scores:
                    tr = SB.run(df, mk, SB.Cfg(W=W, min_score=ms, cost_mult=cost_mult))
                    if len(tr):
                        tr["market"], tr["tf"], tr["W"], tr["ms"] = name, tf, W, ms
                        tr["bad"] = tr.day.map(lambda d: cov.get(d, 0.0) < SB.FULL)
                        rows.append(tr)
    return pd.concat(rows, ignore_index=True)


def table(t: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    t = t[~t.bad]
    out = []
    for key, g in t.groupby(by):
        key = key if isinstance(key, tuple) else (key,)
        for oos in (False, True):
            x = g[g.oos == oos]
            s = SB.stats(x.r_net.to_numpy(), x.day.to_numpy())
            out.append({**dict(zip(by, key)), "выборка": "2023–26" if oos else "2018–22", **s})
    return pd.DataFrame(out)


def repaint_check(df: pd.DataFrame, mk: L.Market, cfg: SB.Cfg, cuts: int = 60) -> tuple[int, int]:
    """История обрезается по свече сигнала и в случайных местах. Все решения о входе до точки
    обрезки (включая сделку, ещё открытую на обрезке) должны совпасть с полным прогоном,
    а закрытые к обрезке сделки — и по результату."""
    cols = ["i", "dir", "entry", "stop", "score"]
    full = SB.run(df, mk, cfg)
    rng = np.random.default_rng(7)
    ks = np.r_[rng.choice(full.i.to_numpy() + 1, min(cuts, len(full)), replace=False),
               rng.integers(len(df) // 3, len(df), 20)]
    bad = 0
    for k in ks:
        part = SB.run(df.iloc[:k], mk, cfg, include_open=True)
        a = full[full.i < k][cols].reset_index(drop=True)
        b = part[cols].reset_index(drop=True) if len(part) else a.iloc[:0]
        ca = full[full.exit_i < k][cols + ["r"]].reset_index(drop=True)
        cb = part[(part.exit_i >= 0) & (part.exit_i < k)][cols + ["r"]].reset_index(drop=True) if len(part) else ca.iloc[:0]
        bad += int(not (a.equals(b) and ca.equals(cb)))
    return bad, len(ks)


if __name__ == "__main__":
    folder = Path(sys.argv[1])
    pd.set_option("display.width", 220)
    t = run_grid(folder)
    d0 = t[(t.W == 30) & (t.ms == 3)]
    print(f"Исключено сделок в днях с дырами в данных: {int(d0.bad.sum())} из {len(d0)} (30 минут, 3 из 4)")
    print("\n=== Все рынки: окно диапазона × строгость × таймфрейм ===")
    print(table(t, ["W", "ms", "tf"]).to_string(index=False))
    d = t[(t.W == 30) & (t.ms == 3)]
    print("\n=== По умолчанию (30 минут, 3 из 4): по рынкам ===")
    print(table(d, ["market"]).to_string(index=False))
    print("\n=== По умолчанию: по рынкам и таймфреймам ===")
    print(table(d, ["market", "tf"]).to_string(index=False))
    print("\n=== По умолчанию: по годам (средний R после издержек, 5 минут) ===")
    d5 = d[(d.tf == 5) & ~d.bad].assign(year=lambda x: x.time.dt.year)
    print(d5.pivot_table(index="market", columns="year", values="r_net", aggfunc="mean").round(2).to_string())

    t0 = run_grid(folder, tfs=(5,), windows=(30,), scores=(0,))
    t0 = t0[~t0.bad]
    print("\n=== Средний R ДО издержек по числу выполненных условий (5 минут, 30 минут, без фильтра) ===")
    print(t0.pivot_table(index="score", columns="oos", values="r", aggfunc=["mean", "size"]).round(3).to_string())

    for k in (0.5, 1.5):
        tk = run_grid(folder, windows=(30,), scores=(3,), cost_mult=k)
        print(f"\n=== Издержки ×{k} (30 минут, 3 из 4) ===")
        print(table(tk, ["tf"]).to_string(index=False))

    print("\n=== Тест на перерисовку (5 минут) ===")
    for name in ("SP500", "GOLD", "DAX"):
        mk = L.MARKETS[name]
        df = L.resample(L.load(folder, name), 5)
        bad, n = repaint_check(df, mk, SB.Cfg())
        ctl, n2 = repaint_check(df, mk, SB.Cfg(leak=True))
        print(f"{name}: расхождений {bad} из {n}; контроль с подглядыванием в закрытие сессии: {ctl} из {n2}")
