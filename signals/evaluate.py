"""Прогон индикатора по всем рынкам и таймфреймам.

Запуск:  python signals/evaluate.py <папка с данными>

Деление истории: до 2024-01-01 — «обучение» (на нём выбирались настройки
по умолчанию), с 2024-01-01 — «экзамен» (настройки уже зафиксированы).
Часовые данные Yahoo есть только за 2 последних года — они целиком экзамен.
"""
from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import backtest as bt  # noqa: E402

SPLIT = pd.Timestamp("2024-01-01", tz="UTC")
COST = {"crypto": 0.001, "other": 0.0003}      # издержки на круг: 0.10 % и 0.03 %


def load(folder: Path) -> dict[str, pd.DataFrame]:
    out = {}
    for f in sorted(folder.glob("*.parquet")):
        df = pd.read_parquet(f)
        df = df[~df.index.duplicated()].sort_index()
        out[f.stem] = df
    return out


def cost_of(name: str) -> float:
    return COST["crypto"] if name.endswith(("USDT_15m", "USDT_1h", "USDT_4h", "USDT_1d")) else COST["other"]


def run_all(data: dict[str, pd.DataFrame], cfg: bt.Cfg, only: str | None = None) -> pd.DataFrame:
    rows = []
    for name, df in data.items():
        if only and not name.endswith(only):
            continue
        tr = bt.run(df, replace(cfg, cost=cost_of(name)))
        if tr.empty:
            continue
        tr["market"] = name
        tr["tf"] = name.split("_")[-1]
        tr["oos"] = tr["time"] >= SPLIT
        rows.append(tr)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def table(tr: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    g = tr.groupby(by)
    out = pd.DataFrame({
        "сделок": g.size(),
        "в плюс %": g["r_net"].apply(lambda r: (r > 0).mean() * 100),
        "ср. R": g["r_net"].mean(),
        "итог R": g["r_net"].sum(),
        "PF": g["r_net"].apply(lambda r: r[r > 0].sum() / max(-r[r < 0].sum(), 1e-9)),
    })
    return out.round(2)


VARIANTS = {
    "трейлинг 3 ATR": bt.Cfg(),
    "цели 1R/2R, половина на 1R": bt.Cfg(exit="fixed", partial=True),
}


def resample_4h(df: pd.DataFrame) -> pd.DataFrame:
    agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    return df.resample("4h").agg(agg).dropna(subset=["close"])


def ci90(r: np.ndarray, b: int = 4000) -> tuple[float, float]:
    """90 % доверительный интервал среднего R (бутстрэп)."""
    bs = np.random.default_rng(1).choice(r, (b, len(r))).mean(1)
    lo, hi = np.percentile(bs, [5, 95])
    return float(lo), float(hi)


def summary(groups: dict[str, pd.Series]) -> pd.DataFrame:
    rows = []
    for name, r in groups.items():
        r = r.to_numpy()
        lo, hi = ci90(r)
        rows.append({"выборка": name, "сделок": len(r), "в плюс %": (r > 0).mean() * 100, "ср. R": r.mean(),
                     "90% ДИ": f"{lo:+.2f} … {hi:+.2f}", "PF": r[r > 0].sum() / -r[r < 0].sum()})
    return pd.DataFrame(rows).round(2)


def repaint_check(df: pd.DataFrame, n_sig: int = 60, n_rand: int = 20) -> tuple[int, int]:
    """Обрезаем историю: ровно по свече сигнала и в случайных местах.
    Сигналы до точки обрезки должны совпасть с полным прогоном."""
    cols = ["i", "dir", "setup", "entry", "stop"]
    full = bt.run(df, bt.Cfg())
    rng = np.random.default_rng(7)
    sig = rng.choice(full["i"].to_numpy() + 1, min(n_sig, len(full)), replace=False)
    cuts = np.r_[sig, rng.integers(len(df) // 3, len(df), n_rand)]
    bad = 0
    for k in cuts:
        part = bt.run(df.iloc[:k], bt.Cfg(), include_open=True)
        a = full[full.i < k][cols].reset_index(drop=True)
        bad += int(not a.equals(part[cols].reset_index(drop=True)))
    return bad, len(cuts)


if __name__ == "__main__":
    data = load(Path(sys.argv[1]))
    crypto = {k: v for k, v in data.items() if "USDT" in k}
    other = {k: v for k, v in data.items() if "USDT" not in k}
    other.update({k.replace("_1h", "_4h"): resample_4h(v) for k, v in other.items() if k.endswith("_1h")})
    pd.set_option("display.width", 200)

    for name, cfg in VARIANTS.items():
        cr, ot = run_all(crypto, cfg), run_all(other, cfg)
        print(f"\n=== {name} ===")
        print(summary({
            "крипта 15м, 2022–23": cr[(cr.tf == "15m") & ~cr.oos].r_net,
            "крипта 15м, 2024–26": cr[(cr.tf == "15m") & cr.oos].r_net,
            "крипта 1ч, 2019–23": cr[(cr.tf == "1h") & ~cr.oos].r_net,
            "крипта 1ч, 2024–26": cr[(cr.tf == "1h") & cr.oos].r_net,
            "крипта 4ч, 2019–23": cr[(cr.tf == "4h") & ~cr.oos].r_net,
            "крипта 4ч, 2024–26": cr[(cr.tf == "4h") & cr.oos].r_net,
            "крипта день, 2019–23": cr[(cr.tf == "1d") & ~cr.oos].r_net,
            "крипта день, 2024–26": cr[(cr.tf == "1d") & cr.oos].r_net,
            "золото, евро, индексы 1ч, 2024–26": ot[ot.tf == "1h"].r_net,
            "золото, евро, индексы 4ч, 2024–26": ot[ot.tf == "4h"].r_net,
            "золото, евро, индексы день, 2000–26": ot[ot.tf == "1d"].r_net,
        }).to_string(index=False))
        if cfg.exit == "trail":
            print("\nпо рынкам, 4ч и день:")
            both = pd.concat([cr[cr.tf.isin(["4h", "1d"])], ot[ot.tf.isin(["4h", "1d"])]])
            print(table(both, ["market"]).to_string())
            print("\nпо сетапам, 4ч и день, новые данные (крипта 2024–26 + остальные рынки):")
            new = pd.concat([cr[cr.tf.isin(["4h", "1d"]) & cr.oos], ot[ot.tf.isin(["4h", "1d"])]])
            print(table(new, ["setup"]).to_string())
            print(table(new, ["dir"]).to_string())

    for name in ("BTCUSDT_4h", "ETHUSDT_1h", "GOLD_1d"):
        bad, total = repaint_check(data[name])
        print(f"\nпроверка на перерисовку {name}: расхождений {bad} из {total} обрезок")
