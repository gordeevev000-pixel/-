"""Прогон SNAP по синтетике: числа для README.

Синтетика ничего не доказывает про реальный рынок — она проверяет механику:
держит ли контроллер заданный винрейт, что он за это берёт в R:R, и где движок
честно сознаётся, что цель недостижима. Режим «случайное блуждание» здесь главный:
эджа там нет по построению, и всё, что остаётся, — геометрия выходов.

    python3 research/study.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from research import snap_engine as e  # noqa: E402
from research import synth  # noqa: E402

MODES = ("mixed", "revert", "trend", "rw")
NAMES = {"mixed": "смешанный", "revert": "возврат к средней", "trend": "тренд", "rw": "случайное блуждание"}
SEEDS = (1, 2, 3, 4, 5)
BARS = 80_000
MT = 0.001  # шаг цены синтетики: ATR ≈ 130 тиков, как у ликвидной минутки


def base(**kw) -> e.P:
    return e.P(mintick=MT, **kw)


def _agg(rows: list[dict]) -> dict:
    return {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}


def table_modes() -> None:
    print("\n=== 1. Режимы × сиды, параметры по умолчанию (цель 66%) ===")
    print(f"{'режим':20s} {'сделок':>7s} {'винрейт':>9s} {'мин':>7s} {'ср. R':>8s} {'ПФ':>6s} {'R:R':>6s} {'баров':>6s} {'просадка':>9s}")
    for m in MODES:
        rows = []
        for sd in SEEDS:
            r = e.run(synth.make(BARS, seed=sd, mode=m), base())
            rows.append(r.stats(True))
        a = _agg(rows)
        wmin = min(x["wr"] for x in rows)
        print(f"{NAMES[m]:20s} {a['n']:7.0f} {a['wr']*100:8.1f}% {wmin*100:6.1f}% {a['avg_r']:+8.3f} "
              f"{a['pf']:6.2f} {a['rr']:6.2f} {a['bars']:6.1f} {a['dd']:8.1f}R")


def table_stability() -> None:
    print("\n=== 2. Устойчивость винрейта по кускам (смешанный режим, куски по 250 сделок) ===")
    for sd in SEEDS[:3]:
        r = e.run(synth.make(BARS, seed=sd, mode="mixed"), base())
        flags = np.array([1.0 if t.r > 0 else 0.0 for t in r.trades if t.live])
        cut = len(flags) // 250 * 250
        ch = flags[:cut].reshape(-1, 250).mean(axis=1) if cut else np.array([])
        below = int((ch < 0.65).sum())
        print(f"  сид {sd}: кусков {len(ch):3d}  медиана {np.median(ch)*100:.1f}%  "
              f"мин {ch.min()*100:.1f}%  макс {ch.max()*100:.1f}%  ниже 65%: {below} ({below/max(len(ch),1)*100:.0f}%)")


def table_costs() -> None:
    print("\n=== 3. Стресс по издержкам (смешанный режим, сид 1) ===")
    print(f"{'тиков':>6s} {'сделок':>7s} {'винрейт':>9s} {'R:R':>6s} {'ср. R':>8s} {'ПФ':>6s} {'молчит баров':>13s}")
    for ct in (0, 2, 4, 8, 16, 32):
        r = e.run(synth.make(BARS, seed=1, mode="mixed"), base(cost_ticks=ct))
        st = r.stats(True)
        print(f"{ct:6d} {st['n']:7d} {st['wr']*100:8.1f}% {st['rr']:6.2f} {st['avg_r']:+8.3f} "
              f"{st['pf']:6.2f} {r.stalled_bars:13d}")


def table_threshold() -> None:
    print("\n=== 4. Порог входа: чем строже отбор, тем дороже стоит тот же винрейт ===")
    print(f"{'порог':>6s} {'сделок':>7s} {'винрейт':>9s} {'R:R':>6s} {'ср. R':>8s} {'ПФ':>6s}")
    for thr in (62, 70, 76, 82, 88, 94):
        rows = [e.run(synth.make(BARS, seed=sd, mode="mixed"), base(thr_entry=thr)).stats(True) for sd in SEEDS[:3]]
        a = _agg(rows)
        print(f"{thr:6d} {a['n']:7.0f} {a['wr']*100:8.1f}% {a['rr']:6.2f} {a['avg_r']:+8.3f} {a['pf']:6.2f}")


def table_gate() -> None:
    print("\n=== 5. Что даёт гейт по недавнему винрейту (проверка идеи, а не реклама) ===")
    print("Идея «показывать сигнал только после горячей полосы» проверена отдельно:")
    print("прироста винрейта она не даёт — полосы удач не продолжаются. Поэтому гейт")
    print("в движке сделан не по недавнему винрейту, а по упору контроллера в издержки.")
    for m in ("mixed", "rw"):
        rows_all, rows_hot = [], []
        for sd in SEEDS[:3]:
            b = synth.make(BARS, seed=sd, mode=m)
            r = e.run(b, base(adapt=False, rr_start=0.55, gate=False))
            ts = r.trades
            hot = [t for t in ts if np.isfinite(t.roll_wr) and t.roll_wr >= 0.66]
            rows_all.append(np.mean([t.r > 0 for t in ts]))
            rows_hot.append(np.mean([t.r > 0 for t in hot]) if hot else np.nan)
        print(f"  {NAMES[m]:20s} все сделки {np.mean(rows_all)*100:5.1f}%   "
              f"после горячей полосы {np.nanmean(rows_hot)*100:5.1f}%")


def main() -> None:
    print("SNAP · прогон по синтетике,", BARS, "баров ×", len(SEEDS), "сидов на режим")
    table_modes()
    table_stability()
    table_costs()
    table_threshold()
    table_gate()
    print("\nСинтетика проверяет механику, а не эдж. Настоящая проверка — таблица")
    print("калибровки индикатора на вашем инструменте и отчёт стратегии в TradingView.")


if __name__ == "__main__":
    main()
