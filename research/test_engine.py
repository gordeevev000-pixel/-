"""Проверки механики SNAP. Запуск: python3 research/test_engine.py

Ни одна из них не про доходность — они про то, что движок не жульничает:
не заглядывает в будущее, а винрейт действительно управляется геометрией выходов.
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from research import snap_engine as e  # noqa: E402
from research import synth  # noqa: E402

MT = 0.001


def slice_bars(b: dict, n: int) -> dict:
    return {k: (v[:n] if hasattr(v, "__len__") else v) for k, v in b.items()}


def test_no_lookahead() -> None:
    """Сделки на первых N барах не зависят от того, что было дальше."""
    b = synth.make(40_000, seed=11, mode="mixed")
    p = e.P(mintick=MT)
    full = e.run(b, p)
    head = e.run(slice_bars(b, 20_000), p)
    ref = [t for t in full.trades if t.i_out < 19_000]
    got = [t for t in head.trades if t.i_out < 19_000]
    assert len(ref) == len(got) and len(ref) > 100, (len(ref), len(got))
    for a, c in zip(ref, got):
        assert (a.i_in, a.side, a.i_out) == (c.i_in, c.side, c.i_out)
        assert abs(a.entry - c.entry) < 1e-12 and abs(a.take - c.take) < 1e-12
        assert abs(a.r - c.r) < 1e-12
    print(f"  нет заглядывания в будущее: {len(ref)} сделок совпали бар в бар")


def test_winrate_is_geometry() -> None:
    """Чем дальше цель, тем реже она достаётся — винрейт монотонно падает по R:R."""
    b = synth.make(60_000, seed=12, mode="mixed")
    wrs = []
    for rr in (0.3, 0.5, 0.8, 1.2, 1.8):
        st = e.run(b, e.P(mintick=MT, adapt=False, gate=False, rr_start=rr)).stats()
        wrs.append(st["wr"])
    assert all(x > y for x, y in zip(wrs, wrs[1:])), wrs
    print("  винрейт по R:R 0.3→1.8: " + " → ".join(f"{w*100:.1f}%" for w in wrs))


def test_controller_holds_target() -> None:
    """Контроллер выводит факт на цель во всех режимах, включая случайное блуждание."""
    for mode in ("mixed", "revert", "trend", "rw"):
        wr = [e.run(synth.make(60_000, seed=sd, mode=mode), e.P(mintick=MT)).stats(True)["wr"]
              for sd in (21, 22, 23)]
        assert min(wr) >= 0.65, (mode, wr)
        print(f"  {mode:7s} винрейт {np.mean(wr)*100:.1f}% (мин {min(wr)*100:.1f}%)")


def test_controller_target_moves() -> None:
    """Подняли цель — факт поднялся, а R:R (и вместе с ним матожидание) просел."""
    b = synth.make(60_000, seed=13, mode="mixed")
    lo = e.run(b, e.P(mintick=MT, target_wr=0.60)).stats(True)
    hi = e.run(b, e.P(mintick=MT, target_wr=0.75)).stats(True)
    assert hi["wr"] > lo["wr"] + 0.05 and hi["rr"] < lo["rr"], (lo, hi)
    print(f"  цель 60% → {lo['wr']*100:.1f}% при R:R {lo['rr']:.2f};  "
          f"цель 75% → {hi['wr']*100:.1f}% при R:R {hi['rr']:.2f}")


def test_gate_mutes_when_hopeless() -> None:
    """Издержки выше любой достижимой цели — движок замолкает, а не рисует сигналы."""
    b = synth.make(60_000, seed=14, mode="rw")
    quiet = e.run(b, e.P(mintick=MT, cost_ticks=40.0))
    loud = e.run(b, e.P(mintick=MT, cost_ticks=40.0, gate=False))
    assert quiet.stalled_bars > 1000 and quiet.stats(True)["n"] < loud.stats(True)["n"]
    print(f"  при 40 тиках издержек молчит {quiet.stalled_bars} баров, "
          f"сигналов {quiet.stats(True)['n']} вместо {loud.stats(True)['n']}")


def test_cost_floor() -> None:
    """Цель никогда не ставится ближе издержек — иначе касание цели даёт минус."""
    b = synth.make(30_000, seed=15, mode="mixed")
    p = e.P(mintick=MT, cost_ticks=8.0, target_wr=0.90)  # цель, которая тянет R:R вниз
    r = e.run(b, p)
    for t in r.trades:
        assert t.rr >= p.cost_ticks * p.mintick / t.risk + p.rr_edge - 1e-9
    takes = [t.r for t in r.trades if t.how == "take"]
    assert min(takes) > 0, min(takes)
    print(f"  {len(r.trades)} сделок: ни одной цели ближе издержек, "
          f"минимальный плюс по цели {min(takes):+.3f}R")


def main() -> None:
    tests = [test_no_lookahead, test_winrate_is_geometry, test_controller_holds_target,
             test_controller_target_moves, test_gate_mutes_when_hopeless, test_cost_floor]
    for t in tests:
        print(t.__name__)
        t()
    print(f"\n{len(tests)} проверок пройдено")


if __name__ == "__main__":
    main()
