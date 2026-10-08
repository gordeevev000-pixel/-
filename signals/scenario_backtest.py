"""Копия логики SCENARIOS.pine на Python — проверка на минутках 2018–2026.

Сценарный движок дня, как в индикаторе:
  1. Сессия рынка в часовом поясе биржи. Вчерашние уровни, сжатие (NR4/NR7) и дневной
     тренд считаются только по барам основной сессии — как в Pine, без request.security.
  2. Диапазон открытия: первые W минут сессии.
  3. Первое закрытие за диапазоном (не позже cutoff минут от открытия) задаёт направление
     сценария «трендовый день». Его поддерживают четыре условия:
       • сжатие накануне: вчерашний диапазон — самый узкий из 4 или из 7 (Крэбел; Zarattini и др.);
       • гэп в сторону пробоя (gap-and-go);
       • открытие за вчерашним максимумом/минимумом в сторону пробоя;
       • вчерашнее закрытие по ту же сторону от средней 20 закрытий.
     Сигнал — если выполнено не меньше min_score условий и издержки не больше cost_share риска.
  4. Стоп — противоположная граница диапазона открытия, выход — по закрытию последнего бара сессии.
Одна попытка в день: если первый пробой не прошёл фильтр, день пропускается.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import intraday_lab as L  # noqa: E402


@dataclass(frozen=True)
class Cfg:
    W: int = 30                # диапазон открытия, минут
    cutoff: int = 240          # пробой не позже, минут от открытия
    min_score: int = 3         # из 4 условий
    cost_share: float = 0.10   # издержки не больше этой доли риска
    cost: float | None = None  # издержки на круг в пунктах; None — по умолчанию рынка


def run(df: pd.DataFrame, mk: L.Market, cfg: Cfg) -> pd.DataFrame:
    o, h, l, c = (df[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    n = len(c)
    loc = df.index.tz_convert(mk.tz)
    tod = (loc.hour * 60 + loc.minute).to_numpy()
    dow = loc.dayofweek.to_numpy()
    dkey = (loc.year * 10000 + loc.month * 100 + loc.day).to_numpy()
    o_min = int(mk.open[:2]) * 60 + int(mk.open[3:])
    c_min = int(mk.close[:2]) * 60 + int(mk.close[3:])
    step = int(round(pd.Series(df.index).diff().dropna().mode().iloc[0].total_seconds() / 60))
    in_s = (tod >= o_min) & (tod < c_min) & (dow < 5)
    cost = mk.cost if cfg.cost is None else cfg.cost

    ranges: list[float] = []      # диапазоны завершённых сессий
    closes: list[float] = []      # закрытия завершённых сессий
    p_h = p_l = p_c = np.nan
    s_open = s_h = s_l = s_c = np.nan
    or_h = or_l = np.nan
    or_ready = done = False
    comp = False
    cur_key = -1
    pos, entry, stop, risk, meta = 0, 0.0, 0.0, 0.0, {}
    trades = []

    def close_session():
        nonlocal p_h, p_l, p_c
        ranges.append(s_h - s_l)
        closes.append(s_c)
        p_h, p_l, p_c = s_h, s_l, s_c

    for i in range(n):
        # сессия кончилась раньше обычного (короткий день, дыра в данных) — выход по открытию бара
        if pos != 0 and (not in_s[i] or dkey[i] != cur_key):
            E = entry * pos
            trades.append({**meta, "exit_i": i, "r": (o[i] * pos - E) / risk, "why": "close"})
            pos = 0
        if not in_s[i]:
            continue
        if dkey[i] != cur_key:                       # новая сессия
            if cur_key != -1:
                close_session()
            cur_key = dkey[i]
            s_open, s_h, s_l = o[i], h[i], l[i]
            or_h, or_l, or_ready, done = h[i], l[i], False, False
            r = ranges
            comp = len(r) >= 7 and (r[-1] <= min(r[-4:]) or r[-1] <= min(r[-7:]))
        else:
            s_h, s_l = max(s_h, h[i]), min(s_l, l[i])
        s_c = c[i]
        m = tod[i] - o_min
        if not or_ready:
            if m < cfg.W:
                or_h, or_l = max(or_h, h[i]), min(or_l, l[i])
            if m + step >= cfg.W:
                or_ready = True
                continue                              # бар, закрывший окно, сам сигналом не бывает
        last = tod[i] + step >= c_min                # последний бар сессии — по часам, как в Pine

        # сопровождение
        if pos != 0:
            hi, lo, op = (h[i], l[i], o[i]) if pos == 1 else (-l[i], -h[i], -o[i])
            S, E = stop * pos, entry * pos
            if lo <= S:
                trades.append({**meta, "exit_i": i, "r": (min(S, op) - E) / risk, "why": "stop"})
                pos = 0
            elif last:
                trades.append({**meta, "exit_i": i, "r": (c[i] * pos - E) / risk, "why": "close"})
                pos = 0
            continue

        # сигнал
        if done or not or_ready or m > cfg.cutoff or last:
            continue
        d = 1 if c[i] > or_h else -1 if c[i] < or_l else 0
        if d == 0:
            continue
        done = True
        if len(closes) < 20:
            continue
        sma20 = float(np.mean(closes[-20:]))
        gap_al = np.sign(s_open - p_c) == d
        open_al = (s_open > p_h) if d == 1 else (s_open < p_l)
        trend_al = np.sign(p_c - sma20) == d
        score = int(comp) + int(gap_al) + int(open_al) + int(trend_al)
        st = or_l if d == 1 else or_h
        rk = (c[i] - st) * d
        if score < cfg.min_score or rk <= 0 or cost > cfg.cost_share * rk:
            continue
        pos, entry, stop, risk = d, c[i], st, rk
        meta = {"i": i, "time": df.index[i], "dir": d, "score": score, "entry": c[i], "stop": st,
                "risk": rk, "cost_r": cost / rk, "comp": comp, "gap": gap_al, "open": open_al, "trend": trend_al}
    tr = pd.DataFrame(trades)
    if len(tr):
        tr["r_net"] = tr.r - tr.cost_r
        tr["oos"] = tr.time.dt.year >= 2023
    return tr


def stats(r: np.ndarray) -> dict:
    if len(r) == 0:
        return {"n": 0}
    bs = np.random.default_rng(1).choice(r, (4000, len(r))).mean(1)
    lo, hi = np.percentile(bs, [5, 95])
    return {"n": len(r), "win%": round((r > 0).mean() * 100, 1), "R": round(r.mean(), 3),
            "ДИ90": f"{lo:+.3f} … {hi:+.3f}", "PF": round(r[r > 0].sum() / max(-r[r < 0].sum(), 1e-9), 2)}
