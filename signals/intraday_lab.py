"""Лаборатория внутридневных сценариев: данные, сессии, признаки дня, симулятор сделок.

Все признаки считаются только по тому, что известно на момент решения:
вчерашняя сессия, ночь до открытия, уже закрытые бары текущей сессии.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numba as nb
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Market:
    tz: str
    open: str          # открытие основной сессии, местное время биржи
    close: str         # закрытие основной сессии
    cost: float        # издержки на круг (спред + комиссия + проскальзывание), в пунктах цены
    tick: float


# Основная (самая ликвидная) сессия и реалистичные издержки на круг для CFD/микрофьючерса.
MARKETS = {
    "SP500":  Market("America/New_York", "09:30", "16:00", 0.75, 0.25),
    "NASDAQ": Market("America/New_York", "09:30", "16:00", 2.5, 0.25),
    "DAX":    Market("Europe/Berlin",    "09:00", "17:30", 2.0, 0.5),
    "GOLD":   Market("America/New_York", "08:20", "13:30", 0.40, 0.1),
    "SILVER": Market("America/New_York", "08:25", "13:25", 0.030, 0.005),
    "WTI":    Market("America/New_York", "09:00", "14:30", 0.04, 0.01),
    "BRENT":  Market("America/New_York", "09:00", "14:30", 0.05, 0.01),
}


def load(folder: Path, name: str) -> pd.DataFrame:
    df = pd.read_parquet(folder / f"{name}_1m.parquet")
    return df[~df.index.duplicated()].sort_index()


def resample(df: pd.DataFrame, minutes: int) -> pd.DataFrame:
    if minutes == 1:
        return df
    agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    return df.resample(f"{minutes}min", label="left", closed="left").agg(agg).dropna(subset=["close"])


def sessions(df: pd.DataFrame, mk: Market) -> tuple[pd.DataFrame, np.ndarray]:
    """Разметка баров по сессиям. Возвращает таблицу сессий и номер сессии каждого бара (-1 — вне сессии).

    Таблица сессий: первый/последний бар основной сессии, её OHLC, ночной диапазон
    (от закрытия прошлой сессии до открытия этой), вчерашние уровни и признаки дня.
    """
    loc = df.index.tz_convert(mk.tz)
    tod = loc.hour * 60 + loc.minute
    o_min = int(mk.open[:2]) * 60 + int(mk.open[3:])
    c_min = int(mk.close[:2]) * 60 + int(mk.close[3:])
    in_rth = (tod >= o_min) & (tod < c_min) & (loc.dayofweek < 5)
    key = np.where(in_rth, loc.year * 10000 + loc.month * 100 + loc.day, -1)
    sid = np.full(len(df), -1, dtype=np.int64)
    uniq = np.unique(key[in_rth])
    pos = np.flatnonzero(in_rth)
    sid[pos] = np.searchsorted(uniq, key[in_rth])

    h, l, c, o = (df[k].to_numpy() for k in ("high", "low", "close", "open"))
    n_s = len(uniq)
    starts = np.searchsorted(sid[pos], np.arange(n_s))          # начало каждой сессии в pos
    first_i = pos[starts]
    last_i = pos[np.r_[starts[1:], len(pos)] - 1]
    s_h = np.maximum.reduceat(h[pos], starts)
    s_l = np.minimum.reduceat(l[pos], starts)
    t = pd.DataFrame({
        "date": pd.to_datetime(uniq.astype(str), format="%Y%m%d"),
        "i0": first_i, "i1": last_i,
        "open": o[first_i], "high": s_h, "low": s_l, "close": c[last_i],
        "bars": last_i - first_i + 1,
    })
    # ночь: всё между концом прошлой сессии и началом этой
    on_h = np.full(n_s, np.nan)
    on_l = np.full(n_s, np.nan)
    for s in range(1, n_s):
        a, b = last_i[s - 1] + 1, first_i[s]
        if b > a:
            on_h[s], on_l[s] = h[a:b].max(), l[a:b].min()
    t["on_high"], t["on_low"] = on_h, on_l
    t["pdh"], t["pdl"], t["pdc"] = t.high.shift(), t.low.shift(), t.close.shift()
    rng = t.high - t.low
    t["prev_range"] = rng.shift()
    t["atr_d"] = rng.rolling(14).mean().shift()                       # средний диапазон 14 прошлых сессий
    t["nr7"] = (rng == rng.rolling(7).min()).shift().fillna(False).astype(bool)
    t["inside"] = ((t.high <= t.high.shift()) & (t.low >= t.low.shift())).shift().fillna(False).astype(bool)
    t["gap"] = (t.open - t.pdc) / t.atr_d
    t["on_range"] = (t.on_high - t.on_low) / t.atr_d
    return t, sid


@nb.njit(cache=True)
def run_trade(o, h, l, c, i, d, entry, stop, target, last, trail_k, atr):
    """Сопровождение одной сделки с бара i+1 до last включительно (last — бар принудительного выхода).

    d — 1 лонг / -1 шорт. target <= 0 — без цели. trail_k > 0 — трейлинг: лучшая цена − k·atr[j].
    Стоп и цель в одной свече — считаем стоп. Гэп за стоп — выход по открытию.
    Возвращает (R до издержек, индекс выхода, причина: 0 стоп, 1 цель, 2 время).
    """
    risk = (entry - stop) * d
    s = stop * d
    best = entry * d
    for j in range(i + 1, last + 1):
        hi = h[j] * d if d == 1 else -l[j]
        lo = l[j] * d if d == 1 else -h[j]
        op = o[j] * d
        if lo <= s:
            px = min(s, op)
            return (px - entry * d) / risk, j, 0
        if target > 0 and hi >= target * d:
            px = max(target * d, op)
            return (px - entry * d) / risk, j, 1
        if trail_k > 0:
            if hi > best:
                best = hi
            ns = best - trail_k * atr[j]
            if ns > s:
                s = ns
    return (c[last] * d - entry * d) / risk, last, 2


def bar_context(df: pd.DataFrame, t: pd.DataFrame, sid: np.ndarray, mk: Market) -> dict[str, np.ndarray]:
    """Поминутный контекст сессии без заглядывания вперёд.

    m       — минут от открытия сессии (по началу бара);
    hi/lo   — максимум/минимум сессии по закрытым барам, включая текущий;
    twap    — средняя типичная цена сессии (замена VWAP: объёма в данных нет);
    or{N}h/l — диапазон первых N минут, известен только после их окончания (до этого NaN).
    """
    loc = df.index.tz_convert(mk.tz)
    tod = loc.hour * 60 + loc.minute
    o_min = int(mk.open[:2]) * 60 + int(mk.open[3:])
    h, l, c = (df[k].to_numpy() for k in ("high", "low", "close"))
    n = len(df)
    m = np.where(sid >= 0, tod - o_min, -1).astype(np.int64)
    step = int(round(pd.Series(df.index).diff().dropna().mode().iloc[0].total_seconds() / 60))
    hi = np.full(n, np.nan)
    lo = np.full(n, np.nan)
    tw = np.full(n, np.nan)
    ctx = {"m": m, "step": np.int64(step)}
    ors = {w: (np.full(n, np.nan), np.full(n, np.nan)) for w in (5, 15, 30, 60)}
    tp = (h + l + c) / 3
    for s in range(len(t)):
        a, b = t.i0.iat[s], t.i1.iat[s] + 1
        hi[a:b] = np.maximum.accumulate(h[a:b])
        lo[a:b] = np.minimum.accumulate(l[a:b])
        tw[a:b] = np.cumsum(tp[a:b]) / np.arange(1, b - a + 1)
        mm = m[a:b]
        for w, (oh, ol) in ors.items():
            inside = mm < w                                   # бары, начавшиеся в первые w минут
            if not inside.any():
                continue
            k = np.flatnonzero(inside)[-1]                     # последний бар окна
            done = mm + step >= w                              # окно закрыто к концу бара k
            if not done[k]:
                continue
            oh[a + k:b] = h[a:a + k + 1].max()
            ol[a + k:b] = l[a:a + k + 1].min()
    ctx.update(hi=hi, lo=lo, twap=tw)
    for w, (oh, ol) in ors.items():
        ctx[f"or{w}h"], ctx[f"or{w}l"] = oh, ol
    return ctx
