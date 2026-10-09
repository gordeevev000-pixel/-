"""Копия логики SCENARIOS.pine на Python — проверка на минутках 2018–2026.

Сценарный движок дня, как в индикаторе:
  1. Сессия рынка в часовом поясе биржи. Вчерашние максимум, минимум и закрытие — по барам
     основной сессии графика; сжатие (NR4/NR7) и средняя 20 закрытий — по 30-минутным барам,
     как request.security в Pine (у 30-минутного ряда длинная история даже на минутном графике).
  2. Диапазон открытия: первые W минут сессии.
  3. Первое закрытие за диапазоном (не позже cutoff минут от открытия) задаёт направление
     сценария «трендовый день». Его поддерживают четыре условия:
       • сжатие накануне: вчерашний диапазон — самый узкий из 4 или из 7 (Крэбел; Zarattini и др.);
       • гэп в сторону пробоя (gap-and-go);
       • открытие за вчерашним максимумом/минимумом в сторону пробоя;
       • вчерашнее закрытие по ту же сторону от средней 20 закрытий.
     Сигнал — если выполнено не меньше min_score условий и издержки не больше cost_share риска.
  4. Стоп — противоположная граница диапазона открытия, выход — по закрытию последнего бара сессии.
     Если сессия кончилась раньше (короткий день) — по открытию следующего бара.
Одна попытка в день: если первый пробой не прошёл фильтр, день пропускается.
Если вчерашняя сессия неполная (бары есть меньше чем в 90 % её 30-минутных слотов: короткий
день, дыра в данных) или была больше недели назад — сегодня сигнала нет: вчерашние уровни
ненадёжны. Неполные сессии не идут и в сжатие со средней.
Издержки по умолчанию — доля цены (см. intraday_lab.MARKETS), в пунктах — если задано cfg.cost.
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
    cost: float | None = None  # издержки на круг в пунктах; None — доля цены по умолчанию рынка
    cost_mult: float = 1.0     # множитель издержек по умолчанию (для проверки чувствительности)
    leak: bool = False         # ТОЛЬКО для контроля теста на перерисовку: условие тренда
                               # подсматривает, закроется ли день дальше точки входа


FULL = 0.9   # сессия «полная», если бары есть хотя бы в 90 % её 30-минутных слотов
MAX_GAP = pd.Timedelta(days=6)   # вчерашняя сессия старше недели (дыра в данных) — не считается


def minutes(hhmm: str) -> int:
    return int(hhmm[:2]) * 60 + int(hhmm[3:])


def n_slots(mk: L.Market) -> int:
    """Сколько 30-минутных слотов (:00/:30) пересекает основная сессия."""
    o_min, c_min = minutes(mk.open), minutes(mk.close)
    return (-(-c_min // 30) * 30 - o_min // 30 * 30) // 30


def htf_context(df: pd.DataFrame, mk: L.Market) -> tuple[np.ndarray, np.ndarray]:
    """Сжатие (NR4/NR7) и средняя 20 закрытий по 30-минутным барам — как htfContext() в Pine.

    Сессия — все 30-минутные бары, пересекающие её часы. В расчёт идут только завершённые
    и полные сессии (баров не меньше 90 % от числа 30-минутных слотов сессии).
    Бару графика достаётся значение предыдущего 30-минутного бара
    (request.security(..., f()[1], lookahead_on)).
    """
    d30 = L.resample(df, 30)
    loc = d30.index.tz_convert(mk.tz)
    t = (loc.hour * 60 + loc.minute).to_numpy()
    dw = loc.dayofweek.to_numpy()
    k = (loc.year * 10000 + loc.month * 100 + loc.day).to_numpy()
    o_min, c_min = minutes(mk.open), minutes(mk.close)
    slots = n_slots(mk)
    ins = (t < c_min) & (t + 30 > o_min) & (dw < 5)
    h3, l3, c3 = (d30[x].to_numpy(float) for x in ("high", "low", "close"))
    rr: list[float] = []
    cc: list[float] = []
    key, live, hh, ll, cl, nb = -1, False, np.nan, np.nan, np.nan, 0
    comp = np.zeros(len(d30), dtype=bool)
    sma = np.full(len(d30), np.nan)

    def finish():
        if nb >= FULL * slots:
            rr.append(hh - ll)
            cc.append(cl)

    for j in range(len(d30)):
        if live and (k[j] != key or not ins[j]):
            finish()
            live = False
        if ins[j]:
            if not live:
                key, hh, ll, live, nb = k[j], h3[j], l3[j], True, 0
            else:
                hh, ll = max(hh, h3[j]), min(ll, l3[j])
            cl = c3[j]
            nb += 1
            if t[j] + 30 >= c_min:
                finish()
                live = False
        del rr[:-30], cc[:-30]
        n = len(rr)
        comp[j] = n >= 7 and (rr[-1] <= min(rr[-4:]) or rr[-1] <= min(rr[-7:]))
        sma[j] = float(np.mean(cc[-20:])) if len(cc) >= 20 else np.nan
    # бар графика → 30-минутный бар, в котором он начался → значение предыдущего 30-минутного бара
    j = np.searchsorted(d30.index.asi8, df.index.floor("30min").asi8) - 1
    ok = j >= 0
    out_c = np.zeros(len(df), dtype=bool)
    out_s = np.full(len(df), np.nan)
    out_c[ok], out_s[ok] = comp[j[ok]], sma[j[ok]]
    return out_c, out_s


def coverage(df: pd.DataFrame, mk: L.Market) -> dict[int, float]:
    """Доля 30-минутных слотов основной сессии, в которых есть бары, по каждому дню (ГГГГММДД).
    Нужна только для чистки данных в проверке: дни с дырами в самих данных не считаются."""
    loc = df.index.tz_convert(mk.tz)
    tod = loc.hour * 60 + loc.minute
    o_min, c_min = minutes(mk.open), minutes(mk.close)
    ins = (tod >= o_min) & (tod < c_min) & (loc.dayofweek < 5)
    key = (loc.year * 10000 + loc.month * 100 + loc.day)[ins]
    slot = (tod // 30)[ins]
    cnt = pd.DataFrame({"k": key, "s": slot}).drop_duplicates().groupby("k").size()
    return (cnt / n_slots(mk)).to_dict()


def run(df: pd.DataFrame, mk: L.Market, cfg: Cfg, include_open: bool = False) -> pd.DataFrame:
    o, h, l, c = (df[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    n = len(c)
    loc = df.index.tz_convert(mk.tz)
    tod = (loc.hour * 60 + loc.minute).to_numpy()
    dow = loc.dayofweek.to_numpy()
    dkey = (loc.year * 10000 + loc.month * 100 + loc.day).to_numpy()
    o_min, c_min = minutes(mk.open), minutes(mk.close)
    step = int(round(pd.Series(df.index).diff().dropna().mode().iloc[0].total_seconds() / 60))
    in_s = (tod >= o_min) & (tod < c_min) & (dow < 5)
    slots = n_slots(mk)
    times = df.index
    h_comp, h_sma = htf_context(df, mk)
    if cfg.leak:   # закрытие сессии, которое станет известно только в её конце
        last_close = pd.Series(c[in_s], index=dkey[in_s]).groupby(level=0).last().to_dict()

    p_h = p_l = p_c = np.nan
    p_full = False
    s_open = s_h = s_l = s_c = np.nan
    s_slots, s_last_slot, s_start = 0, -1, None
    or_h = or_l = np.nan
    or_ready = done = False
    cur_key = -1
    pos, entry, stop, risk, meta = 0, 0.0, 0.0, 0.0, {}
    trades = []

    for i in range(n):
        # сессия кончилась раньше обычного (короткий день, дыра) — выход по открытию следующего бара
        if pos != 0 and (not in_s[i] or dkey[i] != cur_key):
            trades.append({**meta, "exit_i": i, "r": (o[i] * pos - entry * pos) / risk, "why": "early"})
            pos = 0
        if not in_s[i]:
            continue
        if dkey[i] != cur_key:                       # новая сессия
            if cur_key != -1:
                p_h, p_l, p_c = s_h, s_l, s_c
                p_full = s_slots >= FULL * slots and times[i] - s_start <= MAX_GAP
            cur_key = dkey[i]
            s_open, s_h, s_l = o[i], h[i], l[i]
            s_slots, s_last_slot, s_start = 0, -1, times[i]
            or_h, or_l, or_ready, done = h[i], l[i], False, False
        else:
            s_h, s_l = max(s_h, h[i]), min(s_l, l[i])
        s_c = c[i]
        if tod[i] // 30 != s_last_slot:
            s_slots, s_last_slot = s_slots + 1, tod[i] // 30
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
        sma20, comp = h_sma[i], bool(h_comp[i])
        if np.isnan(sma20) or not p_full:            # нет истории или вчерашняя сессия неполная
            continue
        gap_al = np.sign(s_open - p_c) == d
        open_al = (s_open > p_h) if d == 1 else (s_open < p_l)
        trend_al = (np.sign(last_close[cur_key] - c[i]) == d) if cfg.leak else (np.sign(p_c - sma20) == d)
        score = int(comp) + int(gap_al) + int(open_al) + int(trend_al)
        st = or_l if d == 1 else or_h
        rk = (c[i] - st) * d
        cost = cfg.cost if cfg.cost is not None else mk.cost_pct * cfg.cost_mult * c[i]
        if score < cfg.min_score or rk <= 0 or cost > cfg.cost_share * rk:
            continue
        pos, entry, stop, risk = d, c[i], st, rk
        meta = {"i": i, "time": df.index[i], "day": cur_key, "dir": d, "score": score, "entry": c[i],
                "stop": st, "risk": rk, "cost_r": cost / rk, "comp": comp, "gap": gap_al, "open": open_al,
                "trend": trend_al}
    if include_open and pos != 0:
        trades.append({**meta, "exit_i": -1, "r": np.nan, "why": "open"})
    tr = pd.DataFrame(trades)
    if len(tr):
        tr["r_net"] = tr.r - tr.cost_r
        tr["oos"] = tr.time.dt.year >= 2023
    return tr


def stats(r: np.ndarray, groups: np.ndarray | None = None) -> dict:
    """Средний R и 90 % интервал. С groups — бутстрэп кластерами (например, по торговым дням:
    рынки в один день сильно связаны), иначе по сделкам."""
    if len(r) == 0:
        return {"n": 0}
    rng = np.random.default_rng(1)
    if groups is None:
        bs = rng.choice(r, (4000, len(r))).mean(1)
    else:
        codes, inv = np.unique(groups, return_inverse=True)
        sums = np.bincount(inv, weights=r)
        cnts = np.bincount(inv)
        pick = rng.integers(0, len(codes), (4000, len(codes)))
        bs = sums[pick].sum(1) / cnts[pick].sum(1)
    lo, hi = np.percentile(bs, [5, 95])
    return {"n": len(r), "win%": round((r > 0).mean() * 100, 1), "R": round(r.mean(), 3),
            "ДИ90": f"{lo:+.3f} … {hi:+.3f}", "PF": round(r[r > 0].sum() / max(-r[r < 0].sum(), 1e-9), 2)}
