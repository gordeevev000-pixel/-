"""SNAP — эталонная реализация движка на Python.

Бар в бар повторяет `pine/SNAP.pine`: те же формулы компонентов, тот же симулятор
сделок, тот же контроллер винрейта и тот же гейт. Нужна для того, чтобы числа в
README можно было пересчитать, а не принять на веру.

Смысл движка: на минутке выживает возврат к средней после выброса, а винрейт —
не свойство сигнала, а свойство геометрии выходов. Поэтому цель по винрейту здесь
задаётся явно, а R:R подстраивается под неё обратной связью по уже закрытым сделкам.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


# ── параметры (один в один с входами Pine) ────────────────────────────────────
@dataclass
class P:
    # ядро
    mean_len: int = 20
    atr_len: int = 14
    ctx_len: int = 8
    gain: float = 1.25
    thr_entry: float = 82.0
    # веса
    wX: float = 0.26
    wE: float = 0.22
    wB: float = 0.30
    wV: float = 0.22
    # X · растяжение
    z_full: float = 2.2
    a_full: float = 1.6
    # E · истощение
    vol_len: int = 20
    vol_full: float = 2.2
    fade_len: int = 8
    ew_vol: float = 0.40
    ew_fade: float = 0.30
    ew_wick: float = 0.30
    # B · прокол полосы и реклейм (триггер)
    k_mult: float = 1.9
    pierce_look: int = 3
    pierce_full: float = 0.60
    # V · магнит VWAP
    v_full: float = 1.2
    vwap_fallback_len: int = 60
    # Q · режим
    er_len: int = 20
    er_lo: float = 0.22
    er_hi: float = 0.62
    ema_slow: int = 200
    slope_len: int = 10
    vol_reg_len: int = 50
    vol_calm: float = 1.6
    vol_wild: float = 3.2
    q_min: float = 0.30
    min_atr_ticks: float = 6.0
    sess_from: int = 0        # минута суток, включительно
    sess_to: int = 1440       # минута суток, не включая
    # план сделки
    stop_len: int = 3
    stop_atr: float = 0.25
    risk_min_atr: float = 0.35
    risk_max_atr: float = 1.60
    max_bars: int = 10
    cooldown: int = 2
    # контроль винрейта
    target_wr: float = 0.66
    roll_len: int = 40
    min_sample: int = 25
    adapt: bool = True
    adapt_k_up: float = 0.70   # винрейт выше цели → цель по прибыли отодвигается (мягко)
    adapt_k_dn: float = 1.80   # винрейт ниже цели → подтягивается быстро
    rr_start: float = 0.55
    rr_min: float = 0.20
    rr_max: float = 2.00
    gate: bool = True
    rr_edge: float = 0.15     # цель всегда дальше издержек минимум на столько R
    gate_slack: float = 0.05  # насколько винрейт может просесть ниже цели, прежде чем сигналы гаснут     # цель всегда дальше издержек минимум на столько R
    # издержки
    mintick: float = 0.01
    cost_ticks: float = 2.0


# ── скользящие утилиты ────────────────────────────────────────────────────────
def _roll(x: np.ndarray, n: int, fn) -> np.ndarray:
    """fn по окну из n баров, включая текущий; первые n-1 баров — по неполному окну."""
    out = np.empty_like(x, dtype=float)
    if n <= 1:
        return x.astype(float).copy()
    w = sliding_window_view(x, n)
    out[n - 1:] = fn(w, axis=1)
    for i in range(min(n - 1, len(x))):
        out[i] = fn(x[: i + 1][None, :], axis=1)[0]
    return out


def _ema(x: np.ndarray, n: int) -> np.ndarray:
    a = 2.0 / (n + 1.0)
    out = np.empty_like(x, dtype=float)
    out[0] = x[0]
    for i in range(1, len(x)):
        out[i] = a * x[i] + (1.0 - a) * out[i - 1]
    return out


def _rma(x: np.ndarray, n: int) -> np.ndarray:
    a = 1.0 / n
    out = np.empty_like(x, dtype=float)
    out[0] = x[0]
    for i in range(1, len(x)):
        out[i] = a * x[i] + (1.0 - a) * out[i - 1]
    return out


def _clip01(x):
    return np.clip(x, 0.0, 1.0)


# ── ряды ──────────────────────────────────────────────────────────────────────
def series(b: dict, p: P) -> dict:
    o, h, l, c = b["open"], b["high"], b["low"], b["close"]
    v = b.get("volume", np.zeros_like(c))
    n = len(c)
    prev_c = np.concatenate([[c[0]], c[:-1]])

    tr = np.maximum(h - l, np.maximum(np.abs(h - prev_c), np.abs(l - prev_c)))
    atr = _rma(tr, p.atr_len)
    rng = np.maximum(h - l, p.mintick)

    basis = _ema(c, p.mean_len)
    sd = _roll(c, p.mean_len, np.std)
    vol_ma = _roll(v, p.vol_len, np.mean)
    has_vol = vol_ma > 0

    imp = np.abs(c - prev_c)
    imp_pk = _roll(imp, p.fade_len, np.max)

    up_band = basis + p.k_mult * atr
    dn_band = basis - p.k_mult * atr

    # VWAP сессии; без объёма — медленная средняя по hlc3
    hlc3 = (h + l + c) / 3.0
    vwap = np.empty(n)
    cum_pv = cum_v = 0.0
    day = b.get("day", np.zeros(n, dtype=int))
    prev_day = -1
    for i in range(n):
        if day[i] != prev_day:
            cum_pv = cum_v = 0.0
            prev_day = day[i]
        cum_pv += hlc3[i] * v[i]
        cum_v += v[i]
        vwap[i] = cum_pv / cum_v if cum_v > 0 else np.nan
    fallback = _roll(hlc3, p.vwap_fallback_len, np.mean)
    anchor = np.where(has_vol & np.isfinite(vwap), vwap, fallback)

    # режим
    shift = np.concatenate([np.full(p.er_len, c[0]), c[:-p.er_len]])
    path = _roll(np.abs(c - prev_c), p.er_len, np.sum)
    er = np.abs(c - shift) / np.maximum(path, 1e-9)
    ema_slow = _ema(c, p.ema_slow)
    ema_prev = np.concatenate([np.full(p.slope_len, ema_slow[0]), ema_slow[:-p.slope_len]])
    trend = np.where((c > ema_slow) & (ema_slow > ema_prev), 1,
             np.where((c < ema_slow) & (ema_slow < ema_prev), -1, 0))
    vol_rat = atr / np.maximum(_roll(atr, p.vol_reg_len, np.mean), 1e-9)

    sw_lo = _roll(l, p.stop_len, np.min)
    sw_hi = _roll(h, p.stop_len, np.max)
    pierce_lo = _roll(l, p.pierce_look, np.min)
    pierce_hi = _roll(h, p.pierce_look, np.max)

    minute = b.get("minute", np.zeros(n, dtype=int))
    in_sess = (minute >= p.sess_from) & (minute < p.sess_to)

    return dict(o=o, h=h, l=l, c=c, v=v, rng=rng, atr=atr, basis=basis, sd=sd,
                vol_ma=vol_ma, has_vol=has_vol, imp=imp, imp_pk=imp_pk,
                up_band=up_band, dn_band=dn_band, anchor=anchor, er=er,
                trend=trend, vol_rat=vol_rat, sw_lo=sw_lo, sw_hi=sw_hi,
                pierce_lo=pierce_lo, pierce_hi=pierce_hi, in_sess=in_sess)


# ── компоненты ────────────────────────────────────────────────────────────────
def components(s: dict, p: P) -> dict:
    """X/E/V считаются для обеих сторон векторно, B — тоже, но это строго текущий бар."""
    c, h, l = s["c"], s["h"], s["l"]
    atr, basis, sd, rng = s["atr"], s["basis"], s["sd"], s["rng"]
    out = {}
    for d, tag in ((1, "L"), (-1, "S")):
        dev = c - basis
        sz = np.maximum(0.0, -d * dev) / np.maximum(sd, 1e-12)
        sa = np.maximum(0.0, -d * dev) / np.maximum(atr, 1e-12)
        X = _clip01(np.minimum(sz / p.z_full, sa / p.a_full))

        e_vol = np.where(s["has_vol"],
                         _clip01((s["v"] / np.maximum(s["vol_ma"], 1e-9) - 1.0) / max(p.vol_full - 1.0, 0.1)),
                         0.5)
        e_fade = np.where(s["imp_pk"] > 0, _clip01(1.0 - s["imp"] / np.maximum(s["imp_pk"], 1e-12)), 0.0)
        e_wick = _clip01((c - l if d == 1 else h - c) / rng)
        E = _clip01(p.ew_vol * e_vol + p.ew_fade * e_fade + p.ew_wick * e_wick)

        band = s["dn_band"] if d == 1 else s["up_band"]
        depth = (band - s["pierce_lo"]) if d == 1 else (s["pierce_hi"] - band)
        depth = _clip01(np.maximum(0.0, depth) / np.maximum(p.pierce_full * atr, 1e-12))
        inside = (c > band) if d == 1 else (c < band)
        cpos = _clip01((c - l if d == 1 else h - c) / rng)
        B = np.where(inside & (depth > 0), _clip01(0.55 * cpos + 0.45 * depth), 0.0)

        V = _clip01(d * (s["anchor"] - c) / np.maximum(p.v_full * atr, 1e-12))

        against = (d == 1) & (s["trend"] < 0) | (d == -1) & (s["trend"] > 0)
        q_trend = np.where(against,
                           np.clip(1.0 - (s["er"] - p.er_lo) / max(p.er_hi - p.er_lo, 1e-9), p.q_min, 1.0),
                           1.0)
        q_vol = np.clip(1.0 - (s["vol_rat"] - p.vol_calm) / max(p.vol_wild - p.vol_calm, 1e-9), p.q_min, 1.0)
        Q = np.clip(np.minimum(q_trend, q_vol), p.q_min, 1.0)
        Q = np.where(s["in_sess"] & (atr >= p.min_atr_ticks * p.mintick), Q, 0.0)

        out[tag] = dict(X=X, E=E, B=B, V=V, Q=Q)
    return out


def _ctx_max(x: np.ndarray, n: int) -> np.ndarray:
    return _roll(x, n, np.max)


def score_series(s: dict, comp: dict, p: P) -> dict:
    res = {}
    for tag in ("L", "S"):
        k = comp[tag]
        X = _ctx_max(k["X"], p.ctx_len)
        E = _ctx_max(k["E"], p.ctx_len)
        V = _ctx_max(k["V"], p.ctx_len)
        B = k["B"]
        raw = p.wX * X + p.wE * E + p.wB * B + p.wV * V
        res[tag] = dict(score=100.0 * p.gain * k["Q"] * raw, X=X, E=E, B=B, V=V, Q=k["Q"])
    return res


# ── симулятор + контроллер винрейта ───────────────────────────────────────────
@dataclass
class Trade:
    i_in: int
    side: int
    entry: float
    stop: float
    take: float
    risk: float
    score: float
    rr: float
    live: bool
    roll_wr: float
    i_out: int = -1
    bars: int = 0
    r: float = 0.0
    how: str = ""


@dataclass
class Result:
    trades: list = field(default_factory=list)
    stalled_bars: int = 0

    def stats(self, live_only: bool = False) -> dict:
        ts = [t for t in self.trades if t.live or not live_only]
        n = len(ts)
        if n == 0:
            return dict(n=0, wr=float("nan"), avg_r=float("nan"), pf=float("nan"), rr=float("nan"),
                        be=float("nan"), bars=float("nan"), total_r=0.0, dd=0.0, takes=0)
        wins = [t for t in ts if t.r > 0]
        pos = sum(t.r for t in wins)
        neg = -sum(t.r for t in ts if t.r <= 0)
        eq = np.cumsum([t.r for t in ts])
        return dict(n=n, wr=len(wins) / n, avg_r=float(np.mean([t.r for t in ts])),
                    pf=(pos / neg if neg > 0 else float("inf")),
                    rr=float(np.mean([t.rr for t in ts])),
                    be=float(np.mean([1.0 / (t.rr + 1.0) for t in ts])),
                    bars=float(np.mean([t.bars for t in ts])),
                    total_r=float(eq[-1]),
                    dd=float(np.min(eq - np.maximum.accumulate(eq))),
                    takes=sum(1 for t in ts if t.how == "take"))


def run(b: dict, p: P) -> Result:
    """Один проход по барам — ровно как Pine считает свой `var`-стейт.

    Сделки симулируются все подряд (теневая книга): она кормит контроллер и таблицу
    калибровки. Гейт решает только одно — показывать сигнал или молчать, — и молчит
    он не по «плохой полосе», а когда контроллер упёрся в пол по издержкам и всё
    равно не вытягивает цель по винрейту.
    """
    s = series(b, p)
    sc = score_series(s, components(s, p), p)
    h, l, c, atr = s["h"], s["l"], s["c"], s["atr"]
    n = len(c)

    res = Result()
    rr_mult = p.rr_start
    window: list[float] = []
    stalled = False
    just_out = False
    last_exit = -10 ** 9
    t: Trade | None = None
    warm = max(p.ema_slow, p.vol_reg_len, p.mean_len) + p.ctx_len

    for i in range(1, n):
        just_out = False
        if stalled:
            res.stalled_bars += 1

        # ── сопровождение открытой сделки ─────────────────────────────────────
        if t is not None and i > t.i_in:
            cost = p.cost_ticks * p.mintick / t.risk
            stop_hit = l[i] <= t.stop if t.side == 1 else h[i] >= t.stop
            take_hit = h[i] >= t.take if t.side == 1 else l[i] <= t.take
            r = how = None
            if stop_hit:                       # стоп и цель на одном баре — засчитываем стоп
                r, how = -1.0 - cost, "stop"
            elif take_hit:
                r, how = t.rr - cost, "take"
            elif i - t.i_in >= p.max_bars:
                r, how = (c[i] - t.entry) * t.side / t.risk - cost, "time"
            if how is not None:
                t.i_out, t.bars, t.r, t.how = i, i - t.i_in, r, how
                res.trades.append(t)
                window.append(1.0 if r > 0 else 0.0)
                if len(window) > p.roll_len:
                    window.pop(0)
                if p.adapt and len(window) >= p.min_sample:
                    err = float(np.mean(window)) - p.target_wr
                    k = p.adapt_k_up if err > 0 else p.adapt_k_dn
                    rr_mult = float(np.clip(rr_mult * (1.0 + k * err), p.rr_min, p.rr_max))
                last_exit, t, just_out = i, None, True

        if t is not None or just_out or i < warm or i - last_exit < p.cooldown:
            continue

        # ── новый сетап ───────────────────────────────────────────────────────
        side = 1 if sc["L"]["score"][i] >= sc["S"]["score"][i] else -1
        tag = "L" if side == 1 else "S"
        score = sc[tag]["score"][i]
        if score < p.thr_entry or not np.isfinite(atr[i]) or atr[i] <= 0:
            continue

        entry = c[i]
        raw_stop = (s["sw_lo"][i] - p.stop_atr * atr[i]) if side == 1 else (s["sw_hi"][i] + p.stop_atr * atr[i])
        risk = float(np.clip(abs(entry - raw_stop), p.risk_min_atr * atr[i], p.risk_max_atr * atr[i]))
        stop = entry - side * risk

        # пол по издержкам: цель ближе него закрывается в минус даже при касании
        cost_r = p.cost_ticks * p.mintick / risk
        rr_floor = cost_r + p.rr_edge
        rr_eff = float(np.clip(max(rr_mult, rr_floor), rr_floor, p.rr_max))
        take = entry + side * rr_eff * risk

        ready = len(window) >= p.min_sample
        roll_wr = float(np.mean(window)) if ready else float("nan")
        # упор: контроллер уже на полу, а винрейт всё равно ниже цели
        stalled = bool(p.gate and ready and roll_wr < p.target_wr - p.gate_slack
                       and rr_mult <= rr_floor * 1.001)
        live = bool((not p.gate) or (ready and not stalled))

        t = Trade(i_in=i, side=side, entry=entry, stop=stop, take=take, risk=risk,
                  score=float(score), rr=rr_eff, live=live, roll_wr=roll_wr)

    return res
