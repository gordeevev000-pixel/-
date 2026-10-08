"""Копия логики KONFLUENS.pine на Python — для проверки на истории.

Повторяет индикатор бар за баром: те же формулы (в семантике Pine: RMA/EMA с
затравкой SMA, ta.stdev по генеральной совокупности), тот же порядок действий
на закрытой свече: свинги → выносы → сопровождение сделки → фильтр и сетапы →
стоп. Старший ТФ берётся с последнего закрытого бара
(аналог request.security(..., f()[1], lookahead_on)).

Сделка: вход по закрытию сигнальной свечи; если в одной свече задеты и стоп,
и цель — считается стоп; гэп за стоп — выход по открытию. Издержки задаются
долей цены на круг и пересчитываются в R.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

ATR_LEN, PV_L, PV_R, PB_WIN, SW_WIN, WARMUP = 14, 5, 3, 6, 2, 250
STOP_BUF, MIN_STOP = 0.2, 0.5


@dataclass(frozen=True)
class Cfg:
    allow_long: bool = True
    allow_short: bool = True
    use_trend: bool = True
    use_break: bool = True
    use_sweep: bool = True
    sweep_vol: float = 1.2     # объём на выносе, раз от среднего
    exit: str = "trail"        # "trail" — трейлинг-стоп, "fixed" — цели 1 и 2
    trail_k: float = 3.0       # трейлинг: максимум с входа минус k·ATR
    tp1: float = 1.0
    tp2: float = 2.0
    partial: bool = False
    max_stop: float = 3.0
    max_bars: int = 0          # 0 — без выхода по времени
    cost: float = 0.0          # издержки на круг, доля цены (0.001 = 0.1 %)


# ─── индикаторы в семантике Pine ────────────────────────────────────────────
def sma(x: np.ndarray, n: int) -> np.ndarray:
    return pd.Series(x).rolling(n, min_periods=n).mean().to_numpy()


def _rec(x: np.ndarray, n: int, alpha: float) -> np.ndarray:
    """Рекурсивная средняя Pine: первое значение — SMA(n), дальше экспонента."""
    out = np.full(len(x), np.nan)
    s = sma(x, n)
    prev = np.nan
    for i in range(len(x)):
        if np.isnan(prev):
            prev = s[i]
        elif not np.isnan(x[i]):
            prev = alpha * x[i] + (1 - alpha) * prev
        out[i] = prev
    return out


def ema(x, n):
    return _rec(np.asarray(x, float), n, 2.0 / (n + 1))


def rma(x, n):
    return _rec(np.asarray(x, float), n, 1.0 / n)


def true_range(h, l, c, handle_na=True):
    pc = np.r_[np.nan, c[:-1]]
    tr = np.nanmax(np.c_[h - l, np.abs(h - pc), np.abs(l - pc)], axis=1)
    tr[0] = (h[0] - l[0]) if handle_na else np.nan
    return tr


def rsi(c, n=14):
    ch = np.r_[np.nan, np.diff(c)]
    up, dn = rma(np.maximum(ch, 0), n), rma(np.maximum(-ch, 0), n)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = 100 - 100 / (1 + up / dn)
    r = np.where(dn == 0, 100.0, np.where(up == 0, 0.0, r))
    r[np.isnan(up) | np.isnan(dn)] = np.nan
    return r


def roll(x, n, fn):
    return getattr(pd.Series(x).rolling(n, min_periods=n), fn)().to_numpy()


def shift(x, k=1):
    out = np.full(len(x), np.nan)
    out[k:] = x[:-k]
    return out


def barssince(cond: np.ndarray) -> np.ndarray:
    out = np.full(len(cond), np.nan)
    last = -1
    for i, v in enumerate(cond):
        if v:
            last = i
        if last >= 0:
            out[i] = i - last
    return out


# ─── старший таймфрейм ───────────────────────────────────────────────────────
def auto_htf(sec: int) -> str:
    return "15m" if sec <= 60 else "1h" if sec <= 300 else "4h" if sec <= 3600 else "1d" if sec <= 14400 else "1w" if sec <= 86400 else "1M"


def bucket(idx: pd.DatetimeIndex, htf: str) -> pd.Index:
    if htf == "1w":
        d = idx.tz_convert("UTC").normalize()
        return d - pd.to_timedelta(d.dayofweek, unit="D")
    if htf == "1M":
        return idx.tz_convert("UTC").tz_localize(None).to_period("M").to_timestamp().tz_localize("UTC")
    return idx.floor({"15m": "15min", "1h": "1h", "4h": "4h", "1d": "1D"}[htf])


def htf_tide(df: pd.DataFrame, htf: str) -> np.ndarray:
    """Значение прилива последнего закрытого бара старшего ТФ для каждого бара графика."""
    b = bucket(df.index, htf)
    hc = df["close"].groupby(b).last()
    e = ema(hc.to_numpy(), 50)
    c = hc.to_numpy()
    e1 = shift(e)
    d = np.where((c > e) & (e > e1), 1, np.where((c < e) & (e < e1), -1, 0)).astype(float)
    d[np.isnan(e) | np.isnan(e1)] = np.nan
    d = shift(d)                                        # d[1] внутри security
    m = pd.Series(d, index=hc.index)
    return np.nan_to_num(m.reindex(b).to_numpy(), nan=0.0).astype(int)


# ─── основной прогон ─────────────────────────────────────────────────────────
def run(df: pd.DataFrame, cfg: Cfg, htf: str | None = None, include_open: bool = False) -> pd.DataFrame:
    o, h, l, c = (df[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    v = np.nan_to_num(df["volume"].to_numpy(float)) if "volume" in df else np.zeros(len(c))
    n = len(c)
    sec = int(pd.Series(df.index).diff().dropna().mode().iloc[0].total_seconds())
    htf_dir = htf_tide(df, htf or auto_htf(sec))

    atr = rma(true_range(h, l, c), ATR_LEN)
    e20, e50, e200 = ema(c, 20), ema(c, 50), ema(c, 200)
    rs = rsi(c, 14)

    vol_avg = sma(v, 20)
    has_vol = np.nan_to_num(vol_avg) > 0
    rel_vol = np.where(has_vol, v / np.where(has_vol, vol_avg, 1), 0.0)
    rv3 = roll(rel_vol, 3, "max")

    rng = h - l
    clv = np.where(rng > 0, (c - l) / np.where(rng > 0, rng, 1), 0.5)
    rsi_min, rsi_max = roll(rs, PB_WIN, "min"), roll(rs, PB_WIN, "max")
    pb_low, pb_high = roll(l, PB_WIN, "min"), roll(h, PB_WIN, "max")
    low3, high3 = roll(l, 3, "min"), roll(h, 3, "max")
    dc_hi, dc_lo = shift(roll(h, 20, "max")), shift(roll(l, 20, "min"))

    bb_dev = 2.0 * roll(c, 20, "std") * np.sqrt(19 / 20)      # ta.stdev — генеральная
    bb_mid = sma(c, 20)
    kc_rng = 1.5 * ema(true_range(h, l, c), 20)
    sqz = (bb_mid + bb_dev < e20 + kc_rng) & (bb_mid - bb_dev > e20 - kc_rng)
    sqz_n = roll(sqz.astype(float), 20, "sum")
    sqz_ago = np.nan_to_num(barssince(sqz), nan=1000)

    r_hi, l_hi = roll(h, PV_R, "max"), shift(roll(h, PV_L, "max"), PV_R + 1)
    r_lo, l_lo = roll(l, PV_R, "min"), shift(roll(l, PV_L, "min"), PV_R + 1)
    h3, l3 = shift(h, PV_R), shift(l, PV_R)
    is_ph = (h3 > r_hi) & (h3 >= l_hi)
    is_pl = (l3 < r_lo) & (l3 <= l_lo)

    nan = float("nan")
    last_ph = last_pl = nan
    ph_brk = pl_brk = False
    ms = 0
    ph_swp = pl_swp = False
    sw_h_bar = sw_l_bar = -10**9
    sw_h_lvl = sw_l_lvl = nan
    sw_h_use = sw_l_use = False

    t_dir, t_bar = 0, -1
    t_entry = t_stop = t_risk = t_tp1 = t_tp2 = t_ext = nan
    t_t1 = False
    t_meta: dict = {}
    trades = []
    tp2 = max(cfg.tp2, cfg.tp1)
    trail = cfg.exit == "trail"

    def gt(a, b):          # сравнение в духе Pine: с na всегда false
        return not (np.isnan(a) or np.isnan(b)) and a > b

    for i in range(n):
        # 1. свинги и структура
        if is_ph[i]:
            last_ph, ph_brk, ph_swp = h3[i], False, False
        if is_pl[i]:
            last_pl, pl_brk, pl_swp = l3[i], False, False
        if not ph_brk and gt(c[i], last_ph):
            ph_brk, ms = True, 1
        if not pl_brk and gt(last_pl, c[i]):
            pl_brk, ms = True, -1
        # 2. выносы
        if not ph_swp and gt(h[i], last_ph):
            ph_swp, sw_h_bar, sw_h_lvl, sw_h_use = True, i, last_ph, False
        if not pl_swp and gt(last_pl, l[i]):
            pl_swp, sw_l_bar, sw_l_lvl, sw_l_use = True, i, last_pl, False

        # 3. сопровождение
        if t_dir != 0 and i > t_bar:
            d = t_dir
            # всё в «координатах лонга»: для шорта цены умножены на -1
            hi, lo, op = (h[i], l[i], o[i]) if d == 1 else (-l[i], -h[i], -o[i])
            E, S = t_entry * d, t_stop * d
            bank = 0.5 * cfg.tp1 if (cfg.partial and t_t1) else 0.0
            share = 0.5 if (cfg.partial and t_t1) else 1.0
            ex, why = None, ""
            if lo <= S:
                px = min(S, op)                       # гэп за стоп — выход по открытию
                ex = bank + share * (px - E) / t_risk
                why = "stop"
            elif not trail and hi >= t_tp2 * d:
                ex = 0.5 * cfg.tp1 + 0.5 * tp2 if cfg.partial else tp2
                why = "tp2"
            else:
                if not t_t1 and hi >= t_tp1 * d:
                    t_t1 = True
                    if cfg.partial:
                        S = max(S, E)
                if trail:
                    t_ext = max(t_ext * d, hi) * d
                    S = max(S, t_ext * d - cfg.trail_k * atr[i])
                t_stop = S * d
                if cfg.max_bars > 0 and i - t_bar >= cfg.max_bars:
                    ex = bank + share * (c[i] * d - E) / t_risk
                    why = "time"
            if ex is not None:
                cost_r = cfg.cost * t_entry / t_risk
                trades.append({**t_meta, "exit_i": i, "exit_time": df.index[i], "why": why,
                               "r": ex, "r_net": ex - cost_r, "bars": i - t_bar})
                t_dir = 0

        # 4. сетапы — только по направлению старшего ТФ и по нужную сторону EMA 200
        gate_l = htf_dir[i] == 1 and c[i] > e200[i]
        gate_s = htf_dir[i] == -1 and c[i] < e200[i]
        trend_up = e20[i] > e50[i] > e200[i]
        trend_dn = e20[i] < e50[i] < e200[i]
        pb_l = (cfg.use_trend and gate_l and trend_up and ms == 1 and pb_low[i] <= e20[i] and pb_low[i] >= e50[i] - 0.5 * atr[i]
                and (np.isnan(last_pl) or pb_low[i] >= last_pl) and rsi_min[i] < 50 and c[i] > h[i - 1] and c[i] > o[i]
                and c[i] > e20[i] and c[i] - e20[i] <= atr[i])
        pb_s = (cfg.use_trend and gate_s and trend_dn and ms == -1 and pb_high[i] >= e20[i] and pb_high[i] <= e50[i] + 0.5 * atr[i]
                and (np.isnan(last_ph) or pb_high[i] <= last_ph) and rsi_max[i] > 50 and c[i] < l[i - 1] and c[i] < o[i]
                and c[i] < e20[i] and e20[i] - c[i] <= atr[i])
        vol_ok = (not has_vol[i]) or rel_vol[i] >= 1.2
        bo_l = (cfg.use_break and gate_l and sqz_n[i] >= 5 and sqz_ago[i] <= 3 and gt(c[i], dc_hi[i]) and c[i] > o[i]
                and clv[i] >= 0.6 and vol_ok)
        bo_s = (cfg.use_break and gate_s and sqz_n[i] >= 5 and sqz_ago[i] <= 3 and gt(dc_lo[i], c[i]) and c[i] < o[i]
                and clv[i] <= 0.4 and vol_ok)
        sw_vol = (not has_vol[i]) or rv3[i] >= cfg.sweep_vol
        sw_l_on = i - sw_l_bar <= SW_WIN and not sw_l_use
        sw_h_on = i - sw_h_bar <= SW_WIN and not sw_h_use
        sw_l = cfg.use_sweep and gate_l and sw_l_on and c[i] > sw_l_lvl and c[i] > o[i] and sw_l_lvl - low3[i] <= atr[i] and sw_vol
        sw_s = cfg.use_sweep and gate_s and sw_h_on and c[i] < sw_h_lvl and c[i] < o[i] and high3[i] - sw_h_lvl <= atr[i] and sw_vol
        if sw_l_on and c[i] > sw_l_lvl and c[i] > o[i]:
            sw_l_use = True
        if sw_h_on and c[i] < sw_h_lvl and c[i] < o[i]:
            sw_h_use = True

        # 5. сигнал
        sig, st, ext = 0, 0, nan
        if t_dir == 0 and i > WARMUP and atr[i] > 0:
            if cfg.allow_long and (pb_l or bo_l or sw_l):
                sig, st = 1, 1 if pb_l else 2 if bo_l else 3
                ext = pb_low[i] if pb_l else low3[i]
            elif cfg.allow_short and (pb_s or bo_s or sw_s):
                sig, st = -1, 1 if pb_s else 2 if bo_s else 3
                ext = pb_high[i] if pb_s else high3[i]

        # 6. стоп и цели
        if sig != 0:
            stop = ext - sig * STOP_BUF * atr[i]
            risk = (c[i] - stop) * sig
            if risk < MIN_STOP * atr[i]:
                risk = MIN_STOP * atr[i]
                stop = c[i] - sig * risk
            if risk <= cfg.max_stop * atr[i]:
                t_dir, t_bar, t_entry, t_stop, t_risk, t_ext = sig, i, c[i], stop, risk, c[i]
                t_tp1, t_tp2, t_t1 = c[i] + sig * cfg.tp1 * risk, c[i] + sig * tp2 * risk, False
                t_meta = {"i": i, "time": df.index[i], "dir": sig, "setup": st,
                          "entry": c[i], "stop": stop, "risk_pct": risk / c[i] * 100}

    if include_open and t_dir != 0:
        trades.append({**t_meta, "why": "open"})
    return pd.DataFrame(trades)


def stats(tr: pd.DataFrame, col: str = "r_net") -> dict:
    if tr.empty:
        return {"n": 0}
    r = tr[col].to_numpy()
    eq = np.cumsum(r)
    dd = float(np.max(np.maximum.accumulate(np.r_[0, eq]) - np.r_[0, eq]))
    gw, gl = r[r > 0].sum(), -r[r < 0].sum()
    return {"n": len(r), "win": float((r > 0).mean() * 100), "avg": float(r.mean()), "sum": float(r.sum()),
            "pf": float(gw / gl) if gl > 0 else float("inf"), "dd": dd}
