"""Проверка EDGE на истории: Python-двойник логики из EDGE.pine.

Запуск:
    pip install numpy pandas
    python edge_backtest.py --symbols BTCUSDT ETHUSDT --days 60 --cost 0.04

Свечи берутся с публичного API Binance (data-api.binance.vision) и кэшируются в ./data.
Вход — по закрытию сигнального бара, стоп и тейк в одном баре считаются стопом,
издержки вычитаются из каждой сделки. Старшие ТФ — только закрытые бары.
"""
import argparse
import json
import math
import os
import time
import urllib.request
import numpy as np
import pandas as pd

DEFAULT = dict(
    atr_len=14, ema_f=20, ema_s=50,
    htf=(("5min", 0.25), ("15min", 0.35), ("60min", 0.40)), bias_thr=0.2,
    pb_look=5, pb_touch=0.2,
    piv_l=5, piv_r=2, lvl_age=90, sweep_wick=0.4,
    sq_len=15, sq_k=2.5, atr_long=100,
    rvol_len=50, rvol_sweep=1.2, rvol_sq=1.3,
    sl_buf=0.1, sl_min=0.5, sl_max=2.5, rr=1.5, max_bars=30,
    cost_pct=0.04, max_cost_r=0.25,
    memory=150, prior_k=10, prior_mu=-0.05, z=0.5, min_n=20, min_edge=0.05,
    cooldown=5, one_at_a_time=True,
)

SETUPS = ("Откат", "Снятие ликвидности", "Пробой сжатия")


def rma(x, n):
    return x.ewm(alpha=1 / n, adjust=False).mean()


def ema(x, n):
    return x.ewm(span=n, adjust=False).mean()


def atr(df, n):
    pc = df.close.shift(1)
    tr = pd.concat([df.high - df.low, (df.high - pc).abs(), (df.low - pc).abs()], axis=1).max(axis=1)
    return rma(tr, n)


def efficiency(close, n):
    change = (close - close.shift(n)).abs()
    path = close.diff().abs().rolling(n).sum()
    return (change / path).fillna(0)


def htf_bias(df1, rule):
    """Оценка тренда на старшем ТФ в [-1, 1]; берётся ПОСЛЕДНИЙ ЗАКРЫТЫЙ бар (без перерисовки)."""
    h = df1.resample(rule, label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}).dropna()
    e = ema(h.close, 21)
    a = atr(h, 14)
    er = efficiency(h.close, 10)
    slope = (e - e.shift(3)) / (3 * a)
    pos = (h.close - e) / a
    score = np.tanh(1.5 * slope + 0.5 * pos) * (0.5 + 0.5 * er)
    prev = score.shift(1)  # значение закрытого бара
    idx = df1.index.floor(pd.Timedelta(rule))
    return pd.Series(prev.reindex(idx).values, index=df1.index)


def features(df, p):
    f = pd.DataFrame(index=df.index)
    f["atr"] = atr(df, p["atr_len"])
    f["atr_long"] = atr(df, p["atr_long"])
    f["ef"] = ema(df.close, p["ema_f"])
    f["es"] = ema(df.close, p["ema_s"])
    rng = (df.high - df.low).replace(0, np.nan)
    f["clv"] = ((df.close - df.low) / rng).fillna(0.5)
    f["rvol"] = (df.volume / df.volume.rolling(p["rvol_len"]).mean()).fillna(1)
    wsum = sum(w for _, w in p["htf"])
    f["bias"] = sum(htf_bias(df, r) * w for r, w in p["htf"]) / wsum
    n = p["piv_l"] + p["piv_r"] + 1
    f["pl"] = np.where(df.low.shift(p["piv_r"]) == df.low.rolling(n).min(), df.low.shift(p["piv_r"]), np.nan)
    f["ph"] = np.where(df.high.shift(p["piv_r"]) == df.high.rolling(n).max(), df.high.shift(p["piv_r"]), np.nan)
    f["lo_pb"] = df.low.rolling(p["pb_look"]).min()
    f["hi_pb"] = df.high.rolling(p["pb_look"]).max()
    f["clo_min"] = df.close.rolling(p["pb_look"]).min()
    f["clo_max"] = df.close.rolling(p["pb_look"]).max()
    f["box_hi"] = df.high.rolling(p["sq_len"]).max().shift(1)
    f["box_lo"] = df.low.rolling(p["sq_len"]).min().shift(1)
    return f


def run(df, p=None):
    p = {**DEFAULT, **(p or {})}
    f = features(df, p)
    O, H, L, C = (df[c].to_numpy() for c in ("open", "high", "low", "close"))
    A, AL, EF, ES = (f[c].to_numpy() for c in ("atr", "atr_long", "ef", "es"))
    CLV, RV, BIAS = f.clv.to_numpy(), f.rvol.to_numpy(), f.bias.to_numpy()
    PL, PH = f.pl.to_numpy(), f.ph.to_numpy()
    LOPB, HIPB, CMIN, CMAX = (f[c].to_numpy() for c in ("lo_pb", "hi_pb", "clo_min", "clo_max"))
    BH, BL = f.box_hi.to_numpy(), f.box_lo.to_numpy()

    lam = 1 - 1 / p["memory"]
    NB = 9
    sn, ss, ss2 = np.zeros(NB), np.zeros(NB), np.zeros(NB)
    open_vt = []   # [entry, sl, tp, dir, bucket, bar, costR, shown]
    trades = []    # закрытые виртуальные сделки
    last_trig = {}
    lvl_lo = lvl_hi = math.nan
    lvl_lo_bar = lvl_hi_bar = -10**9
    shown_active = False
    warm = max(p["ema_s"], p["atr_long"], p["rvol_len"]) + 5

    def edge(b):
        n = sn[b]
        mean = (ss[b] + p["prior_k"] * p["prior_mu"]) / (n + p["prior_k"])
        var = max(ss2[b] / n - (ss[b] / n) ** 2, 0.25) if n > 0 else 1.0
        return mean - p["z"] * math.sqrt(var / (n + p["prior_k"])), mean

    for i in range(len(C)):
        # 1) разрешаем открытые виртуальные сделки
        still = []
        for vt in open_vt:
            e, sl, tp, d, b, bar, cr, shown = vt
            if i <= bar:
                still.append(vt); continue
            hit_sl = L[i] <= sl if d > 0 else H[i] >= sl
            hit_tp = H[i] >= tp if d > 0 else L[i] <= tp
            dist = abs(e - sl)
            if hit_sl:
                r = -1.0
            elif hit_tp:
                r = p["rr"]
            elif i - bar >= p["max_bars"]:
                r = d * (C[i] - e) / dist
            else:
                still.append(vt); continue
            r -= cr
            sn[b] = sn[b] * lam + 1; ss[b] = ss[b] * lam + r; ss2[b] = ss2[b] * lam + r * r
            trades.append((bar, i, b, r, shown))
            if shown:
                shown_active = False
        open_vt = still

        # 2) уровни ликвидности (подтверждённые пивоты)
        if not math.isnan(PL[i]):
            lvl_lo, lvl_lo_bar = PL[i], i
        if not math.isnan(PH[i]):
            lvl_hi, lvl_hi_bar = PH[i], i
        if i < warm or math.isnan(BIAS[i]):
            continue

        a = A[i]
        cands = []  # (setup, dir, sl)
        rng = H[i] - L[i]
        trend_up = EF[i] > ES[i] and ES[i] > ES[i - 5]
        trend_dn = EF[i] < ES[i] and ES[i] < ES[i - 5]
        # Откат по тренду 1м
        if trend_up and LOPB[i] <= EF[i] + p["pb_touch"] * a and CMIN[i] > ES[i] \
                and C[i] > H[i - 1] and CLV[i] > 0.6 and C[i] > EF[i]:
            cands.append((0, 1, LOPB[i] - p["sl_buf"] * a))
        if trend_dn and HIPB[i] >= EF[i] - p["pb_touch"] * a and CMAX[i] < ES[i] \
                and C[i] < L[i - 1] and CLV[i] < 0.4 and C[i] < EF[i]:
            cands.append((0, -1, HIPB[i] + p["sl_buf"] * a))
        # Снятие ликвидности
        if rng > 0:
            if not math.isnan(lvl_lo) and i - lvl_lo_bar <= p["lvl_age"] and L[i] < lvl_lo < C[i] \
                    and (min(O[i], C[i]) - L[i]) >= p["sweep_wick"] * rng and RV[i] >= p["rvol_sweep"]:
                cands.append((1, 1, L[i] - p["sl_buf"] * a))
                lvl_lo = math.nan
            if not math.isnan(lvl_hi) and i - lvl_hi_bar <= p["lvl_age"] and H[i] > lvl_hi > C[i] \
                    and (H[i] - max(O[i], C[i])) >= p["sweep_wick"] * rng and RV[i] >= p["rvol_sweep"]:
                cands.append((1, -1, H[i] + p["sl_buf"] * a))
                lvl_hi = math.nan
        # Пробой сжатия
        bw = BH[i] - BL[i]
        if bw <= p["sq_k"] * AL[i] and RV[i] >= p["rvol_sq"] and rng > 0 and abs(C[i] - O[i]) >= 0.5 * rng:
            mid = (BH[i] + BL[i]) / 2
            if C[i] > BH[i]:
                cands.append((2, 1, mid - p["sl_buf"] * a))
            elif C[i] < BL[i]:
                cands.append((2, -1, mid + p["sl_buf"] * a))

        best = None
        for s, d, sl in cands:
            key = (s, d)
            if i - last_trig.get(key, -10**9) < p["cooldown"]:
                continue
            dist = d * (C[i] - sl)
            if dist < p["sl_min"] * a:
                dist = p["sl_min"] * a
            if dist > p["sl_max"] * a or dist <= 0:
                continue
            cost_r = C[i] * p["cost_pct"] / 100 / dist
            if cost_r > p["max_cost_r"]:
                continue
            last_trig[key] = i
            ctx_raw = BIAS[i] * d
            ctx = 0 if ctx_raw > p["bias_thr"] else (2 if ctx_raw < -p["bias_thr"] else 1)
            b = s * 3 + ctx
            lcb, _ = edge(b)
            ok = sn[b] >= p["min_n"] and lcb > p["min_edge"]
            vt = [C[i], C[i] - d * dist, C[i] + d * dist * p["rr"], d, b, i, cost_r, False]
            open_vt.append(vt)
            if ok and (best is None or lcb > best[0]):
                best = (lcb, vt)
        if best is not None and not (p["one_at_a_time"] and shown_active):
            best[1][7] = True
            shown_active = True

    t = pd.DataFrame(trades, columns=["bar", "exit", "bucket", "r", "shown"])
    return t


def summary(t, days):
    out = {}
    for name, sub in (("все кандидаты", t), ("показанные", t[t.shown])):
        n = len(sub)
        out[name] = dict(n=n, per_day=round(n / days, 1),
                         wr=round(float((sub.r > 0).mean()) * 100, 1) if n else 0,
                         avg_r=round(float(sub.r.mean()), 3) if n else 0.0,
                         sum_r=round(float(sub.r.sum()), 1))
    return out


def fetch(symbol, days, cache="data"):
    os.makedirs(cache, exist_ok=True)
    path = os.path.join(cache, f"{symbol}_{days}d.csv")
    if os.path.exists(path):
        return pd.read_csv(path, index_col=0, parse_dates=True)
    end = int(time.time() * 1000)
    rows, t = [], end - days * 86_400_000
    while t < end:
        url = ("https://data-api.binance.vision/api/v3/klines"
               f"?symbol={symbol}&interval=1m&limit=1000&startTime={t}")
        for attempt in range(4):
            try:
                data = json.load(urllib.request.urlopen(url, timeout=20))
                break
            except OSError:
                time.sleep(2 ** attempt)
        else:
            raise RuntimeError(f"не удалось скачать {symbol}")
        if not data:
            break
        rows += data
        t = data[-1][0] + 60_000
    df = pd.DataFrame([r[:6] for r in rows], columns=["t", "open", "high", "low", "close", "volume"]).astype(float)
    df["t"] = pd.to_datetime(df["t"], unit="ms", utc=True)
    df = df.drop_duplicates("t").set_index("t")
    df.to_csv(path)
    return df


def main():
    ap = argparse.ArgumentParser(description="Проверка EDGE на минутных свечах Binance")
    ap.add_argument("--symbols", nargs="+", default=["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"])
    ap.add_argument("--days", type=int, default=60)
    ap.add_argument("--cost", type=float, nargs="+", default=[0.0, 0.01, 0.02, 0.04, 0.10],
                    help="издержки за вход и выход, %% (можно несколько)")
    a = ap.parse_args()
    data = {s: fetch(s, a.days) for s in a.symbols}
    print("| Издержки | Инструмент | Сетапов в день | Средний R | Итог R | Сигналов в день | Средний R | Итог R |")
    print("|---|---|---|---|---|---|---|---|")
    for c in a.cost:
        for s, df in data.items():
            days = (df.index[-1] - df.index[0]).total_seconds() / 86400
            sm = summary(run(df, dict(cost_pct=c)), days)
            al, sh = sm["все кандидаты"], sm["показанные"]
            print(f"| {c:.2f}% | {s} | {al['per_day']} | {al['avg_r']:+.3f} | {al['sum_r']:+.1f} "
                  f"| {sh['per_day']} | {sh['avg_r']:+.3f} | {sh['sum_r']:+.1f} |")


if __name__ == "__main__":
    main()
