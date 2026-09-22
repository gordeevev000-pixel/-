"""Проверка EDGE на истории: Python-двойник логики из EDGE.pine.

Запуск:
    pip install numpy pandas
    python edge_backtest.py --source yahoo --symbols NQ=F ES=F --ticks --cost 0 1 2
    python edge_backtest.py --source moex --market futures --symbols SiZ6 --days 60 --ticks --cost 0 1 2
    python edge_backtest.py --source moex --symbols SBER GAZP --cost 0 0.05 0.1
    python edge_backtest.py --source binance --symbols BTCUSDT --cost 0 0.04

Источники минутных свечей (бесплатные, без ключей):
    binance — крипта, любая глубина;
    moex    — акции и фьючерсы Мосбиржи (ISS), любая глубина, фьючерсы — конкретный контракт;
    yahoo   — фьючерсы и акции США, только последние ~30 дней.
Данные кэшируются в ./data.

Вход — по закрытию сигнального бара, стоп и тейк в одном баре считаются стопом,
издержки вычитаются из каждой сделки, на разрыве сессии сделка закрывается по последней
цене перед ним. Старшие ТФ — только закрытые бары.
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
    cost_pct=0.04, cost_ticks=None, tick=None, max_cost_r=0.25,
    gap_min=4, after_gap=3, tod_alpha=0.1, tod_min=3,
    memory=150, prior_k=10, prior_mu=-0.05, z=0.5, min_n=20, min_edge=0.05,
    cooldown=5, one_at_a_time=True, mode="strict",   # strict | soft | all — как «Какие сигналы показывать»
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


def infer_tick(prices):
    u = np.unique(np.round(prices.to_numpy(), 8))
    d = np.round(np.diff(u), 8)
    return float(d[d > 0].min())


def features(df, p):
    f = pd.DataFrame(index=df.index)
    rng_raw = df.high - df.low
    # ATR по диапазону свечи: разрыв на открытии/после клиринга не раздувает его
    f["atr"] = rma(rng_raw, p["atr_len"])
    f["atr_long"] = rma(rng_raw, p["atr_long"])
    f["ef"] = ema(df.close, p["ema_f"])
    f["es"] = ema(df.close, p["ema_s"])
    rng = (df.high - df.low).replace(0, np.nan)
    f["clv"] = ((df.close - df.low) / rng).fillna(0.5)
    # Относительный объём: к обычному объёму в ЭТУ минуту суток (утро и вечер не путаются с серединой дня)
    slot = df.index.hour * 60 + df.index.minute
    g = df.volume.groupby(slot)
    tod = g.transform(lambda v: v.ewm(alpha=p["tod_alpha"], adjust=False).mean().shift(1))
    cnt = g.cumcount()
    plain = df.volume / df.volume.rolling(p["rvol_len"]).mean()
    f["rvol"] = np.where((cnt >= p["tod_min"]) & (tod > 0), df.volume / tod, plain)
    f["rvol"] = f["rvol"].fillna(1)
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
    T = df.index.asi8 // 60_000_000_000  # минуты
    since_break = 10**9
    tick = p["tick"] or infer_tick(df.close)

    def edge(b):
        n = sn[b]
        mean = (ss[b] + p["prior_k"] * p["prior_mu"]) / (n + p["prior_k"])
        var = max(ss2[b] / n - (ss[b] / n) ** 2, 0.25) if n > 0 else 1.0
        return mean - p["z"] * math.sqrt(var / (n + p["prior_k"])), mean

    def close_vt(vt, r, i):
        nonlocal shown_active
        e, sl, tp, d, b, bar, cr, shown = vt
        r -= cr
        sn[b] = sn[b] * lam + 1; ss[b] = ss[b] * lam + r; ss2[b] = ss2[b] * lam + r * r
        trades.append((bar, i, b, r, shown))
        if shown:
            shown_active = False

    for i in range(len(C)):
        # 0) разрыв сессии (ночь, клиринг): открытые сделки закрываются по последней цене до разрыва
        if i > 0 and T[i] - T[i - 1] > p["gap_min"]:
            for vt in open_vt:
                close_vt(vt, vt[3] * (C[i - 1] - vt[0]) / abs(vt[0] - vt[1]), i - 1)
            open_vt = []
            since_break = 0
        else:
            since_break += 1

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
            close_vt(vt, r, i)
        open_vt = still

        # 2) уровни ликвидности (подтверждённые пивоты)
        if not math.isnan(PL[i]):
            lvl_lo, lvl_lo_bar = PL[i], i
        if not math.isnan(PH[i]):
            lvl_hi, lvl_hi_bar = PH[i], i
        if i < warm or math.isnan(BIAS[i]) or since_break < p["after_gap"]:
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
            cost_px = C[i] * p["cost_pct"] / 100 if p["cost_ticks"] is None else p["cost_ticks"] * tick
            cost_r = cost_px / dist
            if cost_r > p["max_cost_r"] and p["mode"] != "all":
                continue
            last_trig[key] = i
            ctx_raw = BIAS[i] * d
            ctx = 0 if ctx_raw > p["bias_thr"] else (2 if ctx_raw < -p["bias_thr"] else 1)
            b = s * 3 + ctx
            lcb, mean = edge(b)
            ok = (sn[b] >= p["min_n"] and lcb > p["min_edge"]) or p["mode"] == "all" \
                or (p["mode"] == "soft" and sn[b] >= max(5, p["min_n"] / 2) and mean > 0)
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
    for name, sub in (("все кандидаты", t), ("показанные", t[t.shown.astype(bool)])):
        n = len(sub)
        out[name] = dict(n=n, per_day=round(n / days, 1),
                         wr=round(float((sub.r > 0).mean()) * 100, 1) if n else 0,
                         avg_r=round(float(sub.r.mean()), 3) if n else 0.0,
                         sum_r=round(float(sub.r.sum()), 1))
    return out


def _get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    for attempt in range(4):
        try:
            return json.load(urllib.request.urlopen(req, timeout=30))
        except OSError:
            time.sleep(2 ** attempt)
    raise RuntimeError(f"не удалось скачать {url}")


def _binance(symbol, days):
    end = int(time.time() * 1000)
    rows, t = [], end - days * 86_400_000
    while t < end:
        data = _get("https://data-api.binance.vision/api/v3/klines"
                    f"?symbol={symbol}&interval=1m&limit=1000&startTime={t}")
        if not data:
            break
        rows += data
        t = data[-1][0] + 60_000
    df = pd.DataFrame([r[:6] for r in rows], columns=["t", "open", "high", "low", "close", "volume"]).astype(float)
    df["t"] = pd.to_datetime(df["t"], unit="ms", utc=True)
    return df.set_index("t")


def _moex(symbol, days, market):
    path = {"shares": "stock/markets/shares", "futures": "futures/markets/forts"}[market]
    till = pd.Timestamp.now(tz="Europe/Moscow").normalize()
    frm = till - pd.Timedelta(days=days)
    rows, start = [], 0
    while True:
        d = _get(f"https://iss.moex.com/iss/engines/{path}/securities/{symbol}/candles.json"
                 f"?from={frm:%Y-%m-%d}&till={till:%Y-%m-%d}&interval=1&iss.meta=off&start={start}")["candles"]["data"]
        if not d:
            break
        rows += d
        start += len(d)
    df = pd.DataFrame(rows, columns=["open", "close", "high", "low", "value", "volume", "begin", "end"])
    df["t"] = pd.to_datetime(df.begin).dt.tz_localize("Europe/Moscow")
    return df.set_index("t")[["open", "high", "low", "close", "volume"]].astype(float)


def _yahoo(symbol, days):
    frames, now = [], int(time.time())
    for k in range(min(days, 29) // 7, -1, -1):   # Yahoo отдаёт 1м не глубже ~30 дней, кусками по 7
        p1, p2 = now - (k + 1) * 7 * 86400 + 3600, now - k * 7 * 86400
        try:
            r = _get(f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
                     f"?interval=1m&period1={p1}&period2={p2}")["chart"]["result"][0]
        except (RuntimeError, KeyError, TypeError):
            continue
        if "timestamp" not in r:
            continue
        q = r["indicators"]["quote"][0]
        idx = pd.to_datetime(r["timestamp"], unit="s", utc=True).tz_convert(r["meta"]["exchangeTimezoneName"])
        frames.append(pd.DataFrame({k2: q[k2] for k2 in ("open", "high", "low", "close", "volume")}, index=idx))
    df = pd.concat(frames).dropna()
    df.index.name = "t"
    return df


def fetch(symbol, days, source="binance", market="shares", cache="data"):
    os.makedirs(cache, exist_ok=True)
    path = os.path.join(cache, f"{source}_{symbol.replace('=', '_')}_{days}d.csv")
    if os.path.exists(path):
        df = pd.read_csv(path, index_col=0)
        df.index = pd.to_datetime(df.index)
    else:
        df = {"binance": lambda: _binance(symbol, days),
              "moex": lambda: _moex(symbol, days, market),
              "yahoo": lambda: _yahoo(symbol, days)}[source]()
        df.to_csv(path)
    df = df[~df.index.duplicated()].sort_index()
    df = df[df.index.second == 0]                       # незакрытая текущая свеча
    return df[(df.high >= df.low) & (df.close > 0)]


def main():
    ap = argparse.ArgumentParser(description="Проверка EDGE на минутных свечах")
    ap.add_argument("--source", choices=["binance", "moex", "yahoo"], default="binance")
    ap.add_argument("--market", choices=["shares", "futures"], default="shares", help="для moex")
    ap.add_argument("--symbols", nargs="+", default=["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"])
    ap.add_argument("--days", type=int, default=60)
    ap.add_argument("--cost", type=float, nargs="+", default=[0.0, 0.01, 0.02, 0.04, 0.10],
                    help="издержки за вход и выход: %% от цены или тики (с --ticks)")
    ap.add_argument("--ticks", action="store_true", help="издержки заданы в тиках")
    ap.add_argument("--tick", type=float, default=None, help="шаг цены; по умолчанию определяется по данным")
    a = ap.parse_args()
    data = {s: fetch(s, a.days, a.source, a.market) for s in a.symbols}
    unit = "тик" if a.ticks else "%"
    print(f"| Издержки | Инструмент | Сетапов в день | Средний R | Итог R | Сигналов в день | Средний R | Итог R |")
    print("|---|---|---|---|---|---|---|---|")
    for c in a.cost:
        for s, df in data.items():
            days = df.index.normalize().nunique()
            extra = dict(cost_ticks=c, tick=a.tick) if a.ticks else dict(cost_pct=c)
            sm = summary(run(df, extra), days)
            al, sh = sm["все кандидаты"], sm["показанные"]
            print(f"| {c:g} {unit} | {s} | {al['per_day']} | {al['avg_r']:+.3f} | {al['sum_r']:+.1f} "
                  f"| {sh['per_day']} | {sh['avg_r']:+.3f} | {sh['sum_r']:+.1f} |")


if __name__ == "__main__":
    main()
