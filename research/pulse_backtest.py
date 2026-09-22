"""Воспроизводимая проверка PULSE и минутных паттернов на реальных минутках.

    python3 research/pulse_backtest.py            # всё
    python3 research/pulse_backtest.py --quick    # только PULSE на S&P 500

Нужны numpy и pandas. Данные (~60 МБ) скачиваются один раз в research/data/:
  * S&P 500, CFD USA500.IDX, 1 минута, фев–сен 2023, без объёма
    (github.com/TheSnowGuru/Stocks-Futures-Financial-Time-series-Tick-Bar-Data)
  * BTC/USD Bitstamp, 1 минута, янв 2025 – сен 2026, с объёмом
    (github.com/ff137/bitstamp-btcusd-minute-data)

Движок `pulse()` повторяет pine/PULSE.pine бар за баром: те же входы, выходы,
пропуск неполных дней и издержки. Если правите логику в Pine — правьте и здесь,
иначе цифры в README перестанут что-либо доказывать.
"""
import os
import sys
import urllib.request

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
URLS = {
    "spx_m1.csv": "https://raw.githubusercontent.com/TheSnowGuru/Stocks-Futures-Financial-Time-series-Tick-Bar-Data/"
                  "main/indices/s%26p500/USA500IDXUSD_M1.csv",
    "btc_m1.csv": "https://raw.githubusercontent.com/ff137/bitstamp-btcusd-minute-data/"
                  "main/data/updates/btcusd_bitstamp_1min_latest.csv",
}


def fetch(name):
    os.makedirs(DATA, exist_ok=True)
    path = os.path.join(DATA, name)
    if not os.path.exists(path):
        print(f"скачиваю {name} …", flush=True)
        urllib.request.urlretrieve(URLS[name], path)
    return path


def load_spx():
    s = pd.read_csv(fetch("spx_m1.csv"), sep="\t")
    s["t"] = pd.to_datetime(s.Time)  # UTC
    s = s.rename(columns=str.lower)[["t", "open", "high", "low", "close", "volume"]]
    s["volume"] = 0.0  # у CFD объёма нет: VWAP вырождается в TWAP, как в Pine
    return s.reset_index(drop=True)


def load_btc():
    b = pd.read_csv(fetch("btc_m1.csv"))
    b["t"] = pd.to_datetime(b.timestamp, unit="s")
    return b[["t", "open", "high", "low", "close", "volume"]].reset_index(drop=True)


def rma(x, n):
    out = np.empty_like(x)
    a = 1.0 / n
    out[0] = x[0]
    for i in range(1, len(x)):
        out[i] = out[i - 1] + a * (x[i] - out[i - 1])
    return out


def atr_of(df, n=14):
    h, l, c = (df[k].to_numpy(float) for k in ["high", "low", "close"])
    pc = np.r_[c[0], c[:-1]]
    return rma(np.maximum(h - l, np.maximum(abs(h - pc), abs(l - pc))), n)


# ── PULSE ──────────────────────────────────────────────────────────────────────
DEFAULTS = dict(noiseN=14, minDays=5, useGap=True, inBuf=1.0, outBuf=0.5, every=1, t1R=0.0,
                fade=True, fadeQuiet=True, zone=0.3, fStop=1.0, fMinRR=1.0, fMax=30,
                skipOpen=0, noEntryEnd=15, mom=True, cost=0.5)


def pulse(df, tz="America/New_York", sess=(570, 960), **kw):
    """Три потока, как в индикаторе: 0 — импульс, 1 — фейд, 2 — живой (одна позиция).
    sess — минуты от полуночи в часовом поясе tz. cost — издержки на круг в цене."""
    P = dict(DEFAULTS, **kw)
    t = df.t.dt.tz_localize("UTC").dt.tz_convert(tz) if tz else df.t
    mod = (t.dt.hour * 60 + t.dt.minute).to_numpy()
    date = t.dt.date.to_numpy()
    o, h, l, c, v = (df[k].to_numpy(float) for k in ["open", "high", "low", "close", "volume"])
    atr = atr_of(df)
    alpha = 2.0 / (P["noiseN"] + 1)
    sig, cnt = {}, {}
    inPrev = False
    dOpen = prevClose = lastClose = np.nan
    sessOk = False
    cpv = cv = 0.0
    broke = False
    S = [dict(side=0) for _ in range(3)]
    trades = []
    live_days = []  # дни, где коридор появился хотя бы раз — как nDays в Pine
    for i in range(len(c)):
        m = mod[i]
        inS = sess[0] <= m < sess[1]
        first = inS and not inPrev
        outNow = (not inS) and inPrev
        inPrev = inS
        if first:
            prevClose, dOpen = lastClose, o[i]
            sessOk = 0 <= m - sess[0] <= 5
            cpv = cv = 0.0
            broke = False
        key = m - sess[0]
        up = dn = vw = np.nan
        last = False
        if inS:
            w = v[i] if v[i] > 0 else 1.0
            cpv += (h[i] + l[i] + c[i]) / 3 * w
            cv += w
            vw = cpv / cv
            if sessOk:
                s_old = sig.get(key)
                n_old = cnt.get(key, 0)
                sg = s_old if n_old >= P["minDays"] else np.nan
                mv = abs(c[i] / dOpen - 1)
                sig[key] = mv if s_old is None else s_old + alpha * (mv - s_old)
                cnt[key] = n_old + 1
                if not np.isnan(sg):
                    gap = P["useGap"] and not np.isnan(prevClose)
                    up = (max(dOpen, prevClose) if gap else dOpen) * (1 + sg)
                    dn = (min(dOpen, prevClose) if gap else dOpen) * (1 - sg)
                    if not live_days or live_days[-1] != date[i]:
                        live_days.append(date[i])
            lastClose = c[i]
            last = key == sess[1] - sess[0] - 1
        A = atr[i]
        # выходы
        for k in range(3):
            s = S[k]
            if s["side"] == 0 or s["bar"] >= i:
                continue
            sd, x = s["side"], None
            if outNow or first:
                x = o[i]
            elif s["kind"] == 0:
                if P["t1R"] > 0 and not s["t1"]:
                    fav = (h[i] - s["e"]) if sd == 1 else (s["e"] - l[i])
                    if fav >= P["t1R"] * s["risk"]:
                        s["t1"], s["got"] = True, 0.5 * P["t1R"]
                trl = (max(up, vw) - P["outBuf"] * A) if sd == 1 else (min(dn, vw) + P["outBuf"] * A)
                if np.isnan(trl) or (sd == 1 and c[i] < trl) or (sd == -1 and c[i] > trl) or last:
                    x = c[i]
            else:
                if (sd == 1 and l[i] <= s["stop"]) or (sd == -1 and h[i] >= s["stop"]):
                    x = min(s["stop"], o[i]) if sd == 1 else max(s["stop"], o[i])
                elif (sd == 1 and h[i] >= s["tgt"]) or (sd == -1 and l[i] <= s["tgt"]):
                    x = s["tgt"]
                elif i - s["bar"] >= P["fMax"] or last:
                    x = c[i]
            if x is not None:
                rem = 0.5 if s["t1"] else 1.0
                R = s["got"] + rem * (x - s["e"]) * sd / s["risk"] - s["cost"] / s["risk"]
                trades.append(dict(kind=s["kind"], stream=k, side=sd, i0=s["bar"], i1=i, R=R,
                                   pts=R * s["risk"], e=s["e"], day=date[s["bar"]]))
                S[k] = dict(side=0)
        ready = inS and not last and not np.isnan(up) and A > 0
        if not ready:
            continue
        canEnter = key >= P["skipOpen"] and (sess[1] - m - 1) >= P["noEntryEnd"]
        brokeNow = c[i] > up + P["inBuf"] * A or c[i] < dn - P["inBuf"] * A
        cands = []
        if canEnter and P["mom"] and key % P["every"] == 0:
            sd = 1 if (c[i] > up + P["inBuf"] * A and c[i] > vw) else -1 if (c[i] < dn - P["inBuf"] * A and c[i] < vw) else 0
            if sd:
                st = (max(up, vw) - P["outBuf"] * A) if sd == 1 else (min(dn, vw) + P["outBuf"] * A)
                cands.append((0, sd, st, np.nan))
        if canEnter and P["fade"] and not (P["fadeQuiet"] and broke):
            sS, sL = h[i] + P["fStop"] * A, l[i] - P["fStop"] * A
            if h[i] >= up - P["zone"] * A and c[i] < up and c[i] < o[i] and (c[i] - vw) >= P["fMinRR"] * (sS - c[i]):
                cands.append((1, -1, sS, vw))
            elif l[i] <= dn + P["zone"] * A and c[i] > dn and c[i] > o[i] and (vw - c[i]) >= P["fMinRR"] * (c[i] - sL):
                cands.append((1, 1, sL, vw))
        for kind, sd, st, tg in cands:
            pos = dict(side=sd, kind=kind, e=c[i], stop=st, tgt=tg, risk=max(abs(c[i] - st), 1e-9),
                       cost=P["cost"], bar=i, t1=False, got=0.0)
            if S[kind]["side"] == 0:
                S[kind] = dict(pos)
            if S[2]["side"] == 0:
                S[2] = dict(pos)
        broke = broke or brokeNow
    T = pd.DataFrame(trades)
    T.attrs["days"] = live_days
    return T


def show(T, label, unit="п.", scale=None):
    """scale(T) → величина на сделку в нужных единицах; по умолчанию пункты."""
    val = T.pts if scale is None else scale(T)
    days = T.attrs["days"]
    split = days[len(days) // 2]
    nd = max(len(days), 1)
    print(f"\n── {label}  (дней с коридором: {nd})")
    print(f"{'поток':<12}{'сделок/д':>9}{'прибыльных/д':>14}{'винрейт':>9}{'ср. R':>8}"
          f"{'ср. ' + unit:>9}{unit + '/день':>10}{'1-я половина':>14}{'2-я половина':>14}")
    for name, m in [("импульс", (T.stream == 0)), ("фейд", (T.stream == 1)), ("живой", (T.stream == 2))]:
        X, V = T[m], val[m]
        if len(X) == 0:
            print(f"{name:<12}{'—':>9}")
            continue
        a, b = V[X.day < split].sum(), V[X.day >= split].sum()
        print(f"{name:<12}{len(X)/nd:>9.2f}{(X.R > 0).sum()/nd:>14.2f}{100*(X.R > 0).mean():>8.0f}%"
              f"{X.R.mean():>+8.3f}{V.mean():>+9.2f}{V.sum()/nd:>+10.2f}{a:>+14.1f}{b:>+14.1f}")


# ── минутные паттерны, которые НЕ прошли проверку ──────────────────────────────
def roll(x, n, f):
    return getattr(pd.Series(x).rolling(n, min_periods=1), f)().to_numpy()


def sweep_test(df, k=2, lookback=30, target_R=2.0, max_bars=30):
    """Свип минимума/максимума за lookback баров и реклейм в пределах k баров.
    Возвращает брутто-результат (без издержек) на сделку в R и в цене."""
    o, h, l, c = (df[x].to_numpy(float) for x in ["open", "high", "low", "close"])
    atr = atr_of(df)
    n = len(c)
    rng = np.maximum(h - l, 1e-12)
    ext_l, ext_h = roll(l, k, "min"), roll(h, k, "max")
    lv_l = np.r_[np.full(k, np.nan), roll(l, lookback, "min")[:-k]]
    lv_h = np.r_[np.full(k, np.nan), roll(h, lookback, "max")[:-k]]
    res = []
    for sd, lv, ext in ((1, lv_l, ext_l), (-1, lv_h, ext_h)):
        pierce = (lv - ext) * sd
        cp = (c - l) / rng if sd == 1 else (h - c) / rng
        hit = (pierce > 0) & (pierce <= atr) & ((c - lv) * sd > 0) & ((c - o) * sd > 0) & (cp >= 0.5)
        for i in np.nonzero(hit)[0]:
            if i + 1 >= n or not atr[i] > 0:
                continue
            e = c[i]
            st = ext[i] - sd * 0.2 * atr[i]
            risk = max((e - st) * sd, 0.5 * atr[i])
            st = e - sd * risk
            tg = e + sd * target_R * risk
            R = None
            for j in range(i + 1, min(i + 1 + max_bars, n)):
                if (sd == 1 and l[j] <= st) or (sd == -1 and h[j] >= st):
                    px = min(st, o[j]) if sd == 1 else max(st, o[j])
                    R = (px - e) * sd / risk
                    break
                if (sd == 1 and h[j] >= tg) or (sd == -1 and l[j] <= tg):
                    R = target_R
                    break
            if R is None:
                R = (c[min(i + max_bars, n - 1)] - e) * sd / risk
            res.append((R, R * risk, R * risk / e * 1e4))
    r = np.array(res)
    return dict(n=len(r), R=r[:, 0].mean(), px=r[:, 1].mean(), bp=r[:, 2].mean())


def main():
    quick = "--quick" in sys.argv
    spx = load_spx()
    # 2 тика ES = 0.5 пункта на круг — умолчание индикатора
    show(pulse(spx, cost=0.5), "S&P 500 · PULSE по умолчанию · издержки 0.5 п. на круг")
    if quick:
        return
    show(pulse(spx, cost=0.5, t1R=1.0), "S&P 500 · половина на +1R")
    show(pulse(spx, cost=0.5, fadeQuiet=False), "S&P 500 · фейд без фильтра тихого дня")
    for lab, kw in [("вход 0.5 ATR", dict(inBuf=0.5)), ("вход 1.5 ATR", dict(inBuf=1.5)),
                    ("выход 1.0 ATR", dict(outBuf=1.0)), ("выход 0 ATR", dict(outBuf=0.0)),
                    ("10 дней", dict(noiseN=10)), ("20 дней", dict(noiseN=20)),
                    ("раз в 5 минут", dict(every=5)), ("раз в 15 минут", dict(every=15))]:
        T = pulse(spx, cost=0.5, fade=False, **kw)
        show(T, f"S&P 500 · только импульс · {lab}")

    s = sweep_test(spx)
    print(f"\n── S&P 500 · свип-реклейм на минутках, цель 2R, БЕЗ издержек: "
          f"{s['n']} сделок, {s['R']:+.3f}R, {s['px']:+.3f} п. на сделку")

    btc = load_btc()
    s = sweep_test(btc)
    print(f"── BTC/USD · тот же свип-реклейм, БЕЗ издержек: {s['n']} сделок, {s['R']:+.3f}R, "
          f"{s['bp']:+.2f} б.п. на сделку (≈ спред: это отскок принтов от бид/аск, а не эдж)")
    T = pulse(btc, cost=0.0)
    bp = lambda X: X.pts / X.e * 1e4
    show(T, "BTC/USD · PULSE, сессия 09:30–16:00 NY, БЕЗ издержек", unit="б.п.", scale=bp)


if __name__ == "__main__":
    main()
