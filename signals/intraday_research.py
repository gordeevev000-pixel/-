"""Исследование внутридневных сценариев — то, что НЕ вошло в индикатор, и базовые частоты.

Здесь собраны все проверенные семейства сценариев (пробой диапазона открытия, ложный пробой,
закрытие гэпа, моментум конца дня, «шумовая зона», пробой ночного диапазона, модель прогноза
остатка дня). Это исследовательский код: правила грубее, чем в scenario_backtest.py, и
служат только для таблицы «Что проверено и отброшено» в SCENARIOS.md.

Запуск:  python signals/intraday_research.py data/hd
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import intraday_lab as L  # noqa: E402

D = Path("data/hd")
CACHE = {}


def prep(name, tf):
    k = (name, tf)
    if k not in CACHE:
        mk = L.MARKETS[name]
        df = L.resample(L.load(D, name), tf)
        s, sid = L.sessions(df, mk)
        ctx = L.bar_context(df, s, sid, mk)
        atr1 = L_atr(df)
        CACHE[k] = (df, s, sid, ctx, atr1)
    return CACHE[k]


def L_atr(df, n=14):
    h, l, c = (df[k].to_numpy() for k in ("high", "low", "close"))
    pc = np.r_[c[0], c[:-1]]
    tr = np.maximum(h - l, np.maximum(abs(h - pc), abs(l - pc)))
    return pd.Series(tr).ewm(alpha=1 / n, adjust=False).mean().to_numpy()


def trade(df, i, d, entry, stop, target, last, trail_k=0.0, atr=None):
    o, h, l, c = (df[k].to_numpy() for k in ("open", "high", "low", "close"))
    a = atr if atr is not None else np.zeros(len(c))
    return L.run_trade(o, h, l, c, i, d, entry, stop, target, last, trail_k, a)


def events_orb(name, tf, W=30, stop_mode="opp", cutoff=180, tgt_r=0.0):
    """S1: пробой диапазона первых W минут, вход по закрытию за ним, стоп — противоположная граница
    (или середина), выход в конце сессии или по цели tgt_r·R."""
    df, s, sid, ctx, atr = prep(name, tf)
    o, h, l, c = (df[k].to_numpy() for k in ("open", "high", "low", "close"))
    m = ctx["m"]; orh = ctx[f"or{W}h"]; orl = ctx[f"or{W}l"]
    rows = []
    for k in range(len(s)):
        a, b = s.i0.iat[k], s.i1.iat[k]
        for i in range(a, b):
            if np.isnan(orh[i]) or m[i] < W - ctx["step"] or m[i] > cutoff:
                continue
            d = 1 if c[i] > orh[i] else -1 if c[i] < orl[i] else 0
            if d == 0:
                continue
            mid = (orh[i] + orl[i]) / 2
            stop = (orl[i] if d == 1 else orh[i]) if stop_mode == "opp" else mid
            if (c[i] - stop) * d <= 0:
                break
            tgt = c[i] + d * tgt_r * (c[i] - stop) * d if tgt_r > 0 else 0.0
            r, j, why = L.run_trade(o, h, l, c, i, d, c[i], stop, tgt, b, 0.0, atr)
            rows.append(dict(sess=k, i=i, d=d, entry=c[i], risk=(c[i] - stop) * d, r=r, why=why, m=m[i]))
            break                                               # одна сделка за сессию
    return finish(name, s, rows)


def finish(name, s, rows):
    ev = pd.DataFrame(rows)
    if ev.empty:
        return ev
    mk = L.MARKETS[name]
    ev["cost_r"] = mk.cost_pct * ev.entry / ev.risk
    ev["r_net"] = ev.r - ev.cost_r
    ev["date"] = pd.to_datetime(s.date.to_numpy()[ev.sess])
    ev["oos"] = ev.date.dt.year >= 2023
    for col in ("gap", "on_range", "nr7", "inside", "atr_d", "prev_range"):
        ev[col] = s[col].to_numpy()[ev.sess]
    ev["market"] = name
    return ev


def summ(ev, by=None):
    def f(g):
        r = g.r_net
        return pd.Series({"n": len(r), "win%": (r > 0).mean() * 100, "R": r.mean(), "R_gross": g.r.mean(),
                          "cost_R": g.cost_r.mean(), "PF": r[r > 0].sum() / max(-r[r < 0].sum(), 1e-9)})
    return (ev.groupby(by).apply(f) if by else f(ev).to_frame().T).round(3)


def add_day_features(ev, name, tf):
    """Признаки дня, известные к моменту входа."""
    df, s, sid, ctx, atr = prep(name, tf)
    W = int(ev.W.iat[0]) if "W" in ev else 30
    cl = s.close.to_numpy()
    sma20 = pd.Series(cl).rolling(20).mean().shift().to_numpy()      # по закрытиям прошлых сессий
    pdc = s.pdc.to_numpy(); pdh = s.pdh.to_numpy(); pdl = s.pdl.to_numpy(); op = s.open.to_numpy()
    k = ev.sess.to_numpy(); d = ev.d.to_numpy(); i = ev.i.to_numpy()
    orh, orl = ctx[f"or{W}h"][i], ctx[f"or{W}l"][i]
    ev["or_w"] = (orh - orl) / ev.atr_d
    ev["gap_al"] = np.sign(ev.gap) * d
    ev["trend_al"] = np.sign(pdc[k] - sma20[k]) * d
    ev["open_loc"] = np.where(op[k] > pdh[k], 1, np.where(op[k] < pdl[k], -1, 0)) * d
    prev_clv = ((s.close - s.low) / (s.high - s.low)).shift().to_numpy()
    ev["prev_clv_al"] = (prev_clv[k] - 0.5) * 2 * d
    return ev


def events_fail(name, tf, level="or30", K=15, tgt_r=0.0, buf=0.05, cutoff=None, exit_m=None):
    """S2: ложный пробой уровня. Цена прошла за уровень и в течение K минут бар закрылся обратно —
    вход против пробоя. Стоп — экстремум выноса + buf·ATR_d. Выход: цель tgt_r·R или конец сессии."""
    df, s, sid, ctx, atr = prep(name, tf)
    o, h, l, c = (df[k].to_numpy() for k in ("open", "high", "low", "close"))
    m = ctx["m"]; step = int(ctx["step"])
    rows = []
    for k in range(len(s)):
        a, b = s.i0.iat[k], s.i1.iat[k]
        ad = s.atr_d.iat[k]
        if not ad > 0:
            continue
        for side in (1, -1):                    # 1: уровень сверху (пробой вверх → шорт), -1: снизу
            if level.startswith("or"):
                W = int(level[2:])
                lv_arr = ctx[f"or{W}h"] if side == 1 else ctx[f"or{W}l"]
            elif level == "pd":
                lv = s.pdh.iat[k] if side == 1 else s.pdl.iat[k]
                lv_arr = None
            elif level == "on":
                lv = s.on_high.iat[k] if side == 1 else s.on_low.iat[k]
                lv_arr = None
            brk_i, ext = -1, np.nan
            for i in range(a, b):
                if cutoff is not None and m[i] > cutoff:
                    break
                L_ = lv_arr[i] if lv_arr is not None else lv
                if np.isnan(L_):
                    continue
                if lv_arr is not None and i > a and np.isnan(lv_arr[i - 1]):
                    continue                     # бар, на котором уровень только что сформировался
                if brk_i < 0:
                    if (h[i] > L_ if side == 1 else l[i] < L_):
                        brk_i, ext = i, (h[i] if side == 1 else l[i])
                        if (c[i] < L_ if side == 1 else c[i] > L_):
                            pass                 # вынос и возврат в одной свече — проверим ниже
                        else:
                            continue
                    else:
                        continue
                ext = max(ext, h[i]) if side == 1 else min(ext, l[i])
                if (m[i] - m[brk_i]) > K:
                    break
                if (c[i] < L_ if side == 1 else c[i] > L_):
                    d = -side
                    stop = ext + side * buf * ad
                    risk = (c[i] - stop) * d
                    if risk <= 0:
                        break
                    tgt = c[i] + d * tgt_r * risk if tgt_r > 0 else 0.0
                    last = b
                    if exit_m is not None:
                        last = min(b, i + max(1, exit_m // step))
                    r, j, why = L.run_trade(o, h, l, c, i, d, c[i], stop, tgt, last, 0.0, atr)
                    rows.append(dict(sess=k, i=i, d=d, entry=c[i], risk=risk, r=r, why=why, m=m[i], side=side))
                    break
    return finish(name, s, rows)


def events_gap(name, tf, lo=0.1, hi=0.6, mode="confirm", W=15):
    """S3: закрытие гэпа. confirm — после диапазона первых W минут вход на пробое его границы
    в сторону закрытия гэпа; цель — вчерашнее закрытие, стоп — другая граница; выход в конце сессии."""
    df, s, sid, ctx, atr = prep(name, tf)
    o, h, l, c = (df[k].to_numpy() for k in ("open", "high", "low", "close"))
    m = ctx["m"]
    rows = []
    for k in range(len(s)):
        g = s.gap.iat[k]
        if not (lo <= abs(g) <= hi):
            continue
        a, b = s.i0.iat[k], s.i1.iat[k]
        pdc = s.pdc.iat[k]
        d = -int(np.sign(g))                      # в сторону закрытия гэпа
        for i in range(a, b):
            orh, orl = ctx[f"or{W}h"][i], ctx[f"or{W}l"][i]
            if np.isnan(orh):
                continue
            if m[i] > 120:
                break
            if (c[i] < orl if d == -1 else c[i] > orh):
                stop = orh if d == -1 else orl
                risk = (c[i] - stop) * d
                if risk <= 0 or (pdc - c[i]) * d <= 0:
                    break
                r, j, why = L.run_trade(o, h, l, c, i, d, c[i], stop, pdc, b, 0.0, atr)
                rows.append(dict(sess=k, i=i, d=d, entry=c[i], risk=risk, r=r, why=why, m=m[i]))
                break
    return finish(name, s, rows)


def events_lateMom(name, tf, first=30, last=30, stop_k=0.15):
    """S4: внутридневной моментум (Gao и др.): знак движения от вчерашнего закрытия до открытия+first минут
    задаёт направление сделки на последние last минут сессии. Стоп — stop_k·ATR_d."""
    df, s, sid, ctx, atr = prep(name, tf)
    o, h, l, c = (df[k].to_numpy() for k in ("open", "high", "low", "close"))
    m = ctx["m"]; step = int(ctx["step"])
    mk = L.MARKETS[name]
    sess_len = (int(mk.close[:2]) * 60 + int(mk.close[3:])) - (int(mk.open[:2]) * 60 + int(mk.open[3:]))
    rows = []
    for k in range(len(s)):
        a, b = s.i0.iat[k], s.i1.iat[k]
        ad = s.atr_d.iat[k]; pdc = s.pdc.iat[k]
        if not ad > 0 or np.isnan(pdc):
            continue
        mm = m[a:b + 1]
        j1 = np.flatnonzero(mm + step >= first)
        j2 = np.flatnonzero(mm + step >= sess_len - last)
        if not len(j1) or not len(j2) or j2[0] <= j1[0]:
            continue
        i1, i2 = a + j1[0], a + j2[0]
        d = 1 if c[i1] > pdc else -1
        stop = c[i2] - d * stop_k * ad
        r, j, why = L.run_trade(o, h, l, c, i2, d, c[i2], stop, 0.0, b, 0.0, atr)
        rows.append(dict(sess=k, i=i2, d=d, entry=c[i2], risk=stop_k * ad, r=r, why=why, m=m[i2]))
    return finish(name, s, rows)


def noise_bands(name, tf, lookback=14):
    """σ(t,k) — средний |close/open − 1| на k-й минуте сессии за lookback прошлых сессий."""
    df, s, sid, ctx, atr = prep(name, tf)
    c = df.close.to_numpy(); m = ctx["m"]; step = int(ctx["step"])
    mk = L.MARKETS[name]
    sess_len = (int(mk.close[:2]) * 60 + int(mk.close[3:])) - (int(mk.open[:2]) * 60 + int(mk.open[3:]))
    K = sess_len // step + 1
    mv = np.full((len(s), K), np.nan)
    for k in range(len(s)):
        a, b = s.i0.iat[k], s.i1.iat[k] + 1
        op = s.open.iat[k]
        kk = m[a:b] // step
        ok = (kk >= 0) & (kk < K)
        mv[k, kk[ok]] = np.abs(c[a:b][ok] / op - 1)
    sig = pd.DataFrame(mv).rolling(lookback, min_periods=lookback // 2).mean().shift().to_numpy()
    return sig, K


def events_noise(name, tf, lookback=14, every=30, use_twap=True, risk_floor=0.1, cutoff_last=0):
    """S5: «шумовая зона» Zarattini–Barbon–Aziz (2024). Вход, когда закрытие вышло за границу
    open·(1±σ) (с поправкой на гэп), проверка раз в `every` минут; стоп/выход — возврат за
    max(граница, TWAP) на такой же проверке; принудительный выход в конце сессии."""
    df, s, sid, ctx, atr = prep(name, tf)
    o, h, l, c = (df[k].to_numpy() for k in ("open", "high", "low", "close"))
    m = ctx["m"]; step = int(ctx["step"]); tw = ctx["twap"]
    sig, K = noise_bands(name, tf, lookback)
    rows = []
    for k in range(len(s)):
        a, b = s.i0.iat[k], s.i1.iat[k]
        ad = s.atr_d.iat[k]; pdc = s.pdc.iat[k]
        if not ad > 0 or np.isnan(pdc):
            continue
        op = s.open.iat[k]
        ub_base, lb_base = max(op, pdc), min(op, pdc)
        pos, entry, risk, ei = 0, 0.0, 0.0, -1
        for i in range(a, b + 1):
            kk = m[i] // step
            if kk < 0 or kk >= K or np.isnan(sig[k, kk]):
                continue
            check = ((m[i] + step) % every == 0) or i == b
            ub, lb = ub_base * (1 + sig[k, kk]), lb_base * (1 - sig[k, kk])
            if pos != 0:
                trail = max(ub, tw[i]) if pos == 1 else min(lb, tw[i])
                if i == b or (check and (c[i] - trail) * pos < 0):
                    rows.append(dict(sess=k, i=ei, d=pos, entry=entry, risk=risk, r=(c[i] - entry) * pos / risk,
                                     why=2 if i == b else 0, m=m[ei], exit_i=i))
                    pos = 0
            if pos == 0 and check and i < b - cutoff_last // step:
                d = 1 if c[i] > ub else -1 if c[i] < lb else 0
                if d != 0:
                    stop = max(ub, tw[i]) if d == 1 else min(lb, tw[i]) if use_twap else (ub if d == 1 else lb)
                    pos, entry, ei = d, c[i], i
                    risk = max((c[i] - stop) * d, risk_floor * ad * 0.25)
    return finish(name, s, rows)


def events_orb_candle(name, tf, W=5, tgt_r=10.0):
    """Zarattini & Aziz (2023): направление — цвет первой W-минутной свечи сессии; вход по её закрытию;
    стоп — противоположный экстремум этой свечи; цель tgt_r·R, иначе выход в конце сессии.
    Доджи (open == close) — без сделки."""
    df, s, sid, ctx, atr = prep(name, tf)
    o, h, l, c = (df[k].to_numpy() for k in ("open", "high", "low", "close"))
    m = ctx["m"]; step = int(ctx["step"])
    rows = []
    if W < step:
        return finish(name, s, rows)
    for k in range(len(s)):
        a, b = s.i0.iat[k], s.i1.iat[k]
        if m[a] != 0:
            continue
        j = a + W // step - 1
        if j >= b or m[j] + step != W:
            continue
        op, cl = o[a], c[j]
        hi, lo = h[a:j + 1].max(), l[a:j + 1].min()
        d = 1 if cl > op else -1 if cl < op else 0
        if d == 0:
            continue
        stop = lo if d == 1 else hi
        risk = (cl - stop) * d
        if risk <= 0:
            continue
        r, jj, why = L.run_trade(o, h, l, c, j, d, cl, stop, cl + d * tgt_r * risk, b, 0.0, atr)
        rows.append(dict(sess=k, i=j, d=d, entry=cl, risk=risk, r=r, why=why, m=m[j]))
    return finish(name, s, rows)


def events_onbreak(name, tf, stop_mode="opp", cutoff=240, market=None):
    """Пробой ночного диапазона (от закрытия прошлой сессии до открытия этой) после открытия:
    первое закрытие за ним, стоп — другая граница (или середина), выход в конце сессии."""
    df, s, sid, ctx, atr = prep(name, tf) if market is None else market
    o, h, l, c = (df[k].to_numpy() for k in ("open", "high", "low", "close"))
    m = ctx["m"]
    rows = []
    for k in range(len(s)):
        a, b = s.i0.iat[k], s.i1.iat[k]
        H, Lo = s.on_high.iat[k], s.on_low.iat[k]
        ad = s.atr_d.iat[k]
        if np.isnan(H) or not ad > 0 or (H - Lo) < 0.1 * ad:
            continue
        for i in range(a, b):
            if m[i] > cutoff:
                break
            d = 1 if c[i] > H else -1 if c[i] < Lo else 0
            if d == 0:
                continue
            stop = (Lo if d == 1 else H) if stop_mode == "opp" else (H + Lo) / 2
            risk = (c[i] - stop) * d
            if risk <= 0:
                break
            r, j, why = L.run_trade(o, h, l, c, i, d, c[i], stop, 0.0, b, 0.0, atr)
            rows.append(dict(sess=k, i=i, d=d, entry=c[i], risk=risk, r=r, why=why, m=m[i]))
            break
    return finish(name, s, rows)


def prep_custom(name, tf, mk):
    df = L.resample(L.load(D, name), tf)
    s, sid = L.sessions(df, mk)
    ctx = L.bar_context(df, s, sid, mk)
    return df, s, sid, ctx, L_atr(df)


def late_mom_atr(name, tf=1, last=30, first=30):
    """Baltussen и др. (2021): r_ROD = от вчерашнего закрытия до (закрытие − last) минут;
    r_ONFH = от вчерашнего закрытия до открытия + first. Сделка: последние last минут в сторону r_ROD.
    Результат и издержки — в долях ATR_d (дневного диапазона), без стопа."""
    df, s, sid, ctx, atr = prep(name, tf)
    c = df.close.to_numpy(); m = ctx["m"]; step = int(ctx["step"])
    mk = L.MARKETS[name]
    sess_len = (int(mk.close[:2]) * 60 + int(mk.close[3:])) - (int(mk.open[:2]) * 60 + int(mk.open[3:]))
    rows = []
    for k in range(len(s)):
        a, b = s.i0.iat[k], s.i1.iat[k]
        ad, pdc = s.atr_d.iat[k], s.pdc.iat[k]
        if not ad > 0 or np.isnan(pdc) or m[b] + step < sess_len - 5:
            continue
        mm = m[a:b + 1]
        j1 = np.flatnonzero(mm + step >= first); j2 = np.flatnonzero(mm + step >= sess_len - last)
        if not len(j1) or not len(j2):
            continue
        i1, i2 = a + j1[0], a + j2[0]
        rod = c[i2] / pdc - 1; onfh = c[i1] / pdc - 1
        d = 1 if rod > 0 else -1
        rows.append(dict(sess=k, d=d, rod=rod, onfh=onfh, agree=np.sign(rod) == np.sign(onfh),
                         pnl_atr=(c[b] - c[i2]) * d / ad, cost_atr=mk.cost_pct * c[i2] / ad,
                         rod_atr=(c[i2] - pdc) / ad))
    ev = pd.DataFrame(rows)
    ev["date"] = pd.to_datetime(s.date.to_numpy()[ev.sess]); ev["oos"] = ev.date.dt.year >= 2023; ev["market"] = name
    return ev


def day_table(name, T=30, tf=1):
    """По сессии: признаки на момент T минут после открытия и исход «остаток дня» в долях ATR_d."""
    df, s, sid, ctx, atr = prep(name, tf)
    o, h, l, c = (df[k].to_numpy() for k in ("open", "high", "low", "close"))
    m = ctx["m"]; step = int(ctx["step"])
    cl = s.close.to_numpy()
    sma20 = pd.Series(cl).rolling(20).mean().shift().to_numpy()
    rng = (s.high - s.low).to_numpy()
    nr4 = pd.Series(rng).rolling(4).apply(lambda x: float(x[-1] == x.min()), raw=True).shift().to_numpy()
    rows = []
    for k in range(len(s)):
        a, b = s.i0.iat[k], s.i1.iat[k]
        ad, pdc = s.atr_d.iat[k], s.pdc.iat[k]
        if not ad > 0 or np.isnan(pdc) or s.bars.iat[k] < 0.8 * s.bars.median():
            continue
        mm = m[a:b + 1]
        j = np.flatnonzero(mm + step >= T)
        if not len(j):
            continue
        iT = a + j[0]
        op = s.open.iat[k]
        orh, orl = h[a:iT + 1].max(), l[a:iT + 1].min()
        # максимальный ход в обе стороны от цены в T до конца сессии
        up = h[iT + 1:b + 1].max() - c[iT] if b > iT else 0.0
        dn = c[iT] - l[iT + 1:b + 1].min() if b > iT else 0.0
        rows.append(dict(
            sess=k, iT=iT, price_T=c[iT],
            r_or=(c[iT] - op) / ad, r_onfh=(c[iT] - pdc) / ad, gap=(op - pdc) / ad,
            or_w=(orh - orl) / ad, or_pos=((c[iT] - orl) / (orh - orl) - 0.5) * 2 if orh > orl else 0.0,
            nr4=nr4[k], nr7=float(s.nr7.iat[k]), trend=(pdc - sma20[k]) / ad,
            open_loc=1.0 if op > s.pdh.iat[k] else -1.0 if op < s.pdl.iat[k] else 0.0,
            on_rng=s.on_range.iat[k], cost=L.MARKETS[name].cost_pct * c[iT] / ad,
            fwd=(c[b] - c[iT]) / ad, mfe_up=up / ad, mfe_dn=dn / ad,
            orh=orh, orl=orl, atr_d=ad))
    t = pd.DataFrame(rows)
    t["date"] = pd.to_datetime(s.date.to_numpy()[t.sess]); t["oos"] = t.date.dt.year >= 2023; t["market"] = name
    return t


MKS = ["SP500", "NASDAQ", "DAX", "GOLD", "SILVER", "WTI", "BRENT"]


def base_rates() -> pd.DataFrame:
    """Базовые частоты сценариев дня на 2018–2022 (диапазон первых 30 минут)."""
    out = []
    for n in MKS:
        df, s, sid, ctx, atr = prep(n, 1)
        h, l = df.high.to_numpy(), df.low.to_numpy()
        s = s[(s.bars >= 0.9 * s.bars.median()) & (pd.to_datetime(s.date).dt.year <= 2022)]
        rec = []
        for _, r in s.iterrows():
            a, b = r.i0, r.i1 + 1
            orh, orl = ctx["or30h"][b - 1], ctx["or30l"][b - 1]
            post = ctx["m"][a:b] >= 30
            up = np.flatnonzero(h[a:b][post] > orh)
            dn = np.flatnonzero(l[a:b][post] < orl)
            fu = up[0] if len(up) else 10**9
            fd = dn[0] if len(dn) else 10**9
            first = 0 if fu == fd == 10**9 else (1 if fu < fd else -1)
            hold = (first == 1 and r.close > orh) or (first == -1 and r.close < orl)
            filled = (r.open > r.pdc and r.low <= r.pdc) or (r.open < r.pdc and r.high >= r.pdc)
            rec.append(dict(first=first, both=len(up) > 0 and len(dn) > 0, hold=hold, gap=abs(r.gap), filled=filled))
        x = pd.DataFrame(rec)
        br = x[x["first"] != 0]
        g = x.dropna(subset=["gap"])
        out.append(dict(рынок=n, сессий=len(x), обе_стороны=round(br.both.mean() * 100), пробой_держится=round(br.hold.mean() * 100),
                        гэп_до_025=round(g[g.gap <= 0.25].filled.mean() * 100), гэп_025_05=round(g[(g.gap > 0.25) & (g.gap <= 0.5)].filled.mean() * 100),
                        гэп_больше_05=round(g[g.gap > 0.5].filled.mean() * 100)))
    return pd.DataFrame(out)


def families() -> pd.DataFrame:
    """Средний R до и после издержек по семействам сценариев, 2018–2022, ТФ 1/5/15 мин."""
    rows = []

    def add(name, ev):
        ev = ev[~ev.oos] if len(ev) else ev
        if len(ev):
            rows.append(dict(семейство=name, сделок=len(ev), R_до=round(ev.r.mean(), 3), R_после=round(ev.r_net.mean(), 3)))

    for tf in (1, 5, 15):
        add(f"пробой диапазона 30 мин, все дни, {tf}м", pd.concat([events_orb(n, tf, W=30) for n in MKS]))
        add(f"первая 5-минутная свеча (Zarattini–Aziz), {tf}м", pd.concat([events_orb_candle(n, tf, W=5) for n in MKS if tf <= 5]) if tf <= 5 else pd.DataFrame())
        for lvl in ("or30", "pd", "on"):
            add(f"разворот после ложного пробоя {lvl}, {tf}м", pd.concat([events_fail(n, tf, level=lvl) for n in MKS]))
        add(f"закрытие гэпа, {tf}м", pd.concat([events_gap(n, tf) for n in MKS]))
        add(f"моментум последних 30 минут со стопом, {tf}м", pd.concat([events_lateMom(n, tf) for n in MKS]))
        add(f"шумовая зона (Zarattini–Barbon–Aziz), {tf}м", pd.concat([events_noise(n, tf) for n in MKS]))
        add(f"пробой ночного диапазона, {tf}м", pd.concat([events_onbreak(n, tf) for n in MKS]))
    return pd.DataFrame(rows)


def late_momentum() -> pd.DataFrame:
    """Baltussen и др.: r_ROD → последние 30 минут, без стопа, в долях дневного диапазона."""
    ev = pd.concat([late_mom_atr(n) for n in MKS])
    ev["net"] = ev.pnl_atr - ev.cost_atr
    g = ev.groupby(["market", "oos"])
    return pd.DataFrame({"дней": g.size(), "до_издержек_ATR": g.pnl_atr.mean().round(4), "после_ATR": g.net.mean().round(4),
                         "t": (g.pnl_atr.mean() / (g.pnl_atr.std() / np.sqrt(g.size()))).round(2)})


def forecast_model(T: int = 30) -> dict:
    """Линейная модель остатка дня через T минут после открытия: обучение 2018–22, проверка 2023–26."""
    dt = pd.concat([day_table(n, T=T) for n in MKS])
    F = ["r_or", "r_onfh", "gap", "or_w", "or_pos", "nr4", "nr7", "trend", "open_loc", "on_rng"]
    X = dt[F].astype(float).fillna(0).clip(-3, 3)
    y = dt.fwd.clip(-3, 3).to_numpy()
    tr = ~dt.oos.to_numpy()
    A = np.c_[np.ones(len(X)), X]
    beta, *_ = np.linalg.lstsq(A[tr], y[tr], rcond=None)
    pred = A @ beta
    return {"корреляция 2018–22": round(float(np.corrcoef(pred[tr], y[tr])[0, 1]), 3),
            "корреляция 2023–26": round(float(np.corrcoef(pred[~tr], y[~tr])[0, 1]), 3)}


if __name__ == "__main__":
    D = Path(sys.argv[1])
    pd.set_option("display.width", 220)
    print("=== Базовые частоты, 2018–2022, % ===")
    print(base_rates().to_string(index=False))
    print("\n=== Семейства сценариев, 2018–2022 ===")
    print(families().to_string(index=False))
    print("\n=== Моментум последних 30 минут (Baltussen и др.) ===")
    print(late_momentum().to_string())
    print("\n=== Модель прогноза остатка дня ===")
    print(forecast_model())
