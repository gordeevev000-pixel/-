"""Проверка индикатора pine/SCALP_4.pine на истории.

Что делает:
  1. Повторяет логику индикатора бар за баром (те же формулы, тот же порядок:
     сброс дня -> закрытие сигнала -> условие 4 -> новый сигнал).
  2. Прогоняет её на минутках фьючерсов (Yahoo, ~30 дней) и печатает статистику.
  3. Тест на перерисовку: история обрезается в сотнях точек (в том числе ровно
     на свече сигнала и в середине 5m бара), сигналы до точки обрезки должны
     совпасть с полным прогоном. Контрольная версия с заглядыванием (EMA 5m
     текущего, ещё не закрытого бара) обязана этот тест провалить.

Запуск:  pip install yfinance pandas numpy matplotlib
         python scripts/scalp4_check.py            # скачает данные в data/
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

# Инструмент: тикер Yahoo, шаг цены, типичный спред в тиках
INSTR = {
    "GC": ("GC=F", 0.10, 1.0),      # золото
    "CL": ("CL=F", 0.01, 1.0),      # нефть WTI
    "NG": ("NG=F", 0.001, 1.0),     # природный газ
    "6E": ("6E=F", 0.00005, 1.0),   # евро (аналог EURUSD)
}


@dataclass
class P:
    htf_min: int = 5
    htf_len: int = 50
    atr_len: int = 14
    atr_avg_len: int = 50
    spread_ticks: float = 1.0
    spread_pct: float = 15.0
    tz: str = "Europe/Moscow"
    sessions: tuple = ((600, 780), (990, 1140))   # 10:00–13:00, 16:30–19:00
    news: tuple = ()                              # ("17:00", ...)
    news_win: int = 5
    ema_len: int = 20
    pb_len: int = 5
    vol_len: int = 20
    vol_k: float = 1.2
    stop_buf: float = 0.1
    max_stop_k: float = 1.0
    rr: float = 1.5
    max_losses: int = 3
    day_loss_r: float = -3.0
    lookahead_bug: bool = False   # только для контрольного теста


# ─── индикаторы в семантике Pine ────────────────────────────────────────────
def sma(x, n):
    return pd.Series(x).rolling(n, min_periods=n).mean().to_numpy()


def _ew(x, n, alpha):
    # старт — SMA первых n значений, дальше s = alpha * x + (1 - alpha) * s[1]
    out = np.full(len(x), np.nan)
    if len(x) < n:
        return out
    seed = x[:n].mean()
    out[n - 1:] = pd.Series(np.r_[seed, x[n:]]).ewm(alpha=alpha, adjust=False).mean().to_numpy()
    return out


def ema(x, n):
    return _ew(np.asarray(x, float), n, 2.0 / (n + 1))


def rma(x, n):
    return _ew(np.asarray(x, float), n, 1.0 / n)


def atr(h, l, c, n):
    pc = np.roll(c, 1)
    pc[0] = np.nan
    tr = np.nanmax(np.vstack([h - l, np.abs(h - pc), np.abs(l - pc)]), axis=0)
    tr[0] = h[0] - l[0]
    return rma(tr, n)


def shift(x, k):
    out = np.full(len(x), np.nan)
    if k < len(x):
        out[k:] = x[:len(x) - k]
    return out


def indicators(df: pd.DataFrame, p: P) -> dict:
    o, h, l, c = (df[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    v = np.nan_to_num(df["volume"].to_numpy(float))
    t = df.index
    n = len(df)

    # VWAP: торговый день CME начинается в 18:00 по Нью-Йорку
    ny = t.tz_convert("America/New_York")
    sess_day = (ny + pd.Timedelta(hours=6)).date
    hlc3 = (h + l + c) / 3
    g = pd.Series(sess_day)
    cpv = pd.Series(hlc3 * v).groupby(g).cumsum().to_numpy()
    cv = pd.Series(v).groupby(g).cumsum().to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        vwap = np.where(cv > 0, cpv / cv, np.nan)

    # EMA 50 со старшего ТФ: request.security(lookahead_off) + фиксация [1]
    # на первом 1m баре нового 5m бара
    per = t.floor(f"{p.htf_min}min")
    c5 = pd.Series(c, index=t).groupby(per).last()
    e5 = ema(c5.to_numpy(), p.htf_len)
    end5 = c5.index + pd.Timedelta(minutes=p.htf_min)
    if p.lookahead_bug:
        # ОШИБОЧНО: EMA текущего 5m бара, который ещё не закрыт
        pos = np.searchsorted(c5.index, per, side="right") - 1
        ema_htf = e5[pos]
    else:
        # на истории lookahead_off отдаёт EMA последнего 5m бара, закрытого к close этой 1m свечи
        pos = np.searchsorted(end5, t + pd.Timedelta(minutes=1), side="right") - 1
        live = np.where(pos >= 0, e5[np.clip(pos, 0, None)], np.nan)
        change = np.r_[True, per[1:] != per[:-1]]
        ema_htf = np.full(n, np.nan)
        cur = np.nan
        for i in range(n):
            if change[i] and i > 0:
                cur = live[i - 1]
            ema_htf[i] = cur

    with np.errstate(invalid="ignore"):
        dir_ = np.where((c > vwap) & (c > ema_htf), 1, np.where((c < vwap) & (c < ema_htf), -1, 0))

    a = atr(h, l, c, p.atr_len)
    a_avg = sma(a, p.atr_avg_len)
    loc = t.tz_convert(p.tz)
    mins = np.asarray(loc.hour * 60 + loc.minute)
    sess = np.zeros(n, bool)
    for s0, s1 in p.sessions:
        sess |= (mins >= s0) & (mins < s1)
    news = np.zeros(n, bool)
    for hhmm in p.news:
        hh, mm = map(int, hhmm.split(":"))
        T = hh * 60 + mm
        for k in (-1, 0, 1):
            tt = T + 1440 * k
            news |= (mins < tt + p.news_win) & (mins + 1 > tt - p.news_win)
    with np.errstate(invalid="ignore"):
        atr_ok = a > a_avg
        spread_ok = (a > 0) & (p.spread_ticks * TICK_NOW[0] <= a * p.spread_pct / 100)
    c2 = atr_ok & sess & ~news & spread_ok

    e20 = ema(c, p.ema_len)
    v_avg = sma(v, p.vol_len)
    with np.errstate(invalid="ignore"):
        vol_ok = v > v_avg * p.vol_k
    red_high = np.full(n, np.nan)
    green_low = np.full(n, np.nan)
    touch_l = np.zeros(n, bool)
    touch_s = np.zeros(n, bool)
    with np.errstate(invalid="ignore"):
        for k in range(p.pb_len, 0, -1):   # от дальних к ближним: побеждает последняя свеча
            ok_, ck, hk, lk = shift(o, k), shift(c, k), shift(h, k), shift(l, k)
            ek, vk = shift(e20, k), shift(vwap, k)
            red = ck < ok_
            green = ck > ok_
            red_high = np.where(red, hk, red_high)
            green_low = np.where(green, lk, green_low)
            touch_l |= (lk <= ek) | (lk <= vk)
            touch_s |= (hk >= ek) | (hk >= vk)
        c_prev = shift(c, 1)
        break_l = ~np.isnan(red_high) & (c > red_high) & (c_prev <= red_high)
        break_s = ~np.isnan(green_low) & (c < green_low) & (c_prev >= green_low)
    trig_l = touch_l & break_l & vol_ok
    trig_s = touch_s & break_s & vol_ok
    c3 = np.where(dir_ == 1, trig_l, np.where(dir_ == -1, trig_s, False))

    w = p.pb_len + 1
    pb_low = pd.Series(l).rolling(w, min_periods=w).min().to_numpy()
    pb_high = pd.Series(h).rolling(w, min_periods=w).max().to_numpy()
    risk_l = np.minimum(c - (pb_low - p.stop_buf * a), p.max_stop_k * a)
    risk_s = np.minimum(pb_high + p.stop_buf * a - c, p.max_stop_k * a)
    risk = np.where(dir_ == 1, risk_l, np.where(dir_ == -1, risk_s, np.nan))
    with np.errstate(invalid="ignore"):
        risk_ok = ~np.isnan(risk) & (risk >= TICK_NOW[0])

    day = np.asarray(loc.date)
    return dict(o=o, h=h, l=l, c=c, t=t, dir=dir_, c1=dir_ != 0, c2=c2, c3=c3, risk=risk,
                risk_ok=risk_ok, atr=a, day=day, sess=sess, atr_ok=atr_ok, spread_ok=spread_ok,
                touch_l=touch_l, touch_s=touch_s, break_l=break_l, break_s=break_s, vol_ok=vol_ok,
                ema_htf=ema_htf, vwap=vwap, e20=e20)


TICK_NOW = [0.01]   # шаг цены текущего инструмента (syminfo.mintick)


def run(df: pd.DataFrame, p: P, tick: float) -> tuple[list[dict], dict]:
    TICK_NOW[0] = tick
    x = indicators(df, p)
    n = len(df)
    h, l, c = x["h"], x["l"], x["c"]
    trades: list[dict] = []
    in_trade, tr = False, None
    loss_row, day_r, blocked = 0, 0.0, False
    blocked_days = set()
    for i in range(n):
        if i == 0 or x["day"][i] != x["day"][i - 1]:
            loss_row, day_r, blocked = 0, 0.0, False
        if in_trade and i > tr["bar"]:
            d = tr["dir"]
            hit_sl = l[i] <= tr["stop"] if d == 1 else h[i] >= tr["stop"]
            hit_tp = h[i] >= tr["take"] if d == 1 else l[i] <= tr["take"]
            if hit_sl or hit_tp:
                res = -1.0 if hit_sl else p.rr
                tr.update(exit_bar=i, exit_time=x["t"][i], res=res)
                day_r += res
                loss_row = loss_row + 1 if hit_sl else 0
                in_trade = False
                if loss_row >= p.max_losses or day_r <= p.day_loss_r:
                    blocked = True
                    blocked_days.add(x["day"][i])
        c4 = (not blocked) and (not in_trade) and x["risk_ok"][i]
        if x["c1"][i] and x["c2"][i] and x["c3"][i] and c4:
            d = int(x["dir"][i])
            entry = c[i]
            stop = np.floor((entry - d * x["risk"][i]) / tick + 0.5) * tick   # math.round_to_mintick: половина — вверх
            take = np.floor((entry + d * abs(entry - stop) * p.rr) / tick + 0.5) * tick
            tr = dict(bar=i, time=x["t"][i], dir=d, entry=entry, stop=stop, take=take,
                      risk=abs(entry - stop), risk_atr=abs(entry - stop) / x["atr"][i], res=np.nan)
            trades.append(tr)
            in_trade = True
    x["blocked_days"] = blocked_days
    return trades, x


# ─── данные ─────────────────────────────────────────────────────────────────
def load(code: str) -> pd.DataFrame:
    f = DATA / f"{code}_1m.csv"
    if not f.exists():
        import yfinance as yf
        end = pd.Timestamp.now("UTC").normalize() + pd.Timedelta(days=1)
        frames = []
        for k in range(5):
            e = end - pd.Timedelta(days=7 * k)
            b = max(e - pd.Timedelta(days=7), end - pd.Timedelta(days=29))
            if b >= e:
                continue
            d = yf.download(INSTR[code][0], start=b.strftime("%Y-%m-%d"), end=e.strftime("%Y-%m-%d"),
                            interval="1m", progress=False, auto_adjust=False)
            if len(d):
                d.columns = [cc[0].lower() for cc in d.columns]
                frames.append(d[["open", "high", "low", "close", "volume"]])
        d = pd.concat(frames).sort_index()
        d = d[~d.index.duplicated()]
        d.index = d.index.tz_convert("UTC")
        DATA.mkdir(exist_ok=True)
        d.to_csv(f)
    d = pd.read_csv(f, index_col=0)
    d.index = pd.to_datetime(d.index, utc=True)
    d = d.dropna(subset=["open", "high", "low", "close"])
    # последний бар Yahoo часто недоформирован (объём 0) — отбрасываем
    return d.iloc[:-1]


# ─── отчёты ─────────────────────────────────────────────────────────────────
def stats(trades, spread_px):
    closed = [t for t in trades if not np.isnan(t["res"])]
    r = np.array([t["res"] for t in closed])
    cost = np.array([spread_px / t["risk"] for t in closed]) if closed else np.array([])
    wins = int((r > 0).sum())
    streak = mx = 0
    for v in r:
        streak = streak + 1 if v < 0 else 0
        mx = max(mx, streak)
    gp, gl = r[r > 0].sum(), -r[r < 0].sum()
    return dict(n=len(trades), closed=len(closed), wins=wins, losses=len(closed) - wins,
                wr=wins / len(closed) * 100 if closed else np.nan,
                gross=r.sum(), net=(r - cost).sum(), avg_cost=cost.mean() if len(cost) else np.nan,
                pf=gp / gl if gl > 0 else np.nan, max_ls=mx,
                risk_ticks=np.median([t["risk"] for t in trades]) / 1 if trades else np.nan)


def repaint_test(df, p, tick, trades_full, x_full, n_cuts=60, seed=1):
    """История обрезается в разных точках. Всё, что посчитано до точки обрезки
    (EMA 5m, VWAP, условия 1–4, стоп, сигналы), должно совпасть с полным прогоном."""
    rng = np.random.default_rng(seed)
    cuts = set()
    for t in trades_full[:25]:
        cuts.update({t["bar"] + 1, t["bar"] + 2, t["bar"] + 3})   # история кончается на свече сигнала
    mids = np.where((df.index.minute % 5) != 4)[0] + 1              # последний бар — середина 5m бара
    cuts.update(rng.choice(mids[mids > 2000], size=min(n_cuts, len(mids)), replace=False).tolist())
    key = lambda t: (t["bar"], t["dir"], round(t["entry"], 8), round(t["stop"], 8), round(t["take"], 8))
    full = [key(t) for t in trades_full]
    arrays = ("ema_htf", "vwap", "e20", "atr", "risk", "dir", "c2", "c3", "risk_ok")
    bad_sig = bad_val = 0
    for cut in sorted(cuts):
        if cut >= len(df):
            continue
        tr, x = run(df.iloc[:cut], p, tick)
        if [key(t) for t in tr] != [k for k in full if k[0] < cut]:
            bad_sig += 1
        for a in arrays:
            u, v = np.asarray(x[a], float), np.asarray(x_full[a][:cut], float)
            if not np.allclose(u, v, rtol=0, atol=1e-9, equal_nan=True):
                bad_val += 1
                break
    return len(cuts), bad_sig, bad_val


def main():
    base = P()
    rows = []
    all_trades = {}
    print("=" * 100)
    print("ПРОГОН НА ИСТОРИИ (настройки по умолчанию, спред 1 тик)")
    print("=" * 100)
    for code, (_, tick, spr) in INSTR.items():
        df = load(code)
        p = replace(base, spread_ticks=spr)
        trades, x = run(df, p, tick)
        all_trades[code] = (df, trades, x, tick, p)
        s = stats(trades, spr * tick)
        in_s = x["sess"]
        nb = in_s.sum()
        funnel = dict(
            c1=(x["c1"] & in_s).sum() / nb * 100, atr=(x["atr_ok"] & in_s).sum() / nb * 100,
            spread=(x["spread_ok"] & in_s).sum() / nb * 100, c2=(x["c2"]).sum() / nb * 100,
            c3=(x["c3"] & in_s).sum() / nb * 100, c123=(x["c1"] & x["c2"] & x["c3"]).sum() / nb * 100)
        days = len(set(x["day"][in_s]))
        rows.append((code, df, s, funnel, days, x, tick, trades))
        print(f"\n{code}: {len(df):,} баров 1m, {df.index[0]:%Y-%m-%d} … {df.index[-1]:%Y-%m-%d}, торговых дней в сессиях: {days}")
        print(f"  баров в сессиях: {nb:,}. Доля баров сессии, где выполнено: "
              f"усл.1 {funnel['c1']:.0f}% · ATR>SMA {funnel['atr']:.0f}% · спред {funnel['spread']:.0f}% · "
              f"усл.2 {funnel['c2']:.0f}% · усл.3 {funnel['c3']:.1f}% · 1+2+3 {funnel['c123']:.2f}%")
        if s["n"]:
            med_ticks = np.median([t["risk"] / tick for t in trades])
            med_atr = np.median([t["risk_atr"] for t in trades])
            print(f"  сигналов {s['n']} (закрыто {s['closed']}): тейков {s['wins']}, стопов {s['losses']}, "
                  f"винрейт {s['wr']:.0f}% (безубыток при 1.5R = 40%)")
            print(f"  итог {s['gross']:+.1f}R до спреда, {s['net']:+.1f}R после спреда · PF {s['pf']:.2f} · "
                  f"макс. серия стопов {s['max_ls']} · дней с блокировкой {len(x['blocked_days'])}")
            print(f"  стоп: медиана {med_ticks:.0f} тиков = {med_atr:.2f} ATR · спред съедает в среднем {s['avg_cost']:.2f}R на сделку")
        else:
            print("  сигналов нет")

    print("\n" + "=" * 100)
    print("ТЕСТ НА ПЕРЕРИСОВКУ И ЗАГЛЯДЫВАНИЕ В БУДУЩЕЕ")
    print("=" * 100)
    for code, (df, trades, x, tick, p) in all_trades.items():
        n, bs, bv = repaint_test(df, p, tick, trades, x)
        pb = replace(p, lookahead_bug=True)
        trb, xb = run(df, pb, tick)
        nb_, bsb, bvb = repaint_test(df, pb, tick, trb, xb)
        print(f"{code}: индикатор — {n} обрезок: сигналы изменились {bs}, значения изменились {bv}  |  "
              f"контроль с заглядыванием — {nb_} обрезок: сигналы {bsb}, значения {bvb}")
    return rows, all_trades


if __name__ == "__main__":
    main()
