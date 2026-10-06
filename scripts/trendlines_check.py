"""Проверка индикатора pine/AUTO_TRENDLINES.pine.

Что делает:
  1. Повторяет логику индикатора бар за баром: те же пивоты, тот же поиск
     линии при подтверждении пивота, те же касания, пробои, ретесты,
     удаление старых линий и средняя линия канала.
  2. Рисует итог в PNG, как он выглядел бы на графике TradingView:
     reports/trendlines_preview.png.
  3. Тест на перерисовку: история обрезается в N точках, и всё, что было
     нарисовано и просигналено до точки обрезки, должно совпасть с полным
     прогоном.

Данные: минутки BTCUSDT из web/data.js (3120 баров) или любой CSV
time,open,high,low,close[,volume].

Запуск:  pip install numpy matplotlib
         python scripts/trendlines_check.py               # данные из web/data.js
         python scripts/trendlines_check.py data/x.csv    # свой CSV
         python scripts/trendlines_check.py --piner /tmp/piner   # + сверка с Pine-кодом,
                                       # исполненным движком piner (см. trendlines_piner.cjs)
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


@dataclass
class P:
    bodies: bool = False        # линии по телам, а не по теням
    atr_len: int = 50
    tol_atr: float = 0.25       # допуск касания, × ATR
    break_atr: float = 0.10     # пробой: закрытие за линией дальше, × ATR
    min_touch: int = 3
    max_slope: float = 0.25     # макс. наклон, × ATR на бар
    keep_broken: int = 240
    retest_bars: int = 30
    par_tol: float = 0.30       # насколько наклоны могут отличаться у канала
    # главные (розовые / зелёные)
    maj_on: bool = True
    maj_left: int = 15
    maj_right: int = 8
    maj_lb: int = 600
    maj_max: int = 2            # линий на сторону
    # локальные (чёрные)
    loc_on: bool = True
    loc_left: int = 5
    loc_right: int = 3
    loc_lb: int = 200
    loc_max: int = 1
    loc_trend_only: bool = True
    # встречные главные линии (↑ по хаям, ↓ по лоям): "channel" — только как
    # вторая граница канала / клина к живой трендовой линии с другой стороны
    counter: str = "channel"    # "channel" | "always" | "never"


@dataclass
class TL:
    uid: int
    side: int      # 1 — по хаям (сверху), -1 — по лоям (снизу)
    tier: int      # 1 — главная, 0 — локальная
    x1: int
    y1: float
    x2: int
    y2: float
    slope: float
    born: int      # бар, на котором линия появилась
    touches: int
    last_touch: int
    broken_at: int | None = None
    retested: bool = False
    deleted_at: int | None = None

    def value(self, x: float) -> float:
        return self.y1 + self.slope * (x - self.x1)


# ─── индикаторы в семантике Pine ────────────────────────────────────────────
def atr(h, l, c, n):
    tr = np.empty_like(c)
    tr[0] = h[0] - l[0]
    pc = c[:-1]
    tr[1:] = np.maximum.reduce([h[1:] - l[1:], np.abs(h[1:] - pc), np.abs(l[1:] - pc)])
    out = np.full_like(c, np.nan)
    if len(c) < n:
        return out
    out[n - 1] = tr[:n].mean()
    a = 1.0 / n
    for i in range(n, len(c)):
        out[i] = out[i - 1] + a * (tr[i] - out[i - 1])
    return out


def pivot(src, left, right, high=True):
    """Свинг индикатора: значение на баре i - right, известное на баре i.
    Выше всех баров слева и не ниже баров справа (из равных вершин — первая)."""
    n = len(src)
    out = np.full(n, np.nan)
    for i in range(left + right, n):
        p = i - right
        v = src[p]
        lw = src[p - left:p]
        rw = src[p + 1:i + 1]
        if high:
            ok = v > lw.max() and v >= rw.max()
        else:
            ok = v < lw.min() and v <= rw.min()
        if ok:
            out[i] = v
    return out


# ─── сам индикатор ──────────────────────────────────────────────────────────
@dataclass
class State:
    lines: list = field(default_factory=list)       # все когда-либо созданные (для отрисовки)
    events: list = field(default_factory=list)      # (bar, kind, uid)
    mids: list = field(default_factory=list)        # (born, dead, x0, y0, slope, dirn)


def run(o, h, l, c, p: P, upto: int | None = None) -> State:
    n = len(c) if upto is None else upto
    hi = np.maximum(o, c) if p.bodies else h
    lo = np.minimum(o, c) if p.bodies else l
    a = atr(h, l, c, p.atr_len)
    tiers = []
    if p.maj_on:
        tiers.append((1, p.maj_left, p.maj_right, p.maj_lb, False,
                      pivot(hi, p.maj_left, p.maj_right, True), pivot(lo, p.maj_left, p.maj_right, False)))
    if p.loc_on:
        tiers.append((0, p.loc_left, p.loc_right, p.loc_lb, p.loc_trend_only,
                      pivot(hi, p.loc_left, p.loc_right, True), pivot(lo, p.loc_left, p.loc_right, False)))
    piv = {(t[0], s): [] for t in tiers for s in (1, -1)}
    st = State()
    alive: list[TL] = []
    uid = 0
    mid = None   # (u.uid, l.uid, idx в st.mids)

    for i in range(n):
        if math.isnan(a[i]):
            continue
        tol = p.tol_atr * a[i]

        # 1. касания, пробои, ретесты, удаление
        for t in list(alive):
            v = t.value(i)
            if t.broken_at is None:
                if t.side == 1 and c[i] > v + p.break_atr * a[i]:
                    t.broken_at = i
                    st.events.append((i, "break_up", t.uid))
                elif t.side == -1 and c[i] < v - p.break_atr * a[i]:
                    t.broken_at = i
                    st.events.append((i, "break_dn", t.uid))
                else:
                    touch = hi[i] >= v - tol if t.side == 1 else lo[i] <= v + tol
                    if touch:
                        gap = p.maj_left if t.tier == 1 else p.loc_left
                        if i - t.last_touch > gap:
                            t.touches += 1
                        t.last_touch = i
            else:
                age = i - t.broken_at
                if not t.retested and 0 < age <= p.retest_bars:
                    if t.side == 1 and lo[i] <= v + tol and c[i] > v:
                        t.retested = True
                        st.events.append((i, "retest_long", t.uid))
                    elif t.side == -1 and hi[i] >= v - tol and c[i] < v:
                        t.retested = True
                        st.events.append((i, "retest_short", t.uid))
                if age >= p.keep_broken:
                    t.deleted_at = i
                    alive.remove(t)

        # 2. новые пивоты → поиск линии
        for tier, left, right, lb, trend_only, ph, pl in tiers:
            for side, arr in ((1, ph), (-1, pl)):
                if math.isnan(arr[i]):
                    continue
                px, py = i - right, arr[i]
                store = piv[(tier, side)]
                explained = any(
                    t.side == side and t.broken_at is None and abs(py - t.value(px)) <= tol
                    and (t.tier == tier or t.tier == 1)
                    for t in alive)
                best = None
                only_trend = trend_only
                if not explained and tier == 1:
                    if p.counter == "always":
                        only_trend = False
                    elif p.counter == "never":
                        only_trend = True
                    else:
                        only_trend = not any(
                            t.tier == 1 and t.side == -side and t.broken_at is None
                            and (t.slope > 0 if side == 1 else t.slope < 0) for t in alive)
                if not explained:
                    for cx, cy in store:
                        span = px - cx
                        if span < left or span > lb:
                            continue
                        s = (py - cy) / span
                        if abs(s) > p.max_slope * a[i]:
                            continue
                        if only_trend and (s >= 0 if side == 1 else s <= 0):
                            continue
                        touches, last, bad = 0, -10**9, False
                        for b in range(cx, i + 1):
                            v = cy + s * (b - cx)
                            if side == 1:
                                if hi[b] > v + tol or c[b] > v:
                                    bad = True
                                    break
                                tch = hi[b] >= v - tol
                            else:
                                if lo[b] < v - tol or c[b] < v:
                                    bad = True
                                    break
                                tch = lo[b] <= v + tol
                            if tch:
                                if b - last > left:
                                    touches += 1
                                last = b
                        if bad or touches < p.min_touch:
                            continue
                        if best is None or touches > best[2] or (touches == best[2] and cx < best[0]):
                            best = (cx, cy, touches, s, last)
                if best is not None:
                    cx, cy, touches, s, last = best
                    def dup(t):
                        # совпадает с живой линией на всём своём отрезке
                        return t.broken_at is None and abs(t.value(cx) - cy) <= 2 * tol \
                            and abs(t.value(i) - (cy + s * (i - cx))) <= 2 * tol
                    same_anchor = [t for t in alive if t.side == side and t.tier == tier and t.x1 == cx]
                    skip = any(t.side == side and t.tier >= tier and t not in same_anchor and dup(t) for t in alive)
                    if not skip:
                        for t in list(alive):
                            if t.side == side and (t in same_anchor or (t.tier < tier and dup(t))):
                                t.deleted_at = i
                                alive.remove(t)
                        mine = [t for t in alive if t.side == side and t.tier == tier]
                        while len(mine) >= (p.maj_max if tier == 1 else p.loc_max):
                            # уходит линия, к которой цена дольше всего не подходила
                            victim = min(mine, key=lambda t: max(t.last_touch, t.broken_at or -1))
                            victim.deleted_at = i
                            alive.remove(victim)
                            mine.remove(victim)
                        uid += 1
                        t = TL(uid, side, tier, cx, cy, px, py, s, i, touches, last)
                        alive.append(t)
                        st.lines.append(t)
                        st.events.append((i, "new", uid))
                store.append((px, py))
                while store and store[0][0] < i - lb - right:
                    store.pop(0)

        # 3. средняя линия канала по последним главным линиям сверху и снизу
        ups = [t for t in alive if t.tier == 1 and t.side == 1]
        dns = [t for t in alive if t.tier == 1 and t.side == -1]
        u = ups[-1] if ups else None
        d = dns[-1] if dns else None
        want = None
        if u and d and u.slope != 0 and d.slope != 0 and (u.slope > 0) == (d.slope > 0) \
                and abs(u.slope - d.slope) <= p.par_tol * max(abs(u.slope), abs(d.slope)):
            want = (u.uid, d.uid)
        cur = mid[:2] if mid else None
        if want != cur:
            if mid:
                st.mids[mid[2]][1] = i
            mid = None
            if want:
                x0 = max(u.x1, d.x1)
                y0 = (u.value(x0) + d.value(x0)) / 2
                st.mids.append([i, None, x0, y0, (u.slope + d.slope) / 2, 1 if u.slope > 0 else -1])
                mid = (u.uid, d.uid, len(st.mids) - 1)
    return st


# ─── данные ─────────────────────────────────────────────────────────────────
def load(path: str | None):
    if path:
        rows = list(csv.DictReader(open(path)))
        t = np.array([r["time"] for r in rows])
        o, h, l, c = (np.array([float(r[k]) for r in rows]) for k in ("open", "high", "low", "close"))
        return t, o, h, l, c, Path(path).stem
    s = (ROOT / "web" / "data.js").read_text()
    d = json.loads(s[s.index("{"):].rstrip().rstrip(";"))["candles"]
    t = np.array(d["t"], dtype="datetime64[s]")
    return t, *(np.array(d[k], float) for k in ("o", "h", "l", "c")), "BTCUSDT 1m"


# ─── картинка ───────────────────────────────────────────────────────────────
PINK, GREEN, BLACK = "#F0347A", "#4CAF50", "#000000"


def color(t: TL):
    if t.tier == 0:
        return BLACK
    return PINK if t.slope < 0 or (t.slope == 0 and t.side == 1) else GREEN


def draw_panel(ax, t, o, h, l, c, st: State, a0: int, a1: int, title: str):
    """Как график выглядит на баре a1 - 1: только живые на этот момент линии."""
    bg = "#131722"
    ax.set_facecolor(bg)
    xs = np.arange(a0, a1)
    up = c[a0:a1] >= o[a0:a1]
    col = np.where(up, "#26a69a", "#ef5350")
    ax.vlines(xs, l[a0:a1], h[a0:a1], color=col, lw=0.6)
    ax.vlines(xs, np.minimum(o, c)[a0:a1], np.maximum(o, c)[a0:a1], color=col, lw=2.0)
    lo_y, hi_y = l[a0:a1].min(), h[a0:a1].max()
    pad = (hi_y - lo_y) * 0.06
    xe = a1 + 15
    for tl in st.lines:
        if tl.born >= a1 or (tl.deleted_at is not None and tl.deleted_at < a1):
            continue
        xb = max(tl.x1, a0)
        ax.plot([xb, xe], [tl.value(xb), tl.value(xe)], color=color(tl),
                lw=2.0 if tl.tier == 1 else 1.6, alpha=0.95, zorder=3)
    for born, dead, x0, y0, s, dirn in st.mids:
        if born >= a1 or (dead is not None and dead < a1):
            continue
        xb = max(x0, a0)
        ax.plot([xb, xe], [y0 + s * (xb - x0), y0 + s * (xe - x0)], "--",
                color=GREEN if dirn > 0 else PINK, lw=1.2, zorder=3)
    for b, kind, _ in st.events:
        if not (a0 <= b < a1):
            continue
        if kind == "break_up":
            ax.plot(b, l[b] - pad * 0.3, "^", color="#26a69a", ms=6, zorder=5)
        elif kind == "break_dn":
            ax.plot(b, h[b] + pad * 0.3, "v", color="#ef5350", ms=6, zorder=5)
        elif kind == "retest_long":
            ax.plot(b, l[b] - pad * 0.3, "o", color="#26a69a", ms=4, mfc="none", zorder=5)
        elif kind == "retest_short":
            ax.plot(b, h[b] + pad * 0.3, "o", color="#ef5350", ms=4, mfc="none", zorder=5)
    ax.set_xlim(a0, xe)
    ax.set_ylim(lo_y - pad, hi_y + pad)
    ticks = list(range(a0 - a0 % 60 + 60, a1, 60))
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(t[k])[11:16] for k in ticks], color="#b2b5be", fontsize=7)
    ax.tick_params(axis="y", colors="#b2b5be", labelsize=7)
    ax.yaxis.tick_right()
    for sp in ax.spines.values():
        sp.set_color("#2a2e39")
    ax.grid(color="#1e222d", lw=0.6)
    ax.set_title(title, color="#d1d4dc", fontsize=9, loc="left")


def draw(t, o, h, l, c, p: P, ends: list, width: int, name: str, out: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = (len(ends) + 1) // 2
    fig, axs = plt.subplots(rows, 2, figsize=(20, 6.2 * rows), dpi=100)
    fig.patch.set_facecolor("#131722")
    for ax, a1 in zip(axs.flat, ends):
        st = run(o, h, l, c, p, upto=a1)
        a0 = max(0, a1 - width)
        draw_panel(ax, t, o, h, l, c, st, a0, a1, f"{name} · график на {str(t[a1 - 1])[:16]}")
    for ax in list(axs.flat)[len(ends):]:
        ax.set_visible(False)
    fig.tight_layout()
    fig.savefig(out, facecolor="#131722")
    plt.close(fig)


# ─── тест на перерисовку ────────────────────────────────────────────────────
def snapshot(st: State, cut: int):
    """Что было на графике и в сигналах к бару cut - 1 включительно."""
    lines = sorted((tl.uid, tl.side, tl.tier, tl.x1, round(tl.y1, 8), tl.x2, round(tl.y2, 8),
                    tl.broken_at if tl.broken_at is not None and tl.broken_at < cut else None,
                    tl.deleted_at if tl.deleted_at is not None and tl.deleted_at < cut else None)
                   for tl in st.lines if tl.born < cut)
    ev = sorted(e for e in st.events if e[0] < cut)
    return lines, ev


PINER_TITLES = {
    "Пробой наклонки вверх": "break_up",
    "Пробой наклонки вниз": "break_dn",
    "Ретест сверху (лонг)": "retest_long",
    "Ретест снизу (шорт)": "retest_short",
}


def piner_check(o, h, l, c, p: P, piner_dir: str, ends: list, inputs: dict | None = None):
    """Сверка с индикатором, исполненным независимым движком Pine v6 (piner).
    inputs — те же настройки, что в p, но в виде названий полей индикатора."""
    import subprocess
    import tempfile

    ok = True
    for n in ends:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out.json"
            subprocess.run(["node", str(ROOT / "scripts" / "trendlines_piner.cjs"),
                            str(ROOT / "pine" / "AUTO_TRENDLINES.pine"), str(ROOT / "web" / "data.js"),
                            str(out), str(n)], check=True,
                           env={"PINER_DIR": piner_dir, "PINER_INPUTS": json.dumps(inputs or {}),
                                "PATH": "/usr/bin:/bin:/usr/local/bin"})
            r = json.loads(out.read_text())
        st = run(o, h, l, c, p, upto=n)
        ev_py = {(b, k) for b, k, _ in st.events if k != "new"}
        ev_pn = {(b, PINER_TITLES[tt]) for tt, bars in r["markers"].items() for b in bars}
        ln_py = {("solid", x.x1, round(x.y1, 6), x.x2, round(x.y2, 6)) for x in st.lines if x.deleted_at is None}
        ln_py |= {("dashed", x0, round(y0, 6)) for born, dead, x0, y0, s_, d_ in st.mids if dead is None}
        ln_pn = set()
        for q in r["lines"]:
            if q.get("style") == "dashed":
                ln_pn.add(("dashed", q["x1"], round(q["y1"], 6)))
            else:
                ln_pn.add(("solid", q["x1"], round(q["y1"], 6), q["x2"], round(q["y2"], 6)))
        same = ev_py == ev_pn and ln_py == ln_pn
        ok &= same
        print(f"  {n} баров: сигналов {len(ev_pn)} / {len(ev_py)}, линий на графике {len(ln_pn)} / {len(ln_py)} "
              f"(piner / python) — {'совпадает' if same else 'РАСХОЖДЕНИЕ'}")
    return ok


def main():
    ap = argparse.ArgumentParser(description="Проверка pine/AUTO_TRENDLINES.pine")
    ap.add_argument("csv", nargs="?", help="CSV time,open,high,low,close; по умолчанию web/data.js")
    ap.add_argument("--fast", action="store_true", help="без теста на перерисовку")
    ap.add_argument("--piner", metavar="DIR", help="папка с установленным @heyphat/piner: сверить с Pine-кодом")
    args = ap.parse_args()
    t, o, h, l, c, name = load(args.csv)
    p = P()
    st = run(o, h, l, c, p)
    n = len(c)
    kinds = {}
    for _, k, _ in st.events:
        kinds[k] = kinds.get(k, 0) + 1
    maj = sum(1 for x in st.lines if x.tier == 1)
    print(f"{name}: {n} баров, линий {len(st.lines)} (главных {maj}, локальных {len(st.lines) - maj}), "
          f"каналов {len(st.mids)}")
    print("события:", kinds)
    print("касаний у линии: в среднем %.1f, максимум %d" % (np.mean([x.touches for x in st.lines]),
                                                          max(x.touches for x in st.lines)))

    if not args.fast:
        rng = np.random.default_rng(7)
        cuts = sorted(set(rng.integers(200, n, 150).tolist()) | {e[0] + 1 for e in st.events[:60]})
        bad = sum(snapshot(run(o, h, l, c, p, upto=cut), cut) != snapshot(st, cut) for cut in cuts)
        print(f"перерисовка: обрезок {len(cuts)}, расхождений {bad}")

    ends = [n * k // 6 for k in range(1, 7)]
    ends[-1] = n
    if args.piner:
        if args.csv:
            print("сверка с piner работает только на web/data.js")
        else:
            print("сверка с piner:")
            piner_check(o, h, l, c, p, args.piner, ends)

    out = ROOT / "reports"
    out.mkdir(exist_ok=True)
    draw(t, o, h, l, c, p, ends, 480, name, out / "trendlines_preview.png")
    print("картинка:", out / "trendlines_preview.png")


if __name__ == "__main__":
    main()
