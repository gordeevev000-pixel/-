"""Этап 3б: отчёт проверки по data/wf_results.parquet.

python scripts/stage3_report.py [spread_price]
spread_price — спред/издержки на круг в единицах цены (плейсхолдер [X пунктов]).
Пишет reports/stage3_validation.md и графики reports/stage3_*.png.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analog_fan.validate import pinball

SPREAD = float(sys.argv[1]) if len(sys.argv) > 1 else None
r = pd.read_parquet("data/wf_results.parquet")
day = r.index.floor("D").to_numpy()
LV = (10, 25, 50, 75, 90)
rng = np.random.default_rng(0)
udays, dinv = np.unique(day, return_inverse=True)
BOOT = [rng.integers(0, len(udays), len(udays)) for _ in range(500)]


def by_day_sum(x):
    return np.bincount(dinv, weights=x, minlength=len(udays))


def skill_ci(loss_m, loss_b):
    """Навык 1 - Lm/Lb и 95% ДИ бутстрэпом по дням."""
    sm, sb = by_day_sum(loss_m), by_day_sum(loss_b)
    point = 1 - sm.sum() / sb.sum()
    bs = [1 - sm[s].sum() / sb[s].sum() for s in BOOT]
    return point, *np.percentile(bs, [2.5, 97.5])


def mean_ci(x):
    s, c = by_day_sum(x), np.bincount(dinv, minlength=len(udays)).astype(float)
    bs = [s[b].sum() / c[b].sum() for b in BOOT]
    return x.mean(), *np.percentile(bs, [2.5, 97.5])


def fmt_ci(p, lo, hi, pct=True):
    d = 2 if max(abs(lo), abs(hi)) < 0.005 else 1
    return f"{p:+.{d}%} [{lo:+.{d}%}; {hi:+.{d}%}]" if pct else f"{p:+.3f} [{lo:+.3f}; {hi:+.3f}]"


def cover(y, q, lo, hi):
    return ((y >= q[f"{lo}"]) & (y <= q[f"{hi}"])).mean()


L = ["# Этап 3. Проверка walk-forward\n",
     f"* Тестовые месяцы: {r['fold'].iloc[0]} … {r['fold'].iloc[-1]} ({r['fold'].nunique()} фолдов), "
     f"запросов: {len(r):,} (каждая 5-я минута), дней: {len(udays)}",
     "* Для месяца M база аналогов и нормировка — только минуты, чей исход закрылся до начала M. "
     "Параметры метода заданы заранее и по тестовым исходам не подбирались.",
     "* База сравнения: распределение исходов того же часа суток из той же базы (без поиска аналогов).",
     "* Доверительные интервалы 95% — бутстрэп по дням (минуты внутри дня зависимы).", ""]

# ------------------------------------------------------------------ 1. калибровка полос
L += ["## 1. Калибровка полос веера\n",
      "Доля случаев, когда реальная цена на горизонте h оказалась внутри полосы (номинал 50% и 80%).", "",
      "| горизонт | веер 50% | база 50% | веер 80% | база 80% | ширина 80% веер / база |",
      "|---|---:|---:|---:|---:|---:|"]
cov_rows = []
for j in (1, 3, 5, 10, 15):
    y = r[f"y{j}"]
    qm = {str(l): r[f"q{j}_{l}"] for l in LV}
    qb = {str(l): r[f"b{j}_{l}"] for l in LV}
    c = (cover(y, qm, 25, 75), cover(y, qb, 25, 75), cover(y, qm, 10, 90), cover(y, qb, 10, 90))
    wr = ((r[f"q{j}_90"] - r[f"q{j}_10"]) / (r[f"b{j}_90"] - r[f"b{j}_10"])).median()
    cov_rows.append((j, *c))
    L.append(f"| {j} мин | {c[0]:.1%} | {c[1]:.1%} | {c[2]:.1%} | {c[3]:.1%} | {wr:.2f} |")

# ------------------------------------------------------------------ 2. качество распределения
L += ["", "## 2. Веер против базы: квантильный (pinball) лосс\n",
      "Собственная оценка всего распределения (усреднение по уровням 10/25/50/75/90). "
      "Навык = 1 − лосс веера / лосс базы; > 0 — веер лучше.", "",
      "| горизонт | лосс веера | лосс базы | навык [95% ДИ] |", "|---|---:|---:|---:|"]
skills = {}
for j in (1, 5, 10, 15):
    y = r[f"y{j}"].to_numpy()
    lm = pinball(y, r[[f"q{j}_{l}" for l in LV]].to_numpy())
    lb = pinball(y, r[[f"b{j}_{l}" for l in LV]].to_numpy())
    skills[j] = skill_ci(lm, lb)
    L.append(f"| {j} мин | {lm.mean():.4f} | {lb.mean():.4f} | {fmt_ci(*skills[j])} |")
# навык по фолдам на h=15
y = r["y15"].to_numpy()
lm15 = pinball(y, r[[f"q15_{l}" for l in LV]].to_numpy())
lb15 = pinball(y, r[[f"b15_{l}" for l in LV]].to_numpy())
fold_skill = pd.Series(lm15).groupby(r["fold"].to_numpy()).sum() / pd.Series(lb15).groupby(r["fold"].to_numpy()).sum()
fold_skill = 1 - fold_skill
L.append(f"\nНавык на 15 мин по месяцам: лучше базы в {(fold_skill > 0).sum()} из {len(fold_skill)} месяцев "
         f"(мин {fold_skill.min():+.1%}, макс {fold_skill.max():+.1%}).")

# отдельно: центр распределения (медиана) — есть ли информация о направлении
y15 = r["y15"].to_numpy()
mae_m = np.abs(y15 - r["q15_50"].to_numpy())
mae_b = np.abs(y15 - r["b15_50"].to_numpy())
ic = pd.Series(r["exp_move"].to_numpy()).corr(pd.Series(y15), method="spearman")
ic_fold = r.groupby("fold").apply(lambda g: g["exp_move"].corr(g["y15"], method="spearman"))
L += ["", "Направление:",
      f"* Ошибка медианы (MAE) на 15 мин: навык {fmt_ci(*skill_ci(mae_m, mae_b))}",
      f"* Ранговая корреляция «ожидаемый ход» ↔ реальный ход на 15 мин (IC): {ic:+.4f}; по месяцам: "
      f"среднее {ic_fold.mean():+.4f}, t = {ic_fold.mean() / ic_fold.std(ddof=1) * np.sqrt(len(ic_fold)):+.2f}, "
      f"положительна в {(ic_fold > 0).sum()} из {len(ic_fold)} месяцев"]

# ------------------------------------------------------------------ 3. вероятность барьера
t = r["touch"].to_numpy()
u = np.where(t == 1, 1.0, np.where(t == 0.5, 0.5, 0.0))        # «ни один» = не вверх
p = r["p_up"].to_numpy()
pb = r["base_p_up"].to_numpy()
brier_m, brier_b, brier_c = (p - u) ** 2, (pb - u) ** 2, (0.5 - u) ** 2
L += ["", "## 3. Вероятность «+1 ATR раньше −1 ATR»\n",
      f"* Разброс прогнозов: 5–95% квантили p = [{np.quantile(p, .05):.3f}; {np.quantile(p, .95):.3f}], σ = {p.std():.3f}",
      f"* Brier: веер {brier_m.mean():.4f}, база часа {brier_b.mean():.4f}, константа 0.5 {brier_c.mean():.4f}",
      f"* Навык Brier против базы часа: {fmt_ci(*skill_ci(brier_m, brier_b))}; против 0.5: {fmt_ci(*skill_ci(brier_m, brier_c))}",
      "", "Надёжность (децили прогноза):", "", "| p прогноз (ср.) | факт | число |", "|---:|---:|---:|"]
bins = pd.qcut(p, 10, labels=False, duplicates="drop")
rel = pd.DataFrame({"p": p, "u": u, "b": bins}).groupby("b").agg(p=("p", "mean"), u=("u", "mean"), n=("u", "size"))
for _, row in rel.iterrows():
    L.append(f"| {row.p:.3f} | {row.u:.3f} | {int(row.n):,} |")
top, bot = rel.iloc[-1], rel.iloc[0]
L.append(f"\nВерхний дециль прогноза: ожидали {top.p:.1%}, получили {top.u:.1%}; нижний: ожидали {bot.p:.1%}, получили {bot.u:.1%}.")

# ------------------------------------------------------------------ 3б. честная калибровка
# Сжатие к базе: p_cal = p_base + λ (p − p_base), λ — МНК по всем ПРОШЛЫМ месяцам (с 4-го фолда).
# То же для ожидаемого хода: ход_cal = β · ход, β — МНК по прошлым месяцам.
folds = sorted(r["fold"].unique())
fa = r["fold"].to_numpy()
em = r["exp_move"].to_numpy()
pc = np.full(len(r), np.nan)
ec = np.full(len(r), np.nan)
lam_hist, beta_hist = [], []
for i, m in enumerate(folds):
    if i < 3:
        continue
    past = np.isin(fa, folds[:i])
    x, yy = p[past] - pb[past], u[past] - pb[past]
    lam = max(0.0, float((x * yy).sum() / (x * x).sum()))
    beta = max(0.0, float((em[past] * y15[past]).sum() / (em[past] ** 2).sum()))
    cur = fa == m
    pc[cur] = pb[cur] + lam * (p[cur] - pb[cur])
    ec[cur] = beta * em[cur]
    lam_hist.append(lam)
    beta_hist.append(beta)
okc = np.isfinite(pc)
bc = (pc - u) ** 2
L += ["", "### 3б. После калибровки по прошлым месяцам", "",
      f"Сжатие к базе: p = p_часа + λ·(p_аналогов − p_часа), λ подбирается МНК только по прошлым месяцам. "
      f"λ по фолдам: {min(lam_hist):.2f}…{max(lam_hist):.2f} (последний {lam_hist[-1]:.2f}) — "
      f"то есть в «сырой» вероятности аналогов реального сигнала примерно {lam_hist[-1]:.0%}, остальное шум.", "",
      f"* Brier на фолдах с калибровкой ({folds[3]}…): откалиброванная {bc[okc].mean():.5f}, база часа {brier_b[okc].mean():.5f}, "
      f"0.5 {brier_c[okc].mean():.5f}; навык против базы {fmt_ci(*skill_ci(np.where(okc, bc, 0), np.where(okc, brier_b, 0)))}",
      f"* Разброс откалиброванной вероятности: 5–95% = [{np.nanquantile(pc, .05):.3f}; {np.nanquantile(pc, .95):.3f}]",
      f"* Ожидаемый ход: коэффициент β = {beta_hist[-1]:.2f} (сырой «ожидаемый ход» надо умножать на него); "
      f"после калибровки MSE {np.mean((ec[okc] - y15[okc]) ** 2):.3f} против {np.mean((r['base_mean'].to_numpy()[okc] - y15[okc]) ** 2):.3f} у среднего часа"]
# для интерфейса: λ и β по всем месяцам, кроме последнего (окно перемотки лежит в последнем)
past = np.isin(fa, folds[:-1])
x, yy = p[past] - pb[past], u[past] - pb[past]
LAM_UI = max(0.0, float((x * yy).sum() / (x * x).sum()))
BETA_UI = max(0.0, float((em[past] * y15[past]).sum() / (em[past] ** 2).sum()))

# ------------------------------------------------------------------ 4. живое сужение
yk = r["y15"].to_numpy()
qn = {str(l): r[f"n15_{l}"] for l in LV}
qnb = {str(l): r[f"nb15_{l}"] for l in LV}
ln = pinball(yk, r[[f"n15_{l}" for l in LV]].to_numpy())
lnb = pinball(yk, r[[f"nb15_{l}" for l in LV]].to_numpy())
L += ["", "## 4. Живое сужение (прошло 5 минут после якоря, прогноз цены на +15)\n",
      "Суженный веер = текущая цена + дальнейшие приращения аналогов, перевзвешенных по совпадению первых 5 минут. "
      "База = текущая цена + обычное 10-минутное приращение для этого часа.", "",
      f"* Живых аналогов через 5 мин: медиана {np.median(r['alive_k']):.0f} из 300 (10–90%: "
      f"{np.quantile(r['alive_k'], .1):.0f}–{np.quantile(r['alive_k'], .9):.0f})",
      f"* Покрытие 50%: сужение {cover(yk, qn, 25, 75):.1%}, база {cover(yk, qnb, 25, 75):.1%}; "
      f"80%: сужение {cover(yk, qn, 10, 90):.1%}, база {cover(yk, qnb, 10, 90):.1%}",
      f"* Pinball-лосс: сужение {ln.mean():.4f}, база {lnb.mean():.4f}, навык {fmt_ci(*skill_ci(ln, lnb))}",
      f"* Для сравнения, веер без сужения на +15 (из якоря): лосс {lm15.mean():.4f} — сужение уменьшает лосс на "
      f"{1 - ln.mean() / lm15.mean():.0%} просто потому, что 5 минут пути уже известны"]

# ------------------------------------------------------------------ 5. торговля и издержки
pnl_l = r["pnl_long"].to_numpy()
pnl_s = np.where(t == 0.5, -1.0, -pnl_l)
atr_px = r["atr"].to_numpy()
close = r["close"].to_numpy()
L += ["", "## 5. Есть ли перекос, переживающий издержки\n",
      "Правило: если P(+1 раньше −1) >= θ — лонг, если P(−1 раньше +1) >= θ — шорт; вход по закрытию минуты, "
      "выход на барьере ±1 ATR или через 15 мин. Если оба барьера в одной минуте — считаем стоп. "
      "Результат в ATR на сделку; издержки = спред на круг / ATR этой минуты.", ""]


def trades(theta, mask=None):
    lo = p >= theta
    sh = (r["p_dn"].to_numpy() >= theta) & ~lo
    g = np.where(lo, pnl_l, np.where(sh, pnl_s, np.nan))
    if mask is not None:
        g = np.where(mask, g, np.nan)
    return g


def stat_line(g, cost):
    m = np.isfinite(g)
    if m.sum() < 30:
        return None
    net = g[m] - cost[m]
    dsum = np.bincount(dinv[m], weights=net, minlength=len(udays))
    dcnt = np.bincount(dinv[m], minlength=len(udays))
    dm = dsum[dcnt > 0] / dcnt[dcnt > 0]
    tstat = dm.mean() / dm.std(ddof=1) * np.sqrt(len(dm))
    return m.sum(), (g[m] > 0).mean(), g[m].mean(), net.mean(), tstat


cost_grid = {"0": np.zeros(len(r)), "0.05 ATR": np.full(len(r), .05), "0.10 ATR": np.full(len(r), .10),
             "0.20 ATR": np.full(len(r), .20)}
if SPREAD is not None:
    cost_grid[f"спред {SPREAD:g}"] = SPREAD / atr_px
else:  # для замещающих данных BTC: реалистичные комиссии спота
    cost_grid["0.02% (maker)"] = 0.0002 * close / atr_px
    cost_grid["0.2% (taker)"] = 0.002 * close / atr_px
L.append("Медианные издержки в ATR: " + ", ".join(f"{k} = {np.median(v):.2f}" for k, v in cost_grid.items() if np.median(v) > 0))
L += ["", "| θ | сделок | доля минут | выигрыш | валовый ATR/сделку | " + " | ".join(f"нетто {k}" for k in cost_grid) + " |",
      "|---:|---:|---:|---:|---:|" + "---:|" * len(cost_grid)]
for th in (0.52, 0.55, 0.58, 0.60, 0.65):
    g = trades(th)
    cells = []
    base_stat = None
    for k, c in cost_grid.items():
        s = stat_line(g, c)
        if s is None:
            cells.append("—")
            continue
        base_stat = base_stat or s
        cells.append(f"{s[3]:+.3f} (t={s[4]:+.1f})")
    if base_stat is None:
        L.append(f"| {th:.2f} | <30 | | | | " + " | ".join(cells) + " |")
        continue
    L.append(f"| {th:.2f} | {base_stat[0]:,} | {base_stat[0] / len(r):.1%} | {base_stat[1]:.1%} | {base_stat[2]:+.3f} | "
             + " | ".join(cells) + " |")

# честный вложенный выбор порога: выбираем θ на первой половине месяцев, проверяем на второй
first = r["fold"].isin(folds[: len(folds) // 2]).to_numpy()
cost_sel = cost_grid["0.10 ATR"]
best = nested = None
for th in np.arange(0.51, 0.70, 0.01):
    s = stat_line(trades(th, first), cost_sel)
    if s and s[0] >= 200 and (best is None or s[3] > best[1]):
        best = (th, s[3])
L.append("")
if best:
    s2 = nested = stat_line(trades(best[0], ~first), cost_sel)
    L.append(f"Вложенная проверка (издержки 0.10 ATR): лучший θ на {folds[0]}…{folds[len(folds) // 2 - 1]} = {best[0]:.2f} "
             f"(нетто {best[1]:+.3f} ATR/сделку в выборке подбора); на {folds[len(folds) // 2]}…{folds[-1]}: "
             + (f"{s2[0]:,} сделок, нетто **{s2[3]:+.3f} ATR/сделку**, t = {s2[4]:+.1f}" if s2 else "сделок < 30"))

# режимы: есть ли устойчивый перекос в каком-то режиме (с поправкой на множественность — через две половины)
vq = np.quantile(r["v_long"], [1 / 3, 2 / 3])
eq = np.quantile(r["eff60"], [1 / 3, 2 / 3])
vol_lbl = np.where(r["v_long"] < vq[0], "низк.вол", np.where(r["v_long"] > vq[1], "выс.вол", "ср.вол"))
tr_lbl = np.where(r["eff60"] > eq[1], "тренд", np.where(r["eff60"] < eq[0], "флэт", "смеш."))
L += ["", "По режимам (θ = 0.58, издержки 0.10 ATR) — перекос считается устойчивым, только если он есть в обеих половинах:", "",
      "| режим | сделок 1-я пол. | нетто 1-я | сделок 2-я пол. | нетто 2-я |", "|---|---:|---:|---:|---:|"]
g = trades(0.58)
for v in ("низк.вол", "ср.вол", "выс.вол"):
    for tr in ("флэт", "смеш.", "тренд"):
        m = (vol_lbl == v) & (tr_lbl == tr)
        s1 = stat_line(np.where(m & first, g, np.nan), cost_sel)
        s2 = stat_line(np.where(m & ~first, g, np.nan), cost_sel)
        f = lambda s: (f"{s[0]:,}", f"{s[3]:+.3f}") if s else ("<30", "—")
        L.append(f"| {tr} · {v} | {' | '.join(f(s1))} | {' | '.join(f(s2))} |")

# ------------------------------------------------------------------ графики
fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
ax[0].plot([0.3, 0.7], [0.3, 0.7], color="grey", lw=.8, ls="--")
ax[0].plot(rel.p, rel.u, "o-", label="веер аналогов")
relb = pd.DataFrame({"p": pb, "u": u}).groupby(pd.qcut(pb, 10, labels=False, duplicates="drop")).mean()
ax[0].plot(relb.p, relb.u, "s-", alpha=.7, label="база: час суток")
ax[0].set_xlabel("прогноз P(+1 ATR раньше −1)"); ax[0].set_ylabel("факт"); ax[0].legend()
ax[0].set_title("Надёжность вероятности барьера")
hs = [c[0] for c in cov_rows]
ax[1].axhline(.5, color="grey", lw=.6, ls=":"); ax[1].axhline(.8, color="grey", lw=.6, ls=":")
ax[1].plot(hs, [c[1] for c in cov_rows], "o-", color="C0", label="веер 50%")
ax[1].plot(hs, [c[2] for c in cov_rows], "o--", color="C1", label="база 50%")
ax[1].plot(hs, [c[3] for c in cov_rows], "s-", color="C0", label="веер 80%")
ax[1].plot(hs, [c[4] for c in cov_rows], "s--", color="C1", label="база 80%")
ax[1].set_xlabel("горизонт, мин"); ax[1].set_title("Покрытие полос"); ax[1].legend(fontsize=8); ax[1].set_ylim(.3, .95)
ax[2].bar(range(len(fold_skill)), fold_skill.to_numpy() * 100, color=np.where(fold_skill > 0, "C2", "C3"))
ax[2].set_xticks(range(len(fold_skill)), [f[2:] for f in fold_skill.index], rotation=90, fontsize=7)
ax[2].axhline(0, color="k", lw=.6); ax[2].set_title("Навык pinball на 15 мин по месяцам, %")
fig.tight_layout(); fig.savefig("reports/stage3_validation.png", dpi=110)

L += ["", "![](stage3_validation.png)"]
c15 = [c for c in cov_rows if c[0] == 15][0]
summary = {"period": f"{folds[0]} … {folds[-1]}", "n_queries": int(len(r)),
           "pinball_skill_15": [round(float(x), 4) for x in skills[15]],
           "pinball_skill_5": [round(float(x), 4) for x in skills[5]],
           "cover50_15": [round(c15[1], 4), round(c15[2], 4)], "cover80_15": [round(c15[3], 4), round(c15[4], 4)],
           "brier_skill_vs_hour": [round(float(x), 4) for x in skill_ci(brier_m, brier_b)],
           "ic_15": round(float(ic), 4), "months_better": int((fold_skill > 0).sum()), "months": int(len(fold_skill)),
           "narrow_skill": [round(float(x), 4) for x in skill_ci(ln, lnb)],
           "calib_lambda": round(LAM_UI, 4), "calib_beta": round(BETA_UI, 4),
           "brier_cal_skill": [round(float(x), 4) for x in skill_ci(np.where(okc, bc, 0), np.where(okc, brier_b, 0))],
           "top_decile": [round(float(top.p), 4), round(float(top.u), 4)],
           "bottom_decile": [round(float(bot.p), 4), round(float(bot.u), 4)],
           "nested": ({"theta": round(float(best[0]), 2), "trades": int(nested[0]), "net_atr": round(float(nested[3]), 4),
                       "t": round(float(nested[4]), 2)} if nested else None)}
import json
Path("reports/stage3_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1))
Path("reports/stage3_validation.md").write_text("\n".join(L) + "\n")
print("\n".join(L))
