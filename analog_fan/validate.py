"""Walk-forward проверка веера аналогов против базы «тот же час суток».

Схема: помесячные фолды. Для тестового месяца M база аналогов = все валидные минуты,
чей исход закрылся до начала M; нормировка признаков — по этой же базе. Базовая
модель («климатология часа») — распределение исходов из той же базы для того же
часа суток. Никакие параметры не подбираются по результатам тестовых месяцев.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from . import search as S
from .config import Config
from .features import y_close_cols


def hour_baseline(feat: pd.DataFrame, db_mask: np.ndarray, cfg: Config) -> dict:
    """Таблицы базовой модели по часу суток (только по базе аналогов)."""
    H = cfg.horizon
    d = feat.loc[db_mask, y_close_cols(cfg) + ["touch", "hour"]]
    Y = d[y_close_cols(cfg)].to_numpy(np.float64)
    t = np.nan_to_num(d["touch"].to_numpy(), nan=-1)
    up = (t == 1) + 0.5 * (t == 0.5)
    hr = d["hour"].to_numpy()
    q = np.full((24, H, 5), np.nan)
    qinc = np.full((24, H, 5), np.nan)        # квантили приращения Y_H - Y_k для суженной базы
    pu = np.full(24, np.nan)
    mean = np.full(24, np.nan)
    for h in range(24):
        m = hr == h
        if m.sum() < 100:
            continue
        q[h] = np.quantile(Y[m], S.QLEVELS, axis=0).T
        qinc[h, 1:] = np.quantile(Y[m, -1:] - Y[m, :-1], S.QLEVELS, axis=0).T  # индекс k: приращение от k до H
        pu[h] = up[m].mean()
        mean[h] = Y[m, -1].mean()
    return {"q": q, "qinc": qinc, "p_up": pu, "mean": mean}


def run_walk_forward(feat: pd.DataFrame, blocks: dict, cfg: Config, first_test: str,
                     step: int = 5, k_narrow: int = 5, log=print) -> pd.DataFrame:
    """Возвращает по строке на тестовую минуту: веер модели и базы, исходы."""
    H = cfg.horizon
    mins = S.minutes_of(feat.index)
    valid = feat["valid_y"].to_numpy()
    months = pd.date_range(pd.Timestamp(first_test, tz="UTC"), feat.index[-1], freq="MS")
    Yall = feat[y_close_cols(cfg)].to_numpy(np.float32)
    out = []
    for m0 in months:
        m1 = m0 + pd.offsets.MonthBegin(1)
        t0 = time.time()
        db_mask = valid & (feat.index < m0 - pd.Timedelta(minutes=H))
        idx = S.AnalogIndex(feat, blocks, cfg, db_mask)
        base = hour_baseline(feat, db_mask, cfg)
        q_rows = np.flatnonzero(valid & (feat.index >= m0) & (feat.index < m1))[::step]
        rows, dist, found = idx.search(q_rows, mins[q_rows])
        Y, T = S.gather_paths(feat, rows, cfg)
        W = S.kernel_weights(dist).astype(np.float64)
        Q = S.wquantiles(Y.astype(np.float64), W, S.QLEVELS)            # (nq,H,5)
        pu, pdn = S.touch_prob(T, W)
        sw = W.sum(1)
        exp_move = (W * np.nan_to_num(Y[:, :, -1])).sum(1) / sw
        ess = sw ** 2 / (W ** 2).sum(1)
        R = Yall[q_rows].astype(np.float64)
        # сужение: прошло k_narrow минут, веер на H пересажен на текущую цену
        Wn, alive = S.live_reweight_batch(Y, W, R, k_narrow, cfg)
        Z = R[:, k_narrow - 1:k_narrow, None] + (Y[:, :, -1:] - Y[:, :, k_narrow - 1:k_narrow]).transpose(0, 2, 1)
        Qn = S.wquantiles(Z.transpose(0, 2, 1).astype(np.float64), Wn, S.QLEVELS)[:, 0]   # (nq,5)
        hr = feat["hour"].to_numpy()[q_rows]
        res = pd.DataFrame(index=feat.index[q_rows])
        res["fold"] = m0.strftime("%Y-%m")
        res["hour"] = hr
        res["found"] = found
        res["ess"] = ess
        res["p_up"] = pu
        res["p_dn"] = pdn
        res["exp_move"] = exp_move
        res["base_p_up"] = base["p_up"][hr]
        res["base_mean"] = base["mean"][hr]
        for j in (1, 3, 5, 10, 15):
            for li, lv in enumerate((10, 25, 50, 75, 90)):
                res[f"q{j}_{lv}"] = Q[:, j - 1, li]
                res[f"b{j}_{lv}"] = base["q"][hr, j - 1, li]
            res[f"y{j}"] = R[:, j - 1]
        for li, lv in enumerate((10, 25, 50, 75, 90)):
            res[f"n15_{lv}"] = Qn[:, li]                                   # суженный веер
            res[f"nb15_{lv}"] = R[:, k_narrow - 1] + base["qinc"][hr, k_narrow, li]  # база: r_k + климат. приращение
        res["alive_k"] = alive.sum(1)
        for c in ("touch", "pnl_long", "atr", "close", "v_long", "eff60", "p60"):
            res[c] = feat[c].to_numpy()[q_rows]
        out.append(res)
        log(f"{m0:%Y-%m}: база {idx.index.ntotal:,}, запросов {len(q_rows):,}, {time.time() - t0:.0f} с")
    return pd.concat(out)


# --------------------------------------------------------------------------- метрики
def pinball(y: np.ndarray, q: np.ndarray, levels=S.QLEVELS) -> np.ndarray:
    """Средний квантильный (pinball) лосс по 5 уровням на каждое наблюдение.
    Это собственная оценка качества всего распределения (приближение CRPS)."""
    d = y[:, None] - q
    return np.mean(np.maximum(levels * d, (levels - 1) * d), axis=1)


def block_bootstrap_ci(values: np.ndarray, groups: np.ndarray, stat=np.mean, n: int = 1000, seed: int = 0):
    """95% ДИ бутстрэпом по блокам (дням): соседние минуты сильно зависимы."""
    rng = np.random.default_rng(seed)
    ug, inv = np.unique(groups, return_inverse=True)
    sums = np.bincount(inv, weights=values)
    cnts = np.bincount(inv)
    bs = []
    for _ in range(n):
        s = rng.integers(0, len(ug), len(ug))
        bs.append(sums[s].sum() / cnts[s].sum())
    return np.percentile(bs, [2.5, 97.5])
