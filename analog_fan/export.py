"""Экспорт окна истории для веб-интерфейса (режим перемотки).

Для каждой минуты окна поиск идёт «как вживую»: база — вся история, но в
аналоги попадают только минуты, чей 15-минутный исход закрылся ДО текущей минуты
(фильтр времени в search._dedup). Нормировка признаков — по данным до начала окна.

Что кладём на минуту (компактно, base64):
  fan    int16 (15,5)     свежий веер, 0.01 ATR
  narrow int8  (105,5)    суженный веер для якоря в этой минуте: для k=1..14 оставшиеся
                          минуты h=k+1..15, как приращение от реальной цены в k, 0.1 ATR
  alive  uint16 (14,)     число живых аналогов для k=1..14
  an_*                    5 ближайших аналогов: время, расстояние, путь −60..+15 мин
                          (до якоря точка раз в 2 минуты, после — каждая минута)
  метрики float32 (10)    p_up, p_dn, exp_move, width_ratio, ess, base_p_up, found, regime,
                          p_up и exp_move после калибровки из walk-forward
"""
from __future__ import annotations

import base64

import numpy as np
import pandas as pd

from . import search as S
from .config import Config
from .features import y_close_cols
from .validate import hour_baseline



def _b64(a: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()


def export_window(feat: pd.DataFrame, clean: pd.DataFrame, blocks: dict, cfg: Config,
                  start: pd.Timestamp, end: pd.Timestamp, context_min: int = 240,
                  instrument: str = "", validation: dict | None = None) -> dict:
    H = cfg.horizon
    mins = S.minutes_of(feat.index)
    valid_y = feat["valid_y"].to_numpy()
    # база: все минуты с известным исходом; фильтр «только прошлое» — в поиске
    db_mask = valid_y & (feat.index < end)
    # нормировку берём по истории до окна, чтобы окно не влияло само на себя
    pre = valid_y & (feat.index < start)
    idx = S.AnalogIndex(feat, blocks, cfg, db_mask, norm_mask=pre)
    base = hour_baseline(feat, pre, cfg)

    win = (feat.index >= start) & (feat.index < end) & feat["valid_x"].to_numpy()
    q_rows = np.flatnonzero(win)
    rows, dist, found = idx.search(q_rows, mins[q_rows])
    Y, T = S.gather_paths(feat, rows, cfg)
    W = S.kernel_weights(dist).astype(np.float64)
    Q = S.wquantiles(Y.astype(np.float64), W, S.QLEVELS)                 # (n,15,5)
    pu, pdn = S.touch_prob(T, W)
    sw = W.sum(1)
    exp_move = (W * np.nan_to_num(Y[:, :, -1])).sum(1) / sw
    ess = sw ** 2 / (W ** 2).sum(1)
    hr = feat["hour"].to_numpy()[q_rows]
    # калибровка из walk-forward (коэффициенты подобраны на месяцах до окна)
    v = validation or {}
    lam, beta = v.get("calib_lambda"), v.get("calib_beta")
    p_cal = base["p_up"][hr] + lam * (pu - base["p_up"][hr]) if lam is not None else pu
    exp_cal = beta * exp_move if beta is not None else exp_move
    width_ratio = (Q[:, -1, 4] - Q[:, -1, 0]) / (base["q"][hr, -1, 4] - base["q"][hr, -1, 0])

    # суженные веера для каждого якоря и k=1..14 (реальный путь известен — это история)
    R = feat[y_close_cols(cfg)].to_numpy(np.float64)[q_rows]
    n = len(q_rows)
    narrow = np.zeros((n, H * (H - 1) // 2, 5), np.int8)
    alive = np.zeros((n, H - 1), np.uint16)
    Yd = Y.astype(np.float64)
    pos = 0
    for k in range(1, H):
        Rk = np.nan_to_num(R)
        Wn, al = S.live_reweight_batch(Yd, W, Rk, k, cfg)
        Z = Yd[:, :, k:] - Yd[:, :, k - 1:k]                               # приращения после k
        qk = S.wquantiles(Z, Wn, S.QLEVELS)                                 # (n,H-k,5)
        m = H - k
        narrow[:, pos:pos + m] = np.clip(np.round(np.nan_to_num(qk) * 10), -127, 127)
        alive[:, k - 1] = al.sum(1)
        pos += m
    known = np.isfinite(R).all(1)             # у последних 15 минут окна будущее неизвестно

    # 5 ближайших аналогов: путь −60..+15 в ATR их якоря
    c = feat["close"].to_numpy()
    atr = feat["atr"].to_numpy()
    top = rows[:, :5]
    offs = np.r_[np.arange(-60, 0, 2), np.arange(0, H + 1)]          # прошлое через 2 мин, будущее по минутам
    an_path = np.zeros((n, 5, len(offs)), np.int16)
    for j in range(5):
        r0 = top[:, j]
        ii = np.clip(r0[:, None] + offs[None, :], 0, len(c) - 1)
        an_path[:, j] = np.clip(np.round((c[ii] - c[r0][:, None]) / atr[r0][:, None] * 100), -32000, 32000)
    an_time = (mins[top] * 60).astype(np.int64)

    # режим рынка по терцилям истории до окна
    vq = tuple(np.quantile(feat.loc[pre, "v_long"], [1 / 3, 2 / 3]))
    eq = tuple(np.quantile(feat.loc[pre, "eff60"], [1 / 3, 2 / 3]))
    fv = feat.iloc[q_rows]
    regimes = [S.regime_label(a, b, p, vq, eq) for a, b, p in zip(fv["v_long"], fv["eff60"], fv["p60"])]
    reg_names = sorted(set(regimes))
    reg_id = np.array([reg_names.index(x) for x in regimes], np.uint8)

    # свечи: окно + контекст до него
    cs = clean.loc[start - pd.Timedelta(minutes=context_min): end - pd.Timedelta(minutes=1)]
    ohlc = cs[["open", "high", "low", "close"]].to_numpy(np.float64)
    tick = float(np.nanmin(np.diff(np.unique(np.round(cs["close"].to_numpy(), 8))))) if len(cs) > 2 else 0.01
    dec = next(d for d in range(9) if abs(tick * 10 ** d - round(tick * 10 ** d)) < 1e-4) if tick > 0 else 2
    ftime = (S.minutes_of(feat.index[q_rows]) * 60).astype(np.int64)
    ctime = (S.minutes_of(cs.index) * 60).astype(np.int64)

    return {
        "meta": {"instrument": instrument, "horizon": H, "k": cfg.k, "barrier": cfg.barrier_atr,
                 "episode_gap": cfg.episode_gap, "alive_rmse": cfg.alive_rmse, "decimals": dec,
                 "levels": [0.1, 0.25, 0.5, 0.75, 0.9], "regimes": reg_names,
                 "an_offsets": offs.tolist(),
                 "start": str(start), "end": str(end), "validation": validation or {}},
        "candles": {"t": ctime.tolist(), "o": np.round(ohlc[:, 0], dec).tolist(), "h": np.round(ohlc[:, 1], dec).tolist(),
                    "l": np.round(ohlc[:, 2], dec).tolist(), "c": np.round(ohlc[:, 3], dec).tolist()},
        "n": n,
        "t": ftime.tolist(),
        "atr": np.round(atr[q_rows], dec + 2).tolist(),
        "known": known.astype(int).tolist(),
        "fan": _b64(np.clip(np.round(np.nan_to_num(Q) * 100), -32000, 32000).astype(np.int16)),
        "narrow": _b64(narrow),
        "alive": _b64(alive),
        "metrics": _b64(np.stack([pu, pdn, exp_move, width_ratio, ess, base["p_up"][hr], found,
                                  reg_id.astype(float), p_cal, exp_cal], 1).astype(np.float32)),
        "an_time": _b64(an_time.astype(np.float64)),
        "an_dist": _b64(dist[:, :5].astype(np.float32)),
        "an_path": _b64(an_path),
    }
