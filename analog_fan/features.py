"""Отпечаток минуты и будущие исходы.

Все признаки в момент t считаются только по барам <= t (закрытие минуты t).
Исходы (y) — по барам t+1..t+H и используются только как «ответы» аналогов,
никогда как признаки.

Отпечаток (блоки):
  path  — форма пути: (C[t-lag] - C[t]) / ATR[t] для 30 лагов за 60 минут
          (каждая минута за последние 15, дальше шаг 3 минуты)
  vol   — log(ATR15/ATR60) и log(ATR60/ATR «средней» за ~5 суток)
  tod   — время суток: sin/cos минуты дня по времени рынка
  dayhl — расстояние до хая и лоя текущего торгового дня в ATR (log1p)
  eff   — эффективность |C[t]-C[t-n]| / сумма |ΔC| за n = 15, 30, 60
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Config

BLOCKS = ("path", "vol", "tod", "dayhl", "eff")


def _roll_mean(x: np.ndarray, n: int) -> np.ndarray:
    cs = np.concatenate([[0.0], np.cumsum(x, dtype=np.float64)])
    out = np.full(len(x), np.nan)
    out[n - 1:] = (cs[n:] - cs[:-n]) / n
    return out


def _lag(x: np.ndarray, k: int) -> np.ndarray:
    """x[t-k] (k>0) или x[t+|k|] (k<0), с NaN на краях."""
    out = np.full(len(x), np.nan)
    if k > 0:
        out[k:] = x[:-k]
    else:
        out[:k] = x[-k:]
    return out


def build(df: pd.DataFrame, cfg: Config) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """Возвращает таблицу: признаки X_*, исходы y_*, служебные колонки; и карту блоков."""
    c = df["close"].to_numpy(np.float64)
    h = df["high"].to_numpy(np.float64)
    lo = df["low"].to_numpy(np.float64)
    seg = df["seg"].to_numpy()
    n = len(c)

    prev_c = _lag(c, 1)
    tr = np.fmax(h - lo, np.fmax(np.abs(h - prev_c), np.abs(lo - prev_c)))
    tr[0] = h[0] - lo[0]
    atr_raw = _roll_mean(tr, cfg.atr_n)
    atr_s = _roll_mean(tr, cfg.atr_short)
    atr_l = _roll_mean(tr, cfg.atr_long)
    # Пол ATR: в мёртвые минуты диапазон бара ~ 1 тик, ATR -> 0 и любые ходы в «ATR»
    # раздуваются до десятков. Единица измерения не опускается ниже доли средней.
    atr = np.fmax(atr_raw, cfg.atr_floor * atr_l)

    out: dict[str, np.ndarray] = {}
    blocks: dict[str, list[str]] = {b: [] for b in BLOCKS}

    def put(block, name, v):
        out[name] = v.astype(np.float32)
        blocks[block].append(name)

    # --- форма пути
    for k in cfg.lags:
        put("path", f"p{k:02d}", np.clip((_lag(c, k) - c) / atr, -20, 20))

    # --- волатильность
    with np.errstate(divide="ignore", invalid="ignore"):
        put("vol", "v_short", np.clip(np.log(atr_s / atr_raw), -3, 3))
        put("vol", "v_long", np.clip(np.log(atr_raw / atr_l), -3, 3))

    # --- время суток (по времени рынка)
    loc = df.index.tz_convert(cfg.market_tz)
    mod = (loc.hour * 60 + loc.minute).to_numpy()
    ang = 2 * np.pi * mod / 1440
    put("tod", "tod_sin", np.sin(ang))
    put("tod", "tod_cos", np.cos(ang))

    # --- хай/лой торгового дня (включая текущую минуту)
    day = (loc - pd.Timedelta(hours=cfg.day_start_hour)).normalize()
    day_key = pd.factorize(day)[0]
    dhi = pd.Series(h).groupby(day_key).cummax().to_numpy()
    dlo = pd.Series(lo).groupby(day_key).cummin().to_numpy()
    put("dayhl", "d_hi", np.log1p(np.clip((dhi - c) / atr, 0, 50)))
    put("dayhl", "d_lo", np.log1p(np.clip((c - dlo) / atr, 0, 50)))

    # --- эффективность движения
    step = np.abs(np.diff(c, prepend=c[0]))
    cs = np.concatenate([[0.0], np.cumsum(step)])
    for m in cfg.eff_n:
        path_len = np.full(n, np.nan)
        path_len[m:] = cs[m + 1:] - cs[1:-m]
        with np.errstate(divide="ignore", invalid="ignore"):
            e = np.abs(c - _lag(c, m)) / path_len
        put("eff", f"eff{m}", np.where(path_len > 0, e, 0.0))

    # --- исходы: путь закрытий и экстремумов на H минут вперёд, в ATR момента t
    H = cfg.horizon
    for j in range(1, H + 1):
        out[f"yc{j:02d}"] = ((_lag(c, -j) - c) / atr).astype(np.float32)
        out[f"yh{j:02d}"] = ((_lag(h, -j) - c) / atr).astype(np.float32)
        out[f"yl{j:02d}"] = ((_lag(lo, -j) - c) / atr).astype(np.float32)

    res = pd.DataFrame(out, index=df.index)
    res["close"] = c
    res["atr"] = atr.astype(np.float32)
    res["atr_floored"] = atr > atr_raw
    res["seg"] = seg
    res["hour"] = loc.hour.to_numpy().astype(np.int8)
    res["filled"] = df["filled"].to_numpy()

    # --- валидность: окно прошлого и горизонт внутри одного сегмента, прогрет ATR
    maxlag = max(max(cfg.lags), max(cfg.eff_n))
    ok_past = (_lag(seg.astype(float), maxlag) == seg) & np.isfinite(atr_l) & (atr_raw > 0)
    ok_fut = _lag(seg.astype(float), -H) == seg
    res["valid_x"] = ok_past
    res["valid_y"] = ok_past & ok_fut

    touch, pnl_long = barrier_outcomes(res, cfg)
    res["touch"] = touch            # 1: +1 ATR первым; 0: −1 ATR первым; 0.5: оба в одной минуте; NaN: ни один
    res["pnl_long"] = pnl_long      # лонг с выходом по барьеру/времени, в ATR, без издержек
    return res, blocks


def barrier_outcomes(res: pd.DataFrame, cfg: Config) -> tuple[np.ndarray, np.ndarray]:
    """Кто первым: +b ATR или −b ATR в течение H минут (по high/low баров)."""
    H, b = cfg.horizon, cfg.barrier_atr
    yh = res[[f"yh{j:02d}" for j in range(1, H + 1)]].to_numpy()
    yl = res[[f"yl{j:02d}" for j in range(1, H + 1)]].to_numpy()
    yc = res[f"yc{H:02d}"].to_numpy()
    up = yh >= b
    dn = yl <= -b
    first_up = np.where(up.any(1), up.argmax(1), H + 1)
    first_dn = np.where(dn.any(1), dn.argmax(1), H + 1)
    touch = np.full(len(res), np.nan)
    touch[first_up < first_dn] = 1.0
    touch[first_dn < first_up] = 0.0
    touch[(first_up == first_dn) & (first_up <= H)] = 0.5
    # P&L лонга: выход на барьере; если оба барьера в одной минуте — считаем худшее (стоп)
    pnl = np.where(first_up < first_dn, b, np.where(first_dn <= first_up, -b, 0.0))
    pnl = np.where((first_up > H) & (first_dn > H), yc, pnl)
    return touch.astype(np.float32), pnl.astype(np.float32)


def y_close_cols(cfg: Config) -> list[str]:
    return [f"yc{j:02d}" for j in range(1, cfg.horizon + 1)]
