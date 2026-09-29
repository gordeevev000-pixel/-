"""Поиск аналогов, веер и живое сужение.

Ключевые правила честности:
* Индекс строится только из минут, чей исход (t+H) полностью известен до
  момента запроса. В walk-forward это все минуты до начала тестового месяца
  минус горизонт; в живом режиме — фильтр по времени кандидатов.
* Нормировка признаков (среднее/σ) считается только по базе аналогов.
* Не больше одного аналога на окно `episode_gap` минут: соседние минуты одного
  эпизода почти одинаковы и иначе заняли бы весь топ, создавая ложную уверенность.
"""
from __future__ import annotations

from dataclasses import dataclass

import faiss
import numba as nb
import numpy as np
import pandas as pd

from .config import Config
from .features import y_close_cols

QLEVELS = np.array([0.10, 0.25, 0.50, 0.75, 0.90])


def minutes_of(index: pd.DatetimeIndex) -> np.ndarray:
    return index.as_unit("s").asi8 // 60


# --------------------------------------------------------------------------- индекс
class AnalogIndex:
    """Точный L2-индекс (FAISS IndexFlatL2) по взвешенным z-оценкам отпечатка."""

    def __init__(self, feat: pd.DataFrame, blocks: dict[str, list[str]], cfg: Config,
                 db_mask: np.ndarray, norm_mask: np.ndarray | None = None):
        """db_mask — какие минуты можно брать в аналоги; norm_mask — по каким считать
        среднее/σ признаков (по умолчанию те же)."""
        self.cfg = cfg
        self.cols = [c for b in blocks.values() for c in b]
        X = feat[self.cols].to_numpy(np.float32)
        self.db_rows = np.flatnonzero(db_mask)
        Xdb = X[self.db_rows]
        Xn = Xdb if norm_mask is None else X[np.flatnonzero(norm_mask)]
        mu = Xn.mean(0)
        sd = Xn.std(0) + 1e-6
        # вес столбца: вклад блока в квадрат расстояния = w_b^2, независимо от числа столбцов
        w = np.concatenate([np.full(len(v), cfg.block_weights[b] / np.sqrt(len(v)))
                            for b, v in blocks.items()]).astype(np.float32)
        self.mu, self.scale = mu, (w / sd).astype(np.float32)
        self.index = faiss.IndexFlatL2(len(self.cols))
        self.index.add(self._z(Xdb))
        self.db_min = minutes_of(feat.index)[self.db_rows]
        self.X = X

    def _z(self, X: np.ndarray) -> np.ndarray:
        return np.ascontiguousarray((X - self.mu) * self.scale, dtype=np.float32)

    def search(self, q_rows: np.ndarray, q_minutes: np.ndarray, k: int | None = None,
               chunk: int = 1024) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Для каждой минуты-запроса возвращает до k аналогов.

        Возврат: rows (nq,k) — номера строк feat (-1 если не хватило),
                 dist (nq,k) — евклидово расстояние, n_found (nq,).
        """
        k = k or self.cfg.k
        kc = min(self.cfg.k_candidates, self.index.ntotal)
        nq = len(q_rows)
        rows = np.full((nq, k), -1, np.int64)
        dist = np.full((nq, k), np.inf, np.float32)
        found = np.zeros(nq, np.int32)
        # «ведро» по 30 минут: два принятых аналога не могут лежать в одном ведре
        t0 = int(self.db_min.min())
        n_buck = int((self.db_min.max() - t0) // self.cfg.episode_gap) + 3
        stamp = np.full(n_buck, -1, np.int64)
        acc_t = np.zeros(n_buck, np.int64)
        for s in range(0, nq, chunk):
            e = min(s + chunk, nq)
            D, I = self.index.search(self._z(self.X[q_rows[s:e]]), kc)
            _dedup(I, D, self.db_min, q_minutes[s:e], t0, self.cfg.episode_gap, self.cfg.horizon,
                   stamp, acc_t, s, rows[s:e], dist[s:e], found[s:e])
            ok = rows[s:e] >= 0
            rows[s:e][ok] = self.db_rows[rows[s:e][ok]]
        return rows, np.sqrt(np.maximum(dist, 0)), found


@nb.njit(cache=True)
def _dedup(I, D, db_min, q_min, t0, gap, horizon, stamp, acc_t, qbase, out_rows, out_d, found):
    """Жадно идём по кандидатам от ближнего к дальнему, принимаем, если в пределах
    ±gap минут ещё нет принятого аналога и исход аналога известен к моменту запроса."""
    nq, kc = I.shape
    k = out_rows.shape[1]
    for q in range(nq):
        qid = qbase + q
        n = 0
        for j in range(kc):
            i = I[q, j]
            if i < 0:
                break
            tm = db_min[i]
            if tm + horizon >= q_min[q]:          # только прошлое с известным исходом
                continue
            b = (tm - t0) // gap
            clash = False
            for bb in range(b - 1, b + 2):
                if bb >= 0 and bb < stamp.shape[0] and stamp[bb] == qid and abs(acc_t[bb] - tm) < gap:
                    clash = True
            if clash:
                continue
            stamp[b] = qid
            acc_t[b] = tm
            out_rows[q, n] = i
            out_d[q, n] = D[q, j]
            n += 1
            if n == k:
                break
        found[q] = n


# --------------------------------------------------------------------------- веса и веер
def kernel_weights(dist: np.ndarray) -> np.ndarray:
    """Гауссово ядро с адаптивной шириной h = d_K / 2 (половина расстояния до
    самого дальнего из K аналогов): ближайший весит примерно в 3 раза больше дальнего."""
    d = np.where(np.isfinite(dist), dist, np.nan)
    h = 0.5 * np.nanmax(d, axis=-1, keepdims=True) + 1e-9
    w = np.exp(-0.5 * (d / h) ** 2)
    return np.nan_to_num(w, nan=0.0)


@nb.njit(cache=True)
def _wquant_1d(x, w, levels):
    """Взвешенные квантили (интерполяция по середине массы каждого наблюдения)."""
    o = np.argsort(x)
    xs, ws = x[o], w[o]
    tot = ws.sum()
    out = np.full(levels.shape[0], np.nan)
    if tot <= 0:
        return out
    cw = (np.cumsum(ws) - 0.5 * ws) / tot
    for i in range(levels.shape[0]):
        out[i] = np.interp(levels[i], cw, xs)
    return out


@nb.njit(cache=True, parallel=True)
def wquantiles(Y, W, levels):
    """Y: (nq, K, H) пути аналогов, W: (nq, K) веса -> (nq, H, L) квантили."""
    nq, K, H = Y.shape
    out = np.full((nq, H, levels.shape[0]), np.nan)
    for q in nb.prange(nq):
        m = W[q] > 0
        for h in range(H):
            y = Y[q, :, h]
            ok = m & np.isfinite(y)
            if ok.sum() >= 5:
                out[q, h] = _wquant_1d(y[ok], W[q][ok], levels)
    return out


@dataclass
class Fan:
    quant: np.ndarray      # (H, 5) квантили 10/25/50/75/90 пути, ATR
    p_up: float            # P(+b ATR раньше −b ATR)
    p_dn: float
    exp_move: float        # взвешенное среднее хода к t+H, ATR
    width80: float         # ширина 80% полосы на t+H, ATR
    ess: float             # эффективный размер выборки аналогов
    n_alive: int


def touch_prob(touch: np.ndarray, w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """touch: 1/0/0.5/NaN(ни один) -> P(вверх первым), P(вниз первым)."""
    t = np.nan_to_num(touch, nan=-1.0)
    up = (t == 1) + 0.5 * (t == 0.5)
    dn = (t == 0) + 0.5 * (t == 0.5)
    sw = w.sum(-1) + 1e-12
    return (w * up).sum(-1) / sw, (w * dn).sum(-1) / sw


def make_fan(Y: np.ndarray, touch: np.ndarray, w: np.ndarray, n_alive: int | None = None) -> Fan:
    """Веер для одного запроса: Y (K,H), touch (K,), w (K,)."""
    q = wquantiles(Y[None].astype(np.float64), w[None].astype(np.float64), QLEVELS)[0]
    pu, pd_ = touch_prob(touch, w)
    sw = w.sum() + 1e-12
    return Fan(quant=q, p_up=float(pu), p_dn=float(pd_),
               exp_move=float((w * Y[:, -1]).sum() / sw), width80=float(q[-1, 4] - q[-1, 0]),
               ess=float(sw ** 2 / ((w ** 2).sum() + 1e-12)),
               n_alive=int((w > 0).sum() if n_alive is None else n_alive))


# --------------------------------------------------------------------------- живое сужение
def live_reweight(Y: np.ndarray, w0: np.ndarray, realized: np.ndarray, cfg: Config
                  ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Прошло k минут после якоря; realized — реальный путь закрытий (k,) в ATR якоря.

    Ошибка аналога — RMSE его первых k минут от реального пути, где каждая минута j
    нормирована на разброс веера в эту минуту s_j (иначе ранние минуты с узким
    веером ничего не весят, а поздние — всё). Вес умножается на exp(-e^2 / 2τ^2),
    «живой» — e <= alive_rmse. Возврат: новые веса, ошибки (K,), маска живых.
    """
    k = len(realized)
    if k == 0:
        return w0.copy(), np.zeros(len(w0)), w0 > 0
    m = w0 > 0
    mu = np.average(np.nan_to_num(Y[m, :k]), axis=0, weights=w0[m])
    s = np.sqrt(np.average((np.nan_to_num(Y[m, :k]) - mu) ** 2, axis=0, weights=w0[m])) + 1e-6
    err = np.sqrt(np.nanmean(((Y[..., :k] - realized) / s) ** 2, axis=-1))
    err = np.where(np.isfinite(err), err, np.inf)
    w = w0 * np.exp(-0.5 * (err / cfg.live_sigma) ** 2)
    alive = (err <= cfg.alive_rmse) & (w0 > 0)
    return w, err, alive


def narrowed_quantiles(Y: np.ndarray, w: np.ndarray, realized: np.ndarray) -> np.ndarray:
    """Веер на оставшиеся минуты k+1..H, «пересаженный» на текущую цену:
    путь_j = r_k + (Y_j - Y_k) по перевзвешенным аналогам. Возврат (H-k, 5)."""
    k = len(realized)
    if k == 0:
        return wquantiles(Y[None].astype(np.float64), w[None].astype(np.float64), QLEVELS)[0]
    Z = realized[-1] + (Y[:, k:] - Y[:, k - 1:k])
    return wquantiles(Z[None].astype(np.float64), w[None].astype(np.float64), QLEVELS)[0]


def live_reweight_batch(Y: np.ndarray, W: np.ndarray, R: np.ndarray, k: int, cfg: Config):
    """Пакетная версия live_reweight для валидации: Y (nq,K,H), W (nq,K), R (nq,H)."""
    Yk = np.nan_to_num(Y[:, :, :k].astype(np.float64))
    sw = W.sum(1, keepdims=True)[:, :, None] + 1e-12
    mu = (W[:, :, None] * Yk).sum(1, keepdims=True) / sw
    s = np.sqrt((W[:, :, None] * (Yk - mu) ** 2).sum(1, keepdims=True) / sw) + 1e-6
    err = np.sqrt((((Yk - R[:, None, :k]) / s) ** 2).mean(-1))
    err = np.where(np.isfinite(Y[:, :, :k]).all(-1), err, np.inf)
    Wn = W * np.exp(-0.5 * (err / cfg.live_sigma) ** 2)
    alive = (err <= cfg.alive_rmse) & (W > 0)
    return Wn, alive


# --------------------------------------------------------------------------- режим рынка
def regime_label(v_long: float, eff60: float, p60: float, v_q: tuple[float, float],
                 e_q: tuple[float, float]) -> str:
    """Словесный режим по терцилям базы: волатильность × направленность."""
    vol = "низкая вол." if v_long < v_q[0] else "высокая вол." if v_long > v_q[1] else "обычная вол."
    if eff60 > e_q[1]:
        tr = "тренд ↑" if p60 < 0 else "тренд ↓"   # p60 = (C[t-60]-C[t])/ATR < 0 -> цена выросла
    elif eff60 < e_q[0]:
        tr = "флэт"
    else:
        tr = "смешанный"
    return f"{tr} · {vol}"


def gather_paths(feat: pd.DataFrame, rows: np.ndarray, cfg: Config) -> tuple[np.ndarray, np.ndarray]:
    """Пути закрытий (…,K,H) и исходы барьера (…,K) по номерам строк; -1 -> NaN."""
    Yall = feat[y_close_cols(cfg)].to_numpy(np.float32)
    T = feat["touch"].to_numpy(np.float32)
    safe = np.where(rows >= 0, rows, 0)
    Y = Yall[safe]
    Tt = T[safe]
    Y[rows < 0] = np.nan
    Tt[rows < 0] = np.nan
    return Y, Tt
