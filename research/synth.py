"""Генератор синтетических минутных баров.

Три режима — возврат к средней (OU), тренд и чистое случайное блуждание —
переключаются цепью Маркова. Бар собирается из под-шагов, поэтому хай и лоу
ведут себя как у настоящей минутки, а не как |open-close|.

Случайное блуждание нужно отдельно: на нём у любой модели эджа нет по построению,
и всё, что покажет там винрейт-контроллер, — это геометрия выходов, а не сигнал.
"""
from __future__ import annotations

import numpy as np

SUBSTEPS = 12  # под-шагов в минуте: из них собираются high/low


def _bars_from_path(path: np.ndarray, n: int, rng: np.random.Generator) -> dict:
    """path: (n*SUBSTEPS+1,) цены → OHLCV минуток."""
    p = path[1:].reshape(n, SUBSTEPS)
    o = np.concatenate([[path[0]], p[:-1, -1]])
    c = p[:, -1]
    h = np.maximum(p.max(axis=1), o)
    l = np.minimum(p.min(axis=1), o)
    move = np.abs(np.diff(path).reshape(n, SUBSTEPS)).sum(axis=1)
    base = np.median(move) if np.median(move) > 0 else 1.0
    v = np.maximum(1.0, rng.lognormal(mean=np.log(1000.0 * move / base + 1.0), sigma=0.35))
    return dict(open=o, high=h, low=l, close=c, volume=v)


def make(n: int = 120_000, seed: int = 7, mode: str = "mixed", price0: float = 100.0) -> dict:
    """mode: mixed | rw | revert | trend."""
    rng = np.random.default_rng(seed)
    steps = n * SUBSTEPS
    sigma = 0.00035  # волатильность под-шага

    if mode == "rw":
        regimes = np.zeros(steps, dtype=np.int8)
    else:
        # цепь Маркова по режимам, средняя длина куска ~ 4 часа
        switch = rng.random(steps) < 1.0 / (240 * SUBSTEPS)
        pool = {"mixed": [0, 1, 2], "revert": [1], "trend": [2]}[mode]
        regimes = np.zeros(steps, dtype=np.int8)
        cur = pool[0]
        for i in range(steps):
            if switch[i]:
                cur = pool[rng.integers(len(pool))]
            regimes[i] = cur

    path = np.empty(steps + 1)
    path[0] = price0
    anchor = price0
    drift = 0.0
    kappa = 0.004  # сила возврата к якорю на под-шаг
    shock = rng.normal(0.0, sigma, steps)
    for i in range(steps):
        x = path[i]
        r = regimes[i]
        anchor += (x - anchor) * 0.002  # якорь тянется за ценой
        if r == 1:
            mu = kappa * (anchor - x) / x
        elif r == 2:
            if i % (SUBSTEPS * 60) == 0:
                drift = rng.choice([-1.0, 1.0]) * sigma * 0.22
            mu = drift
        else:
            mu = 0.0
        path[i + 1] = x * (1.0 + mu + shock[i])

    bars = _bars_from_path(path, n, rng)
    bars["minute"] = np.arange(n) % 1440
    bars["day"] = np.arange(n) // 1440
    bars["mode"] = mode
    return bars
