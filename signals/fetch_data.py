"""Скачивает историю для проверки индикатора.

Крипта — Binance (публичный API, без ключа), остальное — Yahoo Finance.
Запуск:  python signals/fetch_data.py <папка>
"""
from __future__ import annotations

import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import requests

BINANCE = "https://data-api.binance.vision/api/v3/klines"
YAHOO = "https://query1.finance.yahoo.com/v8/finance/chart/{sym}"
UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"}

CRYPTO = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT"]
CRYPTO_TF = ["15m", "1h", "4h", "1d"]
YAHOO_SYMS = {"GOLD": "GC=F", "EURUSD": "EURUSD=X", "SPX": "^GSPC", "NASDAQ": "^NDX"}


def get(url: str, params: dict, headers: dict = UA) -> requests.Response:
    for k in range(6):
        try:
            r = requests.get(url, params=params, headers=headers, timeout=30)
            if r.status_code == 200:
                return r
        except requests.RequestException:
            pass
        time.sleep(2 ** k)
    raise RuntimeError(f"не скачалось: {url} {params}")


def binance(sym: str, tf: str, start: str = "2020-01-01") -> pd.DataFrame:
    t = int(pd.Timestamp(start, tz="UTC").timestamp() * 1000)
    rows = []
    while True:
        batch = get(BINANCE, {"symbol": sym, "interval": tf, "startTime": t, "limit": 1000}).json()
        if not batch:
            break
        rows += batch
        t = batch[-1][0] + 1
        if len(batch) < 1000:
            break
    df = pd.DataFrame(rows, columns=list("tohlcv") + [f"x{k}" for k in range(6)])[list("tohlcv")]
    df["t"] = pd.to_datetime(df["t"], unit="ms", utc=True)
    df = df.set_index("t").astype(float)
    df.columns = ["open", "high", "low", "close", "volume"]
    return df.iloc[:-1]          # последний бар ещё не закрыт


def yahoo(sym: str, interval: str, rng: str) -> pd.DataFrame:
    # range=max Yahoo отдаёт помесячно, поэтому для дневок — явные даты
    q = {"interval": interval, "range": rng} if rng != "max" else \
        {"interval": interval, "period1": 946684800, "period2": int(time.time())}
    d = get(YAHOO.format(sym=sym), q, {"User-Agent": "Mozilla/5.0"}).json()["chart"]["result"][0]
    q = d["indicators"]["quote"][0]
    df = pd.DataFrame({k: q[k] for k in ("open", "high", "low", "close", "volume")},
                      index=pd.to_datetime(d["timestamp"], unit="s", utc=True))
    return df.dropna(subset=["open", "high", "low", "close"]).astype(float).iloc[:-1]   # последний бар не закрыт


def crypto_job(sym: str, tf: str, out: Path) -> None:
    f = out / f"{sym}_{tf}.parquet"
    if f.exists():
        return
    df = binance(sym, tf, "2022-01-01" if tf == "15m" else "2019-01-01")
    df.to_parquet(f)
    print(f.name, len(df), df.index[0].date(), "→", df.index[-1].date(), flush=True)


def main(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=10) as pool:
        for job in [pool.submit(crypto_job, s, tf, out) for s in CRYPTO for tf in CRYPTO_TF]:
            job.result()
    for name, sym in YAHOO_SYMS.items():
        for interval, rng in (("1h", "730d"), ("1d", "max")):
            f = out / f"{name}_{interval}.parquet"
            if f.exists():
                continue
            df = yahoo(sym, interval, rng)
            if interval == "1d":
                df = df[df.index >= "2000-01-01"]
            df.to_parquet(f)
            print(f.name, len(df), df.index[0].date(), "→", df.index[-1].date())
            time.sleep(1.5)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
