"""Скачивает замещающие данные (BTCUSDT 1m, Binance public dump) и пишет CSV
в том же формате, что ожидается от пользователя: time,open,high,low,close,volume.

Это ТОЛЬКО заглушка для отладки конвейера, пока нет CSV целевого инструмента.
"""
import io
import sys
import zipfile
import urllib.request
from pathlib import Path

import pandas as pd

BASE = "https://data.binance.vision/data/spot/monthly/klines/{s}/1m/{s}-1m-{m}.zip"


def months(start: str, end: str):
    for p in pd.period_range(start, end, freq="M"):
        yield p.strftime("%Y-%m")


def main(symbol="BTCUSDT", start="2024-09", end="2026-08", out="data/standin_BTCUSDT_1m.csv"):
    frames = []
    for m in months(start, end):
        raw = urllib.request.urlopen(BASE.format(s=symbol, m=m), timeout=120).read()
        with zipfile.ZipFile(io.BytesIO(raw)) as z:
            df = pd.read_csv(z.open(z.namelist()[0]), header=None, usecols=range(6))
        # С 2025 г. Binance отдаёт open_time в микросекундах, раньше — в миллисекундах
        t = df[0].astype("int64")
        unit = "us" if t.iloc[0] > 1e14 else "ms"
        df[0] = pd.to_datetime(t, unit=unit, utc=True).dt.strftime("%Y-%m-%d %H:%M:%S")
        frames.append(df)
        print(m, len(df), file=sys.stderr)
    out_df = pd.concat(frames, ignore_index=True)
    out_df.columns = ["time", "open", "high", "low", "close", "volume"]
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out, index=False)
    print(f"wrote {out}: {len(out_df):,} rows")


if __name__ == "__main__":
    main(*sys.argv[1:])
