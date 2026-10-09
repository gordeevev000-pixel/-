"""Минутные свечи histdata.com для товаров и индексов (бесплатно, без регистрации).

Прошлые годы отдаются одним архивом на год, текущий — по месяцам.
Время в файлах — «нью-йоркское зимнее (UTC−5) плюс летнее время по ЕВРОПЕЙСКИМ датам»
(с 2019 года; в 2018-м — по американским). Сайт пишет «EST без DST», но это не так.
Проверено по событиям: NFP в 8:30 и решения ФРС в 14:00 по Нью-Йорку стоят на своём месте
и зимой, и летом, а в недели, когда США уже перешли на летнее время, а Европа ещё нет
(конец марта, конец октября — начало ноября), с 2019 года оказываются на час раньше
(ФРС 2019-03-20 — в 13:00 по времени файла). Переводим в UTC по этим правилам.
Объёма в этих данных нет.

Запуск:  python signals/fetch_histdata.py <папка> [первый год]
"""
from __future__ import annotations

import io
import re
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests

PAGE = "https://www.histdata.com/download-free-forex-historical-data/?/ascii/1-minute-bar-quotes/{p}/{y}{m}"
GET = "https://www.histdata.com/get.php"
UA = {"User-Agent": "Mozilla/5.0"}

SYMBOLS = {"GOLD": "XAUUSD", "SILVER": "XAGUSD", "WTI": "WTIUSD", "BRENT": "BCOUSD",
           "SP500": "SPXUSD", "NASDAQ": "NSXUSD", "DAX": "GRXEUR"}


def to_utc(df: pd.DataFrame) -> pd.DataFrame:
    """Время файла → UTC. До 2019 года — нью-йоркское с американским переходом на летнее время;
    с 2019-го — UTC−5 плюс европейское летнее время, то есть (время + 5 ч) по Лондону.
    Неоднозначные часы перевода (ночь на воскресенье, рынки закрыты) выбрасываются."""
    naive = df.index
    old = naive < pd.Timestamp("2019-01-01")
    ny = naive[old].tz_localize("America/New_York", ambiguous="NaT", nonexistent="NaT")
    ld = (naive[~old] + pd.Timedelta(hours=5)).tz_localize("Europe/London", ambiguous="NaT", nonexistent="NaT")
    idx = ny.tz_convert("UTC").append(ld.tz_convert("UTC"))
    order = np.r_[np.flatnonzero(old), np.flatnonzero(~old)]
    out = df.iloc[order].copy()
    out.index = idx
    out = out[~out.index.isna()]
    return out[~out.index.duplicated()].sort_index()


def fetch(pair: str, year: int, month: int | None) -> pd.DataFrame | None:
    m = "" if month is None else f"/{month}"
    page = PAGE.format(p=pair.lower(), y=year, m=m)
    for k in range(6):
        try:
            html = requests.get(page, headers=UA, timeout=60).text
            tk = re.search(r'id="tk" value="([^"]+)"', html)
            if not tk:
                return None
            dm = f"{year}" if month is None else f"{year}{month:02d}"
            r = requests.post(GET, headers={**UA, "Referer": page}, timeout=300,
                              data={"tk": tk.group(1), "date": str(year), "datemonth": dm,
                                    "platform": "ASCII", "timeframe": "M1", "fxpair": pair})
            if r.content[:2] != b"PK":
                return None
            z = zipfile.ZipFile(io.BytesIO(r.content))
            name = next(n for n in z.namelist() if n.endswith(".csv"))
            df = pd.read_csv(z.open(name), sep=";", header=None,
                             names=["t", "open", "high", "low", "close", "volume"])
            return to_utc(df.set_index(pd.to_datetime(df.pop("t"), format="%Y%m%d %H%M%S"))).astype(float)
        except (requests.RequestException, zipfile.BadZipFile, StopIteration):
            time.sleep(2 ** k)
    raise RuntimeError(f"не скачалось: {pair} {year} {month}")


def main(out: Path, first: int) -> None:
    out.mkdir(parents=True, exist_ok=True)
    now = pd.Timestamp.now(tz="UTC")
    for name, pair in SYMBOLS.items():
        f = out / f"{name}_1m.parquet"
        if f.exists():
            continue
        parts = []
        for y in range(first, now.year + 1):
            if y < now.year:
                d = fetch(pair, y, None)
                if d is None:
                    print(f"  {name}: на histdata нет данных за {y}", flush=True)
                parts.append(d)
            else:
                for m in range(1, now.month + 1):
                    d = fetch(pair, y, m)
                    if d is None:
                        print(f"  {name}: на histdata нет данных за {y}-{m:02d}", flush=True)
                    parts.append(d)
        df = pd.concat([p for p in parts if p is not None]).sort_index()
        df = df[~df.index.duplicated()]
        df.to_parquet(f)
        print(name, len(df), df.index[0], df.index[-1], flush=True)


if __name__ == "__main__":
    main(Path(sys.argv[1]), int(sys.argv[2]) if len(sys.argv) > 2 else 2018)
