"""Загрузка минутного CSV и проверка качества данных.

Принципы:
* Загрузчик терпим к формату: заголовок с любым регистром, отдельные колонки
  date+time (экспорт MT4/MT5), unix-время в с/мс/мкс, строки ISO.
* Проверка ничего молча не «чинит». Каждое исправление попадает в отчёт.
* Пропуски НЕ заполняются интерполяцией: короткие дыры (<= max_fill минут)
  заполняются плоскими барами с нулевым объёмом (минута без сделок), длинные
  разрезают ряд на сегменты. Окно отпечатка + горизонт исхода не может
  пересекать границу сегмента, так что искусственных путей в выборке нет.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

_ALIASES = {
    "time": ["time", "datetime", "timestamp", "date_time", "open_time", "dt", "<dtyyyymmdd>"],
    "date": ["date", "<date>", "day"],
    "open": ["open", "o", "<open>"],
    "high": ["high", "h", "<high>"],
    "low": ["low", "l", "<low>"],
    "close": ["close", "c", "<close>", "last"],
    "volume": ["volume", "vol", "v", "tickvol", "tick_volume", "<vol>", "<tickvol>"],
}


def _find(cols: list[str], key: str) -> str | None:
    low = {c.lower().strip(): c for c in cols}
    for a in _ALIASES[key]:
        if a in low:
            return low[a]
    return None


def _parse_time(s: pd.Series) -> pd.Series:
    """Число -> unix (единица по порядку величины), иначе парсинг строк."""
    if pd.api.types.is_numeric_dtype(s):
        v = s.astype("int64")
        mag = np.nanmedian(v.to_numpy())
        unit = "s" if mag < 1e11 else "ms" if mag < 1e14 else "us" if mag < 1e17 else "ns"
        return pd.to_datetime(v, unit=unit)
    return pd.to_datetime(s.astype(str).str.replace(".", "-", regex=False)
                          if s.astype(str).str.match(r"^\d{4}\.\d{2}\.\d{2}").all() else s)


def load_csv(path: str, src_tz: str = "UTC") -> pd.DataFrame:
    """Читает CSV и возвращает DataFrame с индексом времени в UTC.

    src_tz — часовой пояс, в котором записано время в файле (если в строках нет
    смещения). Для брокеров MT4 это часто 'EET' (UTC+2/+3), для CME — 'America/Chicago'.
    """
    df = pd.read_csv(path, sep=None, engine="python") if path.endswith(".txt") else pd.read_csv(path)
    cols = list(df.columns)
    if all(_find(cols, k) is None for k in ("open", "close")):  # без заголовка
        df = pd.read_csv(path, header=None)
        n = df.shape[1]
        names = ["time", "open", "high", "low", "close", "volume"] if n <= 6 else \
                ["date", "time", "open", "high", "low", "close", "volume"] + [f"x{i}" for i in range(n - 7)]
        df.columns = names[:n]
        cols = list(df.columns)

    tcol, dcol = _find(cols, "time"), _find(cols, "date")
    if dcol is not None and tcol is not None and dcol != tcol:
        ts = _parse_time(df[dcol].astype(str) + " " + df[tcol].astype(str))
    else:
        ts = _parse_time(df[tcol or dcol])

    out = pd.DataFrame({k: pd.to_numeric(df[_find(cols, k)], errors="coerce")
                        for k in ("open", "high", "low", "close")})
    vcol = _find(cols, "volume")
    out["volume"] = pd.to_numeric(df[vcol], errors="coerce") if vcol else np.nan
    if ts.dt.tz is None:
        ts = ts.dt.tz_localize(src_tz, ambiguous="NaT", nonexistent="NaT")
    out.index = pd.DatetimeIndex(ts.dt.tz_convert("UTC"), name="time")
    return out


@dataclass
class DataReport:
    """Сводка проверки. `lines` — готовый текст для отчёта (markdown)."""
    stats: dict = field(default_factory=dict)
    lines: list[str] = field(default_factory=list)

    def add(self, s: str = ""):
        self.lines.append(s)

    def text(self) -> str:
        return "\n".join(self.lines)


def _minutes(idx: pd.DatetimeIndex) -> np.ndarray:
    """Минуты от эпохи, независимо от внутреннего разрешения индекса (ns/us/s в pandas 3)."""
    return idx.as_unit("s").asi8 // 60


def _hour_profile(df: pd.DataFrame) -> pd.DataFrame:
    r = np.log(df["close"]).diff().abs()
    g = pd.DataFrame({"absret": r, "vol": df["volume"]}).groupby(df.index.hour)
    prof = g.mean()
    return prof / prof.mean()


def validate_and_clean(df: pd.DataFrame, max_fill: int = 3, market_tz: str = "UTC",
                       ) -> tuple[pd.DataFrame, DataReport]:
    """Проверяет данные и возвращает очищенный ряд на минутной сетке.

    Возвращаемый DataFrame содержит колонки OHLCV, `filled` (бар добавлен как
    минута без сделок) и `seg` (номер непрерывного сегмента).
    """
    rep = DataReport()
    n0 = len(df)
    rep.add(f"* Строк в файле: **{n0:,}**, период {df.index.min()} — {df.index.max()} (UTC)")

    # --- время, которое не удалось распознать / DST-неоднозначное
    bad_t = df.index.isna().sum()
    if bad_t:
        rep.add(f"* Нераспознанное или неоднозначное (переход DST) время: {bad_t} строк — удалены")
        df = df[~df.index.isna()]

    # --- порядок
    if not df.index.is_monotonic_increasing:
        n_back = int((np.diff(df.index.asi8) < 0).sum())
        rep.add(f"* Время идёт не по порядку в {n_back} местах — отсортировано")
        df = df.sort_index(kind="stable")

    # --- секунды внутри минуты (время закрытия вместо открытия и т.п.)
    off = df.index.second != 0
    if off.any():
        rep.add(f"* {off.sum()} меток не на границе минуты — округлены вниз")
        df.index = df.index.floor("min")

    # --- дубли
    dup_mask = df.index.duplicated(keep=False)
    n_dup_rows = int(df.index.duplicated().sum())
    if n_dup_rows:
        d = df[dup_mask]
        same = d.groupby(level=0).nunique().max(axis=1).le(1).sum()
        rep.add(f"* Дубли по времени: {n_dup_rows} лишних строк ({same} полностью одинаковых групп, "
                f"{d.index.nunique() - same} с разными ценами) — оставлена последняя запись")
        df = df[~df.index.duplicated(keep="last")]
    else:
        rep.add("* Дублей по времени нет")

    # --- NaN и некорректный OHLC
    nan_rows = df[["open", "high", "low", "close"]].isna().any(axis=1)
    if nan_rows.any():
        rep.add(f"* Строк с пустыми ценами: {nan_rows.sum()} — удалены")
        df = df[~nan_rows]
    nonpos = (df[["open", "high", "low", "close"]] <= 0).any(axis=1)
    hi_bad = df["high"] < df[["open", "close"]].max(axis=1) - 1e-12
    lo_bad = df["low"] > df[["open", "close"]].min(axis=1) + 1e-12
    bad = nonpos | hi_bad | lo_bad
    rep.add(f"* Нарушения OHLC (high < max(O,C), low > min(O,C), цена <= 0): **{int(bad.sum())}**"
            + (" — исправлены: high/low расширены до O/C" if bad.any() else ""))
    if bad.any():
        df = df.copy()
        df["high"] = df[["open", "high", "close"]].max(axis=1)
        df["low"] = df[["open", "low", "close"]].min(axis=1)
        df = df[~nonpos]

    # --- выбросы: скачок > 15 средних |ΔC| за сутки и возврат >= 80% на следующей минуте
    r = np.log(df["close"]).diff()
    scale = r.abs().rolling(1440, min_periods=60).mean()
    spike = (r.abs() > 15 * scale) & (-r.shift(-1) / r > 0.8)
    rep.add(f"* Подозрительные одиночные «иглы» (скачок > 15 средних минутных ходов и возврат >= 80% "
            f"на следующей минуте): {int(spike.sum())} — оставлены, список в stats['spikes']")
    rep.stats["spikes"] = [str(t) for t in df.index[spike][:50]]

    # --- нулевой объём и плоские бары
    flat = (df["high"] == df["low"])
    zero_v = df["volume"].fillna(0) <= 0
    rng = (df["high"] - df["low"])
    tick = np.nanmin(np.where(np.diff(np.sort(df["close"].unique())) > 0, np.diff(np.sort(df["close"].unique())), np.nan))
    tiny = rng <= 2 * tick + 1e-12
    rep.add(f"* Плоских баров (H=L): {flat.mean():.2%}; с диапазоном <= 2 тиков (тик ~ {tick:.6g}): {tiny.mean():.2%}; "
            f"с нулевым/пустым объёмом: {zero_v.mean():.2%}")
    rep.stats["tick"] = float(tick)

    # --- пропуски
    dt_min = np.diff(_minutes(df.index))
    gap_idx = np.where(dt_min > 1)[0]
    gaps = pd.DataFrame({
        "start": df.index[gap_idx] + pd.Timedelta(minutes=1),   # первая отсутствующая минута
        "missing": dt_min[gap_idx] - 1,
    })
    local_start = gaps["start"].dt.tz_convert(market_tz)
    # выходные: дыра >= 24 ч и содержит субботу по времени рынка
    covers_sat = [(pd.date_range(s, periods=min(m, 4 * 1440), freq="min").tz_convert(market_tz).dayofweek == 5).any()
                  if m >= 1440 else False for s, m in zip(gaps["start"], gaps["missing"])]
    gaps["kind"] = "внутрисессионная (<= %d мин)" % max_fill
    gaps.loc[gaps["missing"] > max_fill, "kind"] = "длинная (> %d мин)" % max_fill
    gaps.loc[np.array(covers_sat, dtype=bool), "kind"] = "выходные"
    # ежедневный перерыв: длинная дыра, начинающаяся в одну и ту же минуту суток на многих днях
    long_mask = gaps["kind"].str.startswith("длинная")
    if long_mask.any():
        hm = local_start[long_mask].dt.strftime("%H:%M")
        n_days = df.index.tz_convert(market_tz).normalize().nunique()
        top = hm.value_counts()
        daily = top[top > 0.3 * n_days].index
        gaps.loc[long_mask & local_start.dt.strftime("%H:%M").isin(daily), "kind"] = "ежедневный перерыв"
    summ = gaps.groupby("kind")["missing"].agg(["count", "sum", "max"]) if len(gaps) else None
    total_grid = int((df.index[-1] - df.index[0]) / pd.Timedelta(minutes=1)) + 1
    rep.add(f"* Минутная сетка от первой до последней записи: {total_grid:,} минут, "
            f"отсутствует {total_grid - len(df):,} ({1 - len(df) / total_grid:.2%})")
    if summ is not None and len(summ):
        rep.add("")
        rep.add("  | тип пропуска | число | минут всего | макс. длина, мин |")
        rep.add("  |---|---:|---:|---:|")
        for k, row in summ.iterrows():
            rep.add(f"  | {k} | {row['count']:,} | {row['sum']:,} | {row['max']:,} |")
        big = gaps[gaps["kind"].str.startswith("длинная")].nlargest(5, "missing")
        if len(big):
            rep.add("")
            rep.add("  Крупнейшие длинные пропуски: " + "; ".join(
                f"{s:%Y-%m-%d %H:%M} ({m} мин)" for s, m in zip(big["start"], big["missing"])))
    else:
        rep.add("* Пропусков нет")
    rep.stats["gaps"] = gaps

    # --- выходные и часовой пояс
    loc = df.index.tz_convert(market_tz)
    wk = loc.dayofweek >= 5
    rep.add(f"* Баров в субботу/воскресенье (время рынка {market_tz}): {wk.mean():.2%} "
            + ("— торговля 24/7" if wk.mean() > 0.2 else "— рынок закрыт по выходным" if wk.mean() < 0.02
               else "— частично (воскресное открытие/праздники)"))
    prof = _hour_profile(df)
    top_h = prof["absret"].nlargest(3).index.tolist()
    rep.add(f"* Часовой пояс: время приведено к UTC. Пики волатильности по часам UTC: {top_h} "
            f"(ориентиры: Лондон ~7–8 UTC, Нью-Йорк ~13–14 UTC зимой / 12–13 летом)")
    week_gaps = gaps[gaps["kind"] == "выходные"]
    if len(week_gaps) > 4:
        reopen = (week_gaps["start"] + pd.to_timedelta(week_gaps["missing"], unit="min"))
        hrs = reopen.dt.hour.value_counts()
        rep.add(f"* Открытие после выходных по UTC (час: число недель): { {int(k): int(v) for k, v in hrs.head(3).items()} }. "
                f"Если час «гуляет» на 1 между летом и зимой — рынок живёт по местному времени с DST, это нормально.")
    rep.stats["hour_profile"] = prof

    # --- заполнение коротких дыр и сегментация
    full = df.reindex(pd.date_range(df.index[0], df.index[-1], freq="min", name="time"))
    missing = full["close"].isna()
    run_id = (missing != missing.shift()).cumsum()
    run_len = missing.groupby(run_id).transform("sum")
    fill = missing & (run_len <= max_fill)
    c = full["close"].ffill()
    for k in ("open", "high", "low", "close"):
        full.loc[fill, k] = c[fill]
    full.loc[fill, "volume"] = 0.0
    full = full[~full["close"].isna()].copy()
    full["filled"] = fill[full.index].to_numpy()
    m = _minutes(full.index)
    step = np.diff(m, prepend=m[0])
    full["seg"] = np.cumsum(step > 1).astype(np.int32)
    rep.add(f"* После очистки: {len(full):,} минут, из них заполнено как «минута без сделок»: "
            f"{int(fill.sum()):,}; непрерывных сегментов: {full['seg'].nunique():,}")
    rep.stats.update(n_raw=n0, n_clean=len(full), n_filled=int(fill.sum()), n_segments=int(full["seg"].nunique()))
    return full, rep
