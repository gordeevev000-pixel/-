"""Самопроверка загрузчика: делает из чистых данных «грязный» экспорт MT4
(без заголовка, дата и время отдельно, пояс EET, FX-расписание, дыры, дубли)
и печатает отчёт проверки. Ожидается: дубли найдены, выходные и ежедневный
перерыв распознаны, случайные дыры попали во внутрисессионные.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analog_fan import data

d = pd.read_parquet("data/clean.parquet").loc["2025-01-01":"2025-04-30"]
loc = d.index.tz_convert("America/New_York")
wd, hm = loc.dayofweek, loc.hour * 60 + loc.minute
closed = ((wd == 4) & (hm >= 17 * 60)) | (wd == 5) | ((wd == 6) & (hm < 17 * 60)) | ((hm >= 17 * 60) & (hm < 17 * 60 + 5))
d = d[~closed]
rng = np.random.default_rng(0)
d = d.drop(d.index[rng.choice(len(d), 3000, replace=False)])
d = pd.concat([d, d.iloc[rng.choice(len(d), 50)]])
e = d.index.tz_convert("EET")
out = pd.DataFrame({"d": e.strftime("%Y.%m.%d"), "t": e.strftime("%H:%M"), "o": d.open, "h": d.high,
                    "l": d.low, "c": d.close, "v": d.volume.round(0)})
tmp = "data/_mt4_test.csv"
out.to_csv(tmp, header=False, index=False)
raw = data.load_csv(tmp, src_tz="EET")
clean, rep = data.validate_and_clean(raw, max_fill=3, market_tz="America/New_York")
print(rep.text())
Path(tmp).unlink()
