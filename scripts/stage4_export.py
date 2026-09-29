"""Этап 4: экспорт окна истории для веб-страницы -> web/data.js

python scripts/stage4_export.py [start] [end] [instrument]
По умолчанию — последние 2 полных дня данных (без выбора «красивого» участка).
"""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analog_fan.config import Config
from analog_fan.export import export_window

cfg = Config()
feat = pd.read_parquet("data/features.parquet")
clean = pd.read_parquet("data/clean.parquet")
blocks = {k: v.split(",") for k, v in json.loads(Path("data/blocks.json").read_text()).items()}
end = pd.Timestamp(sys.argv[2], tz="UTC") if len(sys.argv) > 2 else (feat.index[-1] + pd.Timedelta(minutes=1)).normalize()
start = pd.Timestamp(sys.argv[1], tz="UTC") if len(sys.argv) > 1 else end - pd.Timedelta(days=2)
instrument = sys.argv[3] if len(sys.argv) > 3 else "BTCUSDT (замещающие данные)"
summ = Path("reports/stage3_summary.json")
validation = json.loads(summ.read_text()) if summ.exists() else {}
d = export_window(feat, clean, blocks, cfg, start, end, instrument=instrument, validation=validation)
Path("web").mkdir(exist_ok=True)
txt = "window.AF_DATA=" + json.dumps(d, ensure_ascii=False, separators=(",", ":")) + ";\n"
Path("web/data.js").write_text(txt)
print(f"{start} … {end}: {d['n']} минут, {len(d['candles']['t'])} свечей, web/data.js = {len(txt) / 1e6:.1f} МБ")
