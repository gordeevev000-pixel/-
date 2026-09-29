"""Этап 3а: прогон walk-forward. Пишет data/wf_results.parquet."""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analog_fan.config import Config
from analog_fan.validate import run_walk_forward

first_test = sys.argv[1] if len(sys.argv) > 1 else "2025-03-01"
step = int(sys.argv[2]) if len(sys.argv) > 2 else 5
cfg = Config()
feat = pd.read_parquet("data/features.parquet")
blocks = {k: v.split(",") for k, v in json.loads(Path("data/blocks.json").read_text()).items()}
res = run_walk_forward(feat, blocks, cfg, first_test=first_test, step=step,
                       log=lambda s: print(s, flush=True))
res.to_parquet("data/wf_results.parquet")
print(f"итого запросов: {len(res):,}")
