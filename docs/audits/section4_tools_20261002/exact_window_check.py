"""Prompt 4 classification aid: recompute the frozen dataset config on EXACTLY the app's window for given endpoints
(the last runtime_history_bars bars up to the endpoint: training source for H, recorded bars for R) and compare every
S3 mismatch cell of that endpoint with the app. Usage: exact_window_check.py <segment> <queue_id> <endpoint_iso|auto:<feature>>
[key=value config overrides ...]; auto picks the first and middle endpoints where <feature> mismatches."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from features.builder import FeatureDatasetBuilder  # noqa: E402
from features.config import FeatureBuilderConfig  # noqa: E402
from scripts import section4_parity_harness as H  # noqa: E402
from scripts.section4_parity_compare import CLASS_TOLERANCE, tolerance, within  # noqa: E402

segment, queue_id = sys.argv[1], int(sys.argv[2])
endpoints = [a for a in sys.argv[3:] if "=" not in a]
overrides = dict(a.split("=", 1) for a in sys.argv[3:] if "=" in a and not a.startswith("auto:"))
protocol = H.load_protocol()
model = H.roster_by_id()[queue_id]
seg = next(s for s in json.loads((H.RUN_ROOT / "windows.json").read_text(encoding="utf-8"))["segments"] if s["segment_id"] == segment)
group = next(g for g, spec in protocol["engine_composition"]["groups"].items() if queue_id in spec["queue_ids"])
if seg["source"] == "H":
    frame = H.training_frame(seg["stream"])
else:
    bars_ = H.live_bars(seg["stream"], end=seg["endpoints"][-1])
    frame = pd.DataFrame({"timestamp": pd.to_datetime([b.timestamp for b in bars_], utc=True), "open": [b.open for b in bars_],
                          "high": [b.high for b in bars_], "low": [b.low for b in bars_], "close": [b.close for b in bars_],
                          "volume": [b.volume for b in bars_], "contract_symbol": [b.contract_symbol for b in bars_]})
reference = H.FrozenReference(model)
raw_config = dict(H.dataset_metadata_for(reference.contract)[1]["config"])
raw_config.update(overrides)
config = FeatureBuilderConfig.from_dict(raw_config)
diff = pd.read_parquet(H.RUN_ROOT / seg["stream"] / group / segment / "diff" / f"{model['model_id']}.S3.parquet",
                       columns=["feature", "endpoint", "row_timestamp", "app", "tolerance_class"])
if endpoints and endpoints[0].startswith("auto:"):
    picked = sorted(diff.loc[diff.feature == endpoints[0][5:], "endpoint"].unique())
    endpoints = [picked[len(picked) // 2]]
out = {}
for endpoint in endpoints:
    pos = int(frame.timestamp.searchsorted(pd.Timestamp(endpoint)))
    bars = frame.iloc[pos - 8000 + 1:pos + 1].rename(columns={"timestamp": "datetime"}).reset_index(drop=True)
    if seg["stream"] == "EURUSD":
        bars = bars.drop(columns=["contract_symbol"])
    built, _ = FeatureDatasetBuilder(config).build(bars)
    built = built.set_index(pd.to_datetime(built["datetime"], utc=True))
    cells = diff.loc[diff.endpoint == endpoint]
    res = {}
    for feature, g in cells.groupby("feature"):
        atol, rtol = tolerance(protocol, CLASS_TOLERANCE.get(g.tolerance_class.iloc[0] or "C1_continuous", "C1_continuous"))
        v = built[feature].reindex(pd.to_datetime(g.row_timestamp, utc=True)).to_numpy(float)
        ok = within(pd.to_numeric(g.app, errors="coerce").to_numpy(float), v, atol, rtol)
        res[feature] = dict(cells=len(g), app_equals_exact_window=int(ok.sum()))
    out[endpoint] = res
    print(endpoint, res)
(H.RUN_ROOT / "classification" / f"exact_window_{segment}_{queue_id}.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
