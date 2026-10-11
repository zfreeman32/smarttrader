"""Prompt 4 classification aid: does a frozen-config variant (key=value overrides) reproduce the app?
Recomputes each model's frozen dataset config on the segment's app-span bars (H: training rows from the prefix
start; R: everything recorded) with the overrides, and tests app ~ variant on every S3 mismatch cell.
Usage: variant_recompute.py <stream> <segment> <tag> key=value [...]. Writes classification/variant_<tag>_<segment>.json."""
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from features.builder import FeatureDatasetBuilder  # noqa: E402
from features.config import FeatureBuilderConfig  # noqa: E402
from scripts import section4_parity_harness as H  # noqa: E402
from scripts.section4_parity_compare import CLASS_TOLERANCE, tolerance, within  # noqa: E402

stream, segment, tag = sys.argv[1:4]
overrides = dict(arg.split("=", 1) for arg in sys.argv[4:])
protocol = H.load_protocol()
seg = next(s for s in json.loads((H.RUN_ROOT / "windows.json").read_text(encoding="utf-8"))["segments"] if s["segment_id"] == segment)
if seg["source"] == "H":
    frame = H.training_frame(stream).iloc[seg["prefix_start_pos"]:seg["last_endpoint_pos"] + 1]
    market = frame.rename(columns={"timestamp": "datetime"})
else:
    bars = H.live_bars(stream, end=seg["endpoints"][-1])
    market = pd.DataFrame({"datetime": pd.to_datetime([b.timestamp for b in bars], utc=True), "open": [b.open for b in bars],
                           "high": [b.high for b in bars], "low": [b.low for b in bars], "close": [b.close for b in bars],
                           "volume": [b.volume for b in bars], "contract_symbol": [b.contract_symbol for b in bars]})
if stream == "EURUSD":
    market = market.drop(columns=["contract_symbol"], errors="ignore")
market = market.reset_index(drop=True)
roster = {m["model_id"]: m for m in H.roster_by_id().values()}
cache, out = {}, {}
for group_dir in sorted((H.RUN_ROOT / stream).glob(f"*/{segment}")):
    for diff in sorted((group_dir / "diff").glob("*.S3.parquet")):
        model_id = diff.name[: -len(".S3.parquet")]
        reference = H.FrozenReference(roster[model_id])
        meta_path, meta = H.dataset_metadata_for(reference.contract)
        if meta_path not in cache:
            raw = dict(meta["config"])
            for key, value in overrides.items():
                raw[key] = None if value == "None" else value
            built, _ = FeatureDatasetBuilder(FeatureBuilderConfig.from_dict(raw)).build(market.copy())
            cache = {meta_path: built.set_index(pd.to_datetime(built["datetime"], utc=True))}
        built = cache[meta_path]
        d = pd.read_parquet(diff, columns=["feature", "row_timestamp", "app", "tolerance_class"])
        per, explained = {}, 0
        for feature, g in d.groupby("feature"):
            atol, rtol = tolerance(protocol, CLASS_TOLERANCE.get(g.tolerance_class.iloc[0] or "C1_continuous", "C1_continuous"))
            v = built[feature].reindex(pd.to_datetime(g.row_timestamp, utc=True)).to_numpy(float)
            ok = within(pd.to_numeric(g.app, errors="coerce").to_numpy(float), v, atol, rtol)
            per[feature] = dict(cells=len(g), explained=int(ok.sum()))
            explained += int(ok.sum())
        out[model_id] = dict(cells=len(d), explained=explained, features=per)
        print(f"{model_id}: cells {len(d)} variant-explained {explained}")
(H.RUN_ROOT / "classification" / f"variant_{tag}_{segment}.json").write_text(json.dumps(dict(overrides=overrides, models=out), indent=1), encoding="utf-8")
