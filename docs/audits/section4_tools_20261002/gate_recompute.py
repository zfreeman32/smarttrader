"""Prompt 4 classification aid (run only when no matrix job is running): are R-window S3 mismatches the HTF
completion gate? Recomputes each model's frozen dataset config on the same recorded series with the recorded
``is_complete`` column (as ``runtime_state`` hands it to the builder) and tests app ~ gated recompute on every
S3 mismatch cell. Writes classification/gate_test_<segment>.json; prints per-model totals only."""
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

RUN = H.RUN_ROOT
protocol = H.load_protocol()
stream, segment = sys.argv[1], sys.argv[2]
seg = next(s for s in json.loads((RUN / "windows.json").read_text(encoding="utf-8"))["segments"] if s["segment_id"] == segment)
bars = H.live_bars(stream, end=seg["endpoints"][-1])
market = pd.DataFrame({"datetime": pd.to_datetime([b.timestamp for b in bars], utc=True), "open": [b.open for b in bars],
                       "high": [b.high for b in bars], "low": [b.low for b in bars], "close": [b.close for b in bars],
                       "volume": [b.volume for b in bars], "is_complete": [b.is_complete for b in bars]})
if stream == "ES":
    market["contract_symbol"] = [b.contract_symbol for b in bars]
roster = {m["model_id"]: m for m in H.roster_by_id().values()}
cache, out = {}, {}
for group_dir in sorted((RUN / stream).glob(f"*/{segment}")):
    for diff in sorted((group_dir / "diff").glob("*.S3.parquet")):
        model_id = diff.name[: -len(".S3.parquet")]
        reference = H.FrozenReference(roster[model_id])
        meta_path, meta = H.dataset_metadata_for(reference.contract)
        if meta_path not in cache:
            built, _ = FeatureDatasetBuilder(FeatureBuilderConfig.from_dict(meta["config"])).build(market.copy())
            cache = {meta_path: built.set_index(pd.to_datetime(built["datetime"], utc=True))}
        built = cache[meta_path]
        d = pd.read_parquet(diff, columns=["feature", "row_timestamp", "app", "tolerance_class"])
        explained = 0
        per = {}
        for feature, g in d.groupby("feature"):
            atol, rtol = tolerance(protocol, CLASS_TOLERANCE.get(g.tolerance_class.iloc[0] or "C1_continuous", "C1_continuous"))
            gated = built[feature].reindex(pd.to_datetime(g.row_timestamp, utc=True)).to_numpy(float)
            ok = within(pd.to_numeric(g.app, errors="coerce").to_numpy(float), gated, atol, rtol)
            per[feature] = dict(cells=len(g), gate_explained=int(ok.sum()))
            explained += int(ok.sum())
        out[model_id] = dict(cells=len(d), gate_explained=explained, features=per)
        print(f"{group_dir.parent.name} {model_id}: cells {len(d)} gate-explained {explained}")
(RUN / "classification" / f"gate_test_{segment}.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
