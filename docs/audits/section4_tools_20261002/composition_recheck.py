"""Composition re-check on shared (endpoint, row_offset) keys: the group engine captures max(consumed, context+1)
rows over its models, a single-model engine only its own, so union keys differ by construction. Read-only."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from scripts.section4_parity_compare import _iso, _same, load_app  # noqa: E402

RUN = ROOT / "artifacts/s4parity_20261002_v1"
comp = json.loads((RUN / "composition_ES.json").read_text(encoding="utf-8"))
out = {}
for q, r in comp["results"].items():
    group = load_app(RUN / "ES" / r["group"] / "ES_R_rth_maint" / "app")
    single = load_app(RUN / "ES" / f"single_{int(q):02d}" / "ES_F_composition" / "app")
    def keyed(app):
        e = app["builds"].assign(e=app["builds"].endpoint_timestamp.map(_iso)).set_index("build_id").e
        rows = app["rows"].assign(endpoint=app["rows"].build_id.map(e))
        return rows.drop_duplicates(["endpoint", "row_offset"], keep="last").set_index(["endpoint", "row_offset"])
    a, b = keyed(group), keyed(single)
    keys = a.index.intersection(b.index)
    cols = [c for c in b.columns if c in a.columns and c not in {"build_id", "part", "generation_id", "instance", "window_position"}]
    diff = {}
    for c in cols:
        x, y = a.loc[keys, c], b.loc[keys, c]
        same = (np.isclose(x.to_numpy(float), y.to_numpy(float), rtol=0, atol=0, equal_nan=True)
                if pd.api.types.is_numeric_dtype(x) and pd.api.types.is_numeric_dtype(y) else _same(x.to_numpy(object), y.to_numpy(object)))
        if not same.all():
            diff[c] = int((~same).sum())
    wp = (a.loc[keys, "window_position"].to_numpy() - b.loc[keys, "window_position"].to_numpy())
    out[q] = dict(shared_keys=len(keys), group_only=len(a.index.difference(b.index).intersection(
        pd.MultiIndex.from_tuples([k for k in a.index if k[0] in set(b.index.get_level_values(0))]))),
        single_only=len(b.index.difference(a.index)), differing_columns=diff, window_position_offset=sorted(set(wp.tolist()))[:3])
    print(q, out[q]["shared_keys"], "group-only", out[q]["group_only"], "single-only", out[q]["single_only"], "differing", diff, "wpos offset", out[q]["window_position_offset"])
(RUN / "classification" / "composition_recheck_ES.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
