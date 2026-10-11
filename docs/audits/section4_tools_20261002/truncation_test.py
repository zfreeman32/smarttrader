"""Prompt 4 classification aid: is an app-vs-frozen S3 mismatch explained by history span (window truncation)?

For each S3 mismatch cell on an H segment: if the anchor (recompute on the app's 8,000-bar span) differs from
frozen at that cell, test app ~ recompute (same tolerance class); otherwise recompute == frozen there, so the app
differs from the offline code on identical bars. Read-only; prints per-model/feature summaries only."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from scripts.section4_parity_compare import CLASS_TOLERANCE, tolerance, within  # noqa: E402

RUN = ROOT / "artifacts/s4parity_20261002_v1"
protocol = json.loads((ROOT / "configs/section4_parity_protocol_v2.json").read_text(encoding="utf-8"))
stream, segment = sys.argv[1], sys.argv[2]
out = {}
for anchor_dir in sorted((RUN / "anchor" / stream).glob(f"*/{segment}")):
    group = anchor_dir.parent.name
    seg = RUN / stream / group / segment
    for diff in sorted((seg / "diff").glob("*.S3.parquet")):
        model = diff.name[: -len(".S3.parquet")]
        app = pd.read_parquet(diff, columns=["feature", "row_timestamp", "app", "ref", "tolerance_class"])
        path = anchor_dir / f"{model}.anchor_diff.parquet"
        anchor = pd.read_parquet(path) if path.exists() else pd.DataFrame(columns=["feature", "datetime", "recompute", "frozen"])
        merged = app.merge(anchor.rename(columns={"datetime": "row_timestamp"}), on=["feature", "row_timestamp"], how="left")
        rows = {}
        for feature, g in merged.groupby("feature"):
            cls = g.tolerance_class.iloc[0] or "C1_continuous"
            atol, rtol = tolerance(protocol, CLASS_TOLERANCE.get(cls, "C1_continuous"))
            in_anchor = g.recompute.notna() | g.frozen.notna()
            appv = pd.to_numeric(g.app, errors="coerce").to_numpy(float)
            eq_rec = in_anchor.to_numpy() & within(appv, pd.to_numeric(g.recompute, errors="coerce").to_numpy(float), atol, rtol)
            rows[feature] = dict(cells=len(g), anchor_explained=int(eq_rec.sum()), anchor_diff_but_app_differs=int((in_anchor & ~eq_rec).sum()),
                                 recompute_equals_frozen=int((~in_anchor).sum()))
        out[model] = rows
        tot = pd.DataFrame(rows).T.sum()
        print(f"{group} {model}: cells {int(tot.cells)} truncation-explained {int(tot.anchor_explained)} "
              f"anchor-diff-app-differs {int(tot.anchor_diff_but_app_differs)} recompute==frozen {int(tot.recompute_equals_frozen)}")
(RUN / "classification").mkdir(exist_ok=True)
(RUN / "classification" / f"truncation_test_{segment}.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
