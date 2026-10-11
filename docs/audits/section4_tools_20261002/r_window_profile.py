"""Prompt 4 classification aid: S3 mismatch profile per producer on a segment (read-only, summaries only).

Per model and producer: cells, app NaN vs ref finite, ref NaN vs app finite, both finite; completion and
observation kind of the mismatching rows; first row timestamp."""
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
RUN = ROOT / "artifacts/s4parity_20261002_v1"
stream, segment = sys.argv[1], sys.argv[2]
out = {}
for group_dir in sorted((RUN / stream).glob(f"*/{segment}")):
    summary = json.loads((group_dir / "summary.json").read_text(encoding="utf-8"))
    for model, m in summary["models"].items():
        producers = {g["feature"]: g["producer"] for g in m["mismatch_groups"] if g["stage"] == "S3"}
        path = group_dir / "diff" / f"{model}.S3.parquet"
        if not path.exists():
            continue
        d = pd.read_parquet(path, columns=["feature", "row_timestamp", "app", "ref", "store_is_complete", "store_observation_kind"])
        d["producer"] = d.feature.map(producers).fillna("?")
        a, r = pd.to_numeric(d.app, errors="coerce"), pd.to_numeric(d.ref, errors="coerce")
        d["kind"] = (a.isna() & r.notna()).map({True: "app_nan"}).fillna((r.isna() & a.notna()).map({True: "ref_nan"}).fillna("both_finite"))
        prof = d.groupby(["producer", "kind"]).size().unstack(fill_value=0)
        prof["incomplete_rows"] = d.assign(x=d.store_is_complete.astype(str) != "True").groupby("producer").x.sum()
        prof["first_row"] = d.groupby("producer").row_timestamp.min()
        out[model] = json.loads(prof.reset_index().to_json(orient="records"))
        print(group_dir.parent.name, model); print(prof.to_string())
(RUN / "classification").mkdir(exist_ok=True)
(RUN / "classification" / f"s3_profile_{segment}.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
