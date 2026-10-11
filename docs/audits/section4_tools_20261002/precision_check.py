"""Protocol v3 float32_tree_split check. For every segment x model: S6 mismatch endpoints with no S3/S4/S4_mask/S5
mismatch at that endpoint are candidates; each is verified by rebuilding the model input from each side's recorded
S5 (``sc:``) rows and predicting with the hash-verified frozen model (XGB). Writes classification/precision_check.json."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from ote_live.models.runners import _build_sparse_lag_view, _sigmoid  # noqa: E402
from scripts.run_section4_r1_value_neutrality import staged_model  # noqa: E402
from scripts.section4_parity_compare import load_app  # noqa: E402
from scripts.section4_parity_harness import RUN_ROOT, roster_by_id  # noqa: E402

WINDOWS = {s["segment_id"] for s in json.loads((RUN_ROOT / "windows.json").read_text(encoding="utf-8"))["segments"]}
models = {m["model_id"]: m for m in roster_by_id().values()}
staged, out = {}, {}
for summary_path in sorted(RUN_ROOT.glob("*/*/*/summary.json")):
    seg_dir = summary_path.parent
    if seg_dir.name not in WINDOWS:
        continue
    for s6 in sorted((seg_dir / "diff").glob("*.S6.parquet")):
        model_id = s6.name[: -len(".S6.parquet")]
        upstream = set()
        for stage in ("S3", "S4", "S4_mask", "S5"):
            path = seg_dir / "diff" / f"{model_id}.{stage}.parquet"
            if path.exists():
                upstream |= set(pd.read_parquet(path, columns=["endpoint"]).endpoint)
        candidates = sorted(set(pd.read_parquet(s6, columns=["endpoint"]).endpoint) - upstream)
        if not candidates:
            continue
        model = models[model_id]
        result = dict(candidates=candidates, verified=[], not_verified=[], backend=None)
        if model_id not in staged:
            staged[model_id] = staged_model(model)[0]
        runtime = staged[model_id]
        result["backend"] = runtime.backend
        if runtime.backend != "xgboost":
            result["not_verified"] = candidates
        else:
            fm, names = runtime.manifest.feature_manifest, list(runtime.selected_feature_names)
            app = load_app(seg_dir / "app")
            aio, rio = app["io"][model_id], pd.read_parquet(seg_dir / "ref" / f"{model_id}.io.parquet")
            ascores, rscores = app["scores"], pd.read_parquet(seg_dir / "ref" / f"{model_id}.scores.parquet")
            import xgboost as xgb
            for endpoint in candidates:
                ok = True
                for io, scores in ((aio, ascores.loc[ascores.model_id == model_id]), (rio, rscores)):
                    rows = io.loc[io.endpoint == endpoint].drop_duplicates("row_offset", keep="last").sort_values("row_offset")
                    view = _build_sparse_lag_view(rows[["sc:" + n for n in names]].to_numpy(np.float32),
                                                  lag_steps=fm.lag_steps or [0], delta_feature_count=fm.delta_feature_count)
                    score = float(_sigmoid(np.asarray(runtime.model.predict(xgb.DMatrix(view[-1:])), dtype=np.float32))[0])
                    ok &= score == float(np.float32(scores.loc[scores.endpoint == endpoint].raw_score.iloc[-1]))
                (result["verified"] if ok else result["not_verified"]).append(endpoint)
        out[f"{seg_dir.name}|{seg_dir.parent.name}|{model_id}"] = result
        print(seg_dir.name, model_id, "candidates", len(candidates), "verified", len(result["verified"]), "not", len(result["not_verified"]))
(RUN_ROOT / "classification" / "precision_check.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
