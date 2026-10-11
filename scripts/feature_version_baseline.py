"""Feature-version guard: hash the raw (pre-fill) builder output of the five frozen B2 dataset configs on fixed
windows, so a producer change can prove v1 stays byte-identical.
Usage: python scripts/feature_version_baseline.py <out.json> [key=value config overrides ...]
       python scripts/feature_version_baseline.py --compare <baseline.json> <candidate.json> [<baseline_repeat.json>]
With a repeat of the baseline, columns that already differ between two identical baseline runs (non-deterministic
producers, e.g. strategies that draw random numbers) are reported and ignored.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CASES = (  # (dataset dir, stream, end bar, bars)
    ("artifacts/b2_ict_prepared_20260921_v1", "ES", "2026-06-16T13:30:00+00:00", 3000),
    ("artifacts/b2_ict_specialists_prepared_20260921_v1", "ES", "2026-03-09T06:00:00+00:00", 3000),
    ("artifacts/b2_frvp_july_prepared_20260921_v1", "ES", "2025-11-03T07:00:00+00:00", 3000),
    ("artifacts/b2_frvp_setups_prepared_20260921_v1", "ES", "2026-06-16T13:30:00+00:00", 3000),
    ("artifacts/b2_ote_prepared_20260921_v1", "EURUSD", "2026-03-10T12:00:00+00:00", 1500),
)


def column_hash(series: pd.Series) -> str:
    return hashlib.sha256(pd.util.hash_pandas_object(series, index=False).to_numpy().tobytes()).hexdigest()


def build(overrides: dict) -> dict:
    from features.config import FeatureBuilderConfig
    from scripts.audit_feature_causality import raw_build
    from scripts.section4_parity_harness import training_frame

    out = {}
    for dataset, stream, end, n in CASES:
        meta = json.loads((ROOT / dataset / "dataset.metadata.json").read_text(encoding="utf-8"))
        raw = dict(meta["config"])
        raw.update(overrides)
        source = training_frame(stream)
        pos = int(source.timestamp.searchsorted(pd.Timestamp(end)))
        bars = source.iloc[pos - n + 1:pos + 1].rename(columns={"timestamp": "datetime"}).reset_index(drop=True)
        if stream == "EURUSD":
            bars = bars.drop(columns=["contract_symbol"])
        frame, producers = raw_build(FeatureBuilderConfig.from_dict(raw), bars)
        out[f"{Path(dataset).name}@{end}"] = dict(columns=list(frame.columns),
                                                  hashes={c: column_hash(frame[c]) for c in frame.columns},
                                                  producers=producers)
        print(Path(dataset).name, end, len(frame.columns), "columns", flush=True)
    return out


def main() -> int:
    if sys.argv[1] == "--compare":
        a, b = (json.loads(Path(p).read_text(encoding="utf-8")) for p in sys.argv[2:4])
        noisy = {}
        if len(sys.argv) > 4:
            r = json.loads(Path(sys.argv[4]).read_text(encoding="utf-8"))
            noisy = {case: {c for c in a[case]["columns"] if a[case]["hashes"][c] != r[case]["hashes"].get(c)} for case in a}
        bad = 0
        for case in a:
            ca, cb = a[case], b.get(case)
            if cb is None or ca["columns"] != cb["columns"]:
                print("COLUMNS DIFFER", case, sorted(set(ca["columns"]) ^ set((cb or {}).get("columns", []))))
                bad += 1
                continue
            diff = [c for c in ca["columns"] if ca["hashes"][c] != cb["hashes"][c] and c not in noisy.get(case, set())]
            ignored = len(noisy.get(case, ()))
            print(case, ("identical" if not diff else f"DIFFERS in {len(diff)}: {diff[:8]}")
                  + (f" (ignored {ignored} non-deterministic baseline columns)" if ignored else ""))
            bad += bool(diff)
        print("BASELINE IDENTICAL" if not bad else "BASELINE DIFFERS")
        return 1 if bad else 0
    overrides = dict(a.split("=", 1) for a in sys.argv[2:])
    result = dict(overrides=overrides, cases=None)
    cases = build({k: (int(v) if v.isdigit() else v) for k, v in overrides.items()})
    Path(sys.argv[1]).write_text(json.dumps(cases, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
