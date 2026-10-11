"""Evaluate frozen B1 inputs, with no training, relabeling or threshold fitting."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from model_testing.outcome_artifacts import run_outcome_evaluation, replay_outcome_run
from model_testing.outcome_contracts import CATALOG, PositionContract, file_hash
from model_testing.outcome_engine import EvaluationTape, SignalEvidence


def run_input_manifest(path: str | Path, *, output_dir: str | Path) -> dict:
    """Shared Phase 7 entrypoint for frozen B1 inputs (paths relative to manifest)."""
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if set(payload) - {"signals", "bars", "schedule", "calendar", "finer_bars", "position_contract", "catalog", "structural_atr_mode"}:
        raise ValueError("Unknown outcome-input manifest fields")
    atr_mode = payload.pop("structural_atr_mode", "supplied_evidence")
    if atr_mode not in {"supplied_evidence", "derive_from_tape"}:
        raise ValueError("Unknown structural_atr_mode")
    paths = {key: (path.parent / value).resolve() for key, value in payload.items()}
    for required in ("signals", "bars", "schedule", "calendar"):
        if required not in paths:
            raise ValueError(f"Missing B1 input: {required}")
    paths.setdefault("catalog", CATALOG)
    calendar = json.loads(paths["calendar"].read_text(encoding="utf-8"))
    calendar.pop("calendar_hash", None)
    tape = EvaluationTape(pd.read_csv(paths["bars"]), pd.read_csv(paths["schedule"]), **calendar,
        finer_bars=pd.read_csv(paths["finer_bars"]) if "finer_bars" in paths else None)
    signals = [SignalEvidence.model_validate_json(line)
        for line in paths["signals"].read_text(encoding="utf-8").splitlines() if line.strip()]
    simulator = (PositionContract.model_validate_json(paths["position_contract"].read_text(encoding="utf-8"))
                 if "position_contract" in paths else None)
    lineage_paths = [path, *paths.values(), paths["catalog"].with_suffix(".lock.json")]
    return run_outcome_evaluation(signals, tape, output_dir=output_dir, simulator=simulator,
        derive_structural_atr=atr_mode == "derive_from_tape",
        catalog_path=paths["catalog"], input_lineage={str(p): file_hash(p) for p in lineage_paths})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-manifest", type=Path, help="Manifest of frozen B1 input paths")
    parser.add_argument("--replay-run", type=Path, help="Verify and replay a saved B1 run")
    parser.add_argument("--signals", type=Path, help="SignalEvidence JSONL; timestamps must have timezone offsets")
    parser.add_argument("--bars", type=Path, help="Unfilled open_ts/open/high/low/close CSV; futures require coordinate_id")
    parser.add_argument("--schedule", type=Path, help="Independently sourced expected open_ts/session_id CSV")
    parser.add_argument("--calendar", type=Path, help="Calendar ID/source, UTC coverage bounds, timeframe_minutes JSON")
    parser.add_argument("--finer-bars", type=Path)
    parser.add_argument("--position-contract", type=Path, help="Optional predeclared PositionContract; no default portfolio policy")
    parser.add_argument("--catalog", type=Path, default=CATALOG)
    parser.add_argument("--derive-structural-atr", action="store_true", help="Compute the declared completed 1h/30m Wilder ATR14 from raw tape and schedule; requires warmup history")
    parser.add_argument("--output-dir", type=Path, required=True, help="Must not exist")
    args = parser.parse_args(argv)
    if args.input_manifest is not None:
        if args.derive_structural_atr:
            parser.error("Specify structural_atr_mode in the input manifest")
        if args.replay_run is not None or any(value is not None for value in (args.signals, args.bars, args.schedule, args.calendar, args.position_contract, args.finer_bars)):
            parser.error("Input manifest cannot be combined with other input modes")
        print(json.dumps(run_input_manifest(args.input_manifest, output_dir=args.output_dir), indent=2))
        return 0
    if args.replay_run is not None:
        if args.derive_structural_atr:
            parser.error("Replay preserves the saved ATR mode")
        if any(value is not None for value in (args.signals, args.bars, args.schedule, args.calendar, args.position_contract, args.finer_bars)):
            parser.error("Replay uses saved inputs exclusively")
        print(json.dumps(replay_outcome_run(args.replay_run, args.output_dir), indent=2))
        return 0
    if any(value is None for value in (args.signals, args.bars, args.schedule, args.calendar)):
        parser.error("--signals, --bars, --schedule and --calendar are required for a new run")
    signals = [SignalEvidence.model_validate_json(line) for line in args.signals.read_text(encoding="utf-8").splitlines() if line.strip()]
    calendar = json.loads(args.calendar.read_text(encoding="utf-8"))
    calendar.pop("calendar_hash", None)
    tape = EvaluationTape(pd.read_csv(args.bars), pd.read_csv(args.schedule), **calendar,
                          finer_bars=None if args.finer_bars is None else pd.read_csv(args.finer_bars))
    simulator = None if args.position_contract is None else PositionContract.model_validate_json(args.position_contract.read_text(encoding="utf-8"))
    paths = [args.signals, args.bars, args.schedule, args.calendar, args.catalog, args.catalog.with_suffix(".lock.json")]
    paths += [p for p in (args.finer_bars, args.position_contract) if p is not None]
    result = run_outcome_evaluation(signals, tape, output_dir=args.output_dir, simulator=simulator,
        derive_structural_atr=args.derive_structural_atr,
        catalog_path=args.catalog, input_lineage={str(p): file_hash(p) for p in paths})
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
