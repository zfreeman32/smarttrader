from __future__ import annotations

import argparse
import pickle
from pathlib import Path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a single standalone strategy in a subprocess")
    parser.add_argument("--context", required=True, help="Path to the serialized strategy execution context")
    parser.add_argument("--strategy-name", required=True, help="Requested strategy id or alias")
    parser.add_argument("--output", required=True, help="Path to the serialized strategy result")
    parser.add_argument("--strategies-dir", required=True, help="Directory containing strategy modules")
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    from features.feature_sets.strategy_signals import _run_strategy_job_core
    from features.strategy_registry import StrategyRegistry

    context_path = Path(args.context)
    output_path = Path(args.output)

    with context_path.open("rb") as handle:
        context = pickle.load(handle)

    registry = StrategyRegistry(Path(args.strategies_dir))
    result = _run_strategy_job_core(
        requested_name=args.strategy_name,
        strategy_input=context["strategy_input"],
        target_index=context["target_index"],
        input_columns=set(context["input_columns"]),
        registry=registry,
    )

    with output_path.open("wb") as handle:
        pickle.dump(result, handle)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
