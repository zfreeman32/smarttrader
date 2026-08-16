from __future__ import annotations

from typing import Any, Mapping

from model_testing.promotion_gates import accepted_for_paper_trading, drawdown_acceptance_passed

PROMOTION_QUALITY_MODE = "promotion_quality"
RESEARCH_MODE = "research"
DEFAULT_PROMOTION_MIN_TRADES_PER_WEEK_FLOOR = 3.0


def build_evaluation_contract(
    *,
    evaluation_contract_mode: str = PROMOTION_QUALITY_MODE,
    min_trades_per_week: float,
    promotion_min_trades_per_week_floor: float = DEFAULT_PROMOTION_MIN_TRADES_PER_WEEK_FLOOR,
    requested_min_folds: int | None = None,
    available_min_folds: int | None = None,
    effective_min_folds: int | None = None,
) -> dict[str, Any]:
    mode = _normalize_evaluation_contract_mode(evaluation_contract_mode)
    requested = None if requested_min_folds is None else int(requested_min_folds)
    available = None if available_min_folds is None else int(available_min_folds)
    effective = None if effective_min_folds is None else int(effective_min_folds)
    min_trades = float(min_trades_per_week)
    promotion_floor = float(promotion_min_trades_per_week_floor)

    used_low_frequency_override = bool(min_trades < promotion_floor)
    used_auto_relaxed_min_folds = bool(
        requested is not None and effective is not None and effective < requested
    )

    disqualifiers: list[str] = []
    if mode != PROMOTION_QUALITY_MODE:
        disqualifiers.append("research_mode")
    if used_low_frequency_override:
        disqualifiers.append("min_trades_per_week_below_promotion_floor")
    if used_auto_relaxed_min_folds:
        disqualifiers.append("auto_relaxed_min_folds")

    return {
        "evaluation_contract_mode": mode,
        "promotion_quality_run": mode == PROMOTION_QUALITY_MODE,
        "promotion_min_trades_per_week_floor": promotion_floor,
        "min_trades_per_week": min_trades,
        "requested_min_folds": requested,
        "available_min_folds": available,
        "effective_min_folds": effective,
        "used_low_frequency_override": used_low_frequency_override,
        "used_auto_relaxed_min_folds": used_auto_relaxed_min_folds,
        "promotion_quality_disqualifiers": disqualifiers,
        "promotion_quality_gate_eligible": not disqualifiers,
    }


def build_paper_trading_gate(
    acceptance: Mapping[str, Any] | None,
    *,
    evaluation_contract: Mapping[str, Any],
) -> dict[str, Any]:
    accepted_raw = accepted_for_paper_trading(acceptance)
    return {
        "accepted": bool(accepted_raw and bool(evaluation_contract.get("promotion_quality_gate_eligible", True))),
        "accepted_raw": bool(accepted_raw),
        "promotion_quality_gate_eligible": bool(
            evaluation_contract.get("promotion_quality_gate_eligible", True)
        ),
        "promotion_quality_disqualifiers": [
            str(value)
            for value in evaluation_contract.get("promotion_quality_disqualifiers", [])
        ],
        "drawdown_gate_passed": drawdown_acceptance_passed(acceptance),
        "drawdown_gate_is_advisory": True,
    }


def resolve_saved_paper_trading_gate(
    model_output: Mapping[str, Any] | None,
    *,
    run_summary: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    output_payload = model_output if isinstance(model_output, Mapping) else {}
    saved_gate = output_payload.get("paper_trading_gate")
    if isinstance(saved_gate, Mapping) and "accepted_raw" in saved_gate:
        return {
            "accepted": bool(saved_gate.get("accepted")),
            "accepted_raw": bool(saved_gate.get("accepted_raw")),
            "promotion_quality_gate_eligible": bool(
                saved_gate.get("promotion_quality_gate_eligible", True)
            ),
            "promotion_quality_disqualifiers": [
                str(value)
                for value in saved_gate.get("promotion_quality_disqualifiers", [])
            ],
            "drawdown_gate_passed": bool(saved_gate.get("drawdown_gate_passed", False)),
            "drawdown_gate_is_advisory": bool(saved_gate.get("drawdown_gate_is_advisory", True)),
        }

    evaluation_contract = _resolve_saved_evaluation_contract(output_payload, run_summary=run_summary)
    return build_paper_trading_gate(
        output_payload.get("acceptance"),
        evaluation_contract=evaluation_contract,
    )


def _resolve_saved_evaluation_contract(
    model_output: Mapping[str, Any],
    *,
    run_summary: Mapping[str, Any] | None,
) -> dict[str, Any]:
    output_contract = model_output.get("evaluation_contract")
    if isinstance(output_contract, Mapping):
        return dict(output_contract)

    if isinstance(run_summary, Mapping):
        run_contract = run_summary.get("evaluation_contract")
        if isinstance(run_contract, Mapping):
            return dict(run_contract)

        min_trades = float(
            run_summary.get(
                "min_trades_per_week",
                DEFAULT_PROMOTION_MIN_TRADES_PER_WEEK_FLOOR,
            )
        )
        effective_min_folds = _coerce_optional_int(run_summary.get("min_folds"))
        return build_evaluation_contract(
            evaluation_contract_mode=(
                RESEARCH_MODE
                if min_trades < DEFAULT_PROMOTION_MIN_TRADES_PER_WEEK_FLOOR
                else PROMOTION_QUALITY_MODE
            ),
            min_trades_per_week=min_trades,
            promotion_min_trades_per_week_floor=DEFAULT_PROMOTION_MIN_TRADES_PER_WEEK_FLOOR,
            requested_min_folds=_coerce_optional_int(run_summary.get("requested_min_folds")),
            available_min_folds=_coerce_optional_int(run_summary.get("available_min_folds")),
            effective_min_folds=effective_min_folds,
        )

    return build_evaluation_contract(
        min_trades_per_week=DEFAULT_PROMOTION_MIN_TRADES_PER_WEEK_FLOOR,
    )


def _normalize_evaluation_contract_mode(value: str | None) -> str:
    mode = str(value or PROMOTION_QUALITY_MODE).strip().lower()
    if mode not in {PROMOTION_QUALITY_MODE, RESEARCH_MODE}:
        raise ValueError(
            f"Unsupported evaluation contract mode {value!r}. "
            f"Expected one of: {PROMOTION_QUALITY_MODE}, {RESEARCH_MODE}."
        )
    return mode


def _coerce_optional_int(value: object) -> int | None:
    if value in {None, ""}:
        return None
    return int(value)
