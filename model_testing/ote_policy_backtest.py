from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from model_testing.ote_abstain_policy import HardAbstainConfig, apply_abstain_policy
from model_testing.ote_policy_metrics import (
    build_equity_curve,
    build_group_breakdown,
    compute_buy_hold_period_returns,
    sanitize_for_json,
    summarize_trade_performance,
    summarize_walk_forward_efficiency,
)
from model_testing.ote_threshold_policy import (
    ThresholdSearchConfig,
    apply_threshold_policy,
    attach_forward_trade_outcomes,
    evaluate_policy_variants,
    prepare_policy_frame,
    search_thresholds_by_composite_regime,
)


@dataclass(frozen=True)
class WalkForwardFold:
    fold_id: str
    fold_index: int
    scheduled_test_start: pd.Timestamp
    scheduled_test_end: pd.Timestamp
    train_position_end: int
    test_position_start: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    train_rows: int
    test_rows: int
    purge_gap_bars: int


@dataclass(frozen=True)
class WalkForwardBacktestConfig:
    min_train_years: int = 2
    test_window_months: int = 3
    rolling_step_months: int = 3
    purge_gap_bars: int = 40
    min_folds: int = 8
    max_folds: int | None = None
    min_trades_per_week: float = 3.0
    minimum_wfe: float = 0.50
    max_single_trade_share: float = 0.10
    minimum_profitable_quarter_share: float = 0.60
    minimum_positive_composite_share: float = 0.60
    confidence_bucket_count: int = 5
    group_columns: tuple[str, ...] = field(
        default_factory=lambda: (
            "calendar_year",
            "calendar_quarter",
            "session_regime",
            "composite_regime",
            "confidence_quintile",
        )
    )


def build_walk_forward_folds(
    frame: pd.DataFrame,
    *,
    config: WalkForwardBacktestConfig,
    datetime_column: str = "datetime",
    position_column: str = "source_row_idx",
) -> list[WalkForwardFold]:
    if frame.empty:
        return []
    if datetime_column not in frame.columns or position_column not in frame.columns:
        raise ValueError(
            f"Walk-forward fold generation requires {datetime_column!r} and {position_column!r} columns."
        )

    working = frame.loc[:, [datetime_column, position_column]].copy()
    working[datetime_column] = pd.to_datetime(working[datetime_column], errors="coerce")
    working[position_column] = pd.to_numeric(working[position_column], errors="coerce")
    working = working.dropna(subset=[datetime_column, position_column]).sort_values(datetime_column).reset_index(drop=True)
    if working.empty:
        return []

    earliest = pd.Timestamp(working[datetime_column].min())
    latest = pd.Timestamp(working[datetime_column].max())
    first_candidate = earliest + pd.DateOffset(years=int(config.min_train_years))
    current_start = _next_quarter_start(first_candidate)

    folds: list[WalkForwardFold] = []
    fold_index = 1
    while current_start <= latest:
        scheduled_end = current_start + pd.DateOffset(months=int(config.test_window_months))
        test_mask = (working[datetime_column] >= current_start) & (working[datetime_column] < scheduled_end)
        if bool(test_mask.any()):
            test_rows = working.loc[test_mask].copy()
            min_test_position = int(test_rows[position_column].min())
            train_position_end = min_test_position - int(config.purge_gap_bars)
            train_mask = (working[datetime_column] < current_start) & (working[position_column] <= train_position_end)
            if bool(train_mask.any()):
                train_rows = working.loc[train_mask].copy()
                train_start = pd.Timestamp(train_rows[datetime_column].min())
                train_end = pd.Timestamp(train_rows[datetime_column].max())
                train_years = (train_end - train_start).total_seconds() / (365.25 * 86400.0)
                if train_years >= float(config.min_train_years):
                    folds.append(
                        WalkForwardFold(
                            fold_id=f"fold_{fold_index:02d}",
                            fold_index=fold_index,
                            scheduled_test_start=current_start,
                            scheduled_test_end=scheduled_end,
                            train_position_end=int(train_position_end),
                            test_position_start=min_test_position,
                            train_start=train_start,
                            train_end=train_end,
                            test_start=pd.Timestamp(test_rows[datetime_column].min()),
                            test_end=pd.Timestamp(test_rows[datetime_column].max()),
                            train_rows=int(len(train_rows)),
                            test_rows=int(len(test_rows)),
                            purge_gap_bars=int(config.purge_gap_bars),
                        )
                    )
                    fold_index += 1
                    if config.max_folds is not None and len(folds) >= int(config.max_folds):
                        break
        current_start = current_start + pd.DateOffset(months=int(config.rolling_step_months))
        if config.max_folds is not None and len(folds) >= int(config.max_folds):
            break

    return folds


def run_walk_forward_backtest(
    frame: pd.DataFrame,
    *,
    market_frame: pd.DataFrame,
    direction: str,
    threshold_config: ThresholdSearchConfig,
    backtest_config: WalkForwardBacktestConfig,
    abstain_config: HardAbstainConfig | None = None,
    model_id: str | None = None,
    backend: str | None = None,
) -> dict[str, Any]:
    prepared = prepare_policy_frame(frame, threshold_config)
    if prepared.empty:
        raise ValueError("Walk-forward backtest requires a non-empty prepared frame.")

    if threshold_config.net_pnl_column not in prepared.columns or threshold_config.gross_pnl_column not in prepared.columns:
        prepared = attach_forward_trade_outcomes(
            prepared,
            market_frame=market_frame,
            direction=direction,
            config=threshold_config,
        )

    folds = build_walk_forward_folds(
        prepared,
        config=backtest_config,
        datetime_column=threshold_config.datetime_column,
        position_column=threshold_config.position_column,
    )
    if len(folds) < int(backtest_config.min_folds):
        raise ValueError(
            f"Walk-forward backtest only produced {len(folds)} folds; need at least {backtest_config.min_folds}."
        )

    benchmark_monthly_returns = compute_buy_hold_period_returns(
        market_frame,
        datetime_column=threshold_config.datetime_column,
        price_column=threshold_config.entry_price_column,
        pip_size=threshold_config.pip_size,
        freq="M",
    )
    abstain_policy = abstain_config or HardAbstainConfig(
        cooldown_bars=threshold_config.event_cooldown_bars,
        probability_column="policy_probability",
        signal_candidate_column="policy_signal_candidate",
        position_column=threshold_config.position_column,
    )

    policy_tables: list[pd.DataFrame] = []
    policy_evaluations: list[pd.DataFrame] = []
    train_trades: list[pd.DataFrame] = []
    test_trades: list[pd.DataFrame] = []
    fold_rows: list[dict[str, Any]] = []

    for fold in folds:
        train_frame = _slice_train_frame(prepared, fold, threshold_config=threshold_config)
        test_frame = _slice_test_frame(prepared, fold, threshold_config=threshold_config)
        policy_table, global_result = search_thresholds_by_composite_regime(
            train_frame,
            direction=direction,
            config=threshold_config,
        )
        fold_threshold_config = replace(threshold_config, global_threshold=float(global_result["threshold"]))
        evaluation = evaluate_policy_variants(
            oof_frame=train_frame,
            test_frame=test_frame,
            policy_table=policy_table,
            config=fold_threshold_config,
            abstain_config=abstain_policy,
        ).copy()
        evaluation["dataset_split"] = evaluation["dataset_split"].replace({"oof": "train"})
        selection = select_walk_forward_policy(
            evaluation,
            split_name="train",
            min_trades_per_week=backtest_config.min_trades_per_week,
        )
        selected_policy_name = str(selection["selected_policy_name"])

        train_decisions = build_policy_variant_decisions(
            train_frame,
            policy_name=selected_policy_name,
            policy_table=policy_table,
            threshold_config=fold_threshold_config,
            abstain_config=abstain_policy,
        )
        test_decisions = build_policy_variant_decisions(
            test_frame,
            policy_name=selected_policy_name,
            policy_table=policy_table,
            threshold_config=fold_threshold_config,
            abstain_config=abstain_policy,
        )

        fold_train_trades = extract_emitted_trades(
            train_decisions,
            fold=fold,
            dataset_split="train",
            threshold_config=fold_threshold_config,
            model_id=model_id,
            direction=direction,
            backend=backend,
        )
        fold_test_trades = extract_emitted_trades(
            test_decisions,
            fold=fold,
            dataset_split="test",
            threshold_config=fold_threshold_config,
            model_id=model_id,
            direction=direction,
            backend=backend,
        )
        train_metrics = summarize_trade_performance(
            fold_train_trades,
            period_start=fold.train_start,
            period_end=fold.train_end,
            benchmark_monthly_returns=benchmark_monthly_returns,
        )
        test_metrics = summarize_trade_performance(
            fold_test_trades,
            period_start=fold.scheduled_test_start,
            period_end=fold.scheduled_test_end - pd.Timedelta(microseconds=1),
            benchmark_monthly_returns=benchmark_monthly_returns,
        )
        fold_wfe = None
        if train_metrics.get("annualized_net_pnl_pips") is not None and abs(float(train_metrics["annualized_net_pnl_pips"])) > 0.0:
            fold_wfe = float(test_metrics["annualized_net_pnl_pips"] / train_metrics["annualized_net_pnl_pips"])

        train_eval_row = _lookup_policy_evaluation_row(evaluation, "train", selected_policy_name)
        test_eval_row = _lookup_policy_evaluation_row(evaluation, "test", selected_policy_name)
        fold_rows.append(
            {
                "fold_id": fold.fold_id,
                "fold_index": int(fold.fold_index),
                "scheduled_test_start": fold.scheduled_test_start,
                "scheduled_test_end": fold.scheduled_test_end,
                "train_start": fold.train_start,
                "train_end": fold.train_end,
                "test_start": fold.test_start,
                "test_end": fold.test_end,
                "train_rows": int(fold.train_rows),
                "test_rows": int(fold.test_rows),
                "purge_gap_bars": int(fold.purge_gap_bars),
                "global_threshold": float(global_result["threshold"]),
                "selected_policy_name": selected_policy_name,
                "qualified_policy_names": "|".join(selection["qualified_policy_names"]),
                "selected_policy_reason": selection["selection_reason"],
                "train_event_f05": float(train_eval_row["event_f05"]),
                "train_post_cost_expectancy_pips": float(train_eval_row["post_cost_expectancy_pips"]),
                "train_net_pnl_pips": float(train_eval_row["net_pnl_pips"]),
                "test_event_f05": float(test_eval_row["event_f05"]),
                "test_post_cost_expectancy_pips": float(test_eval_row["post_cost_expectancy_pips"]),
                "test_net_pnl_pips_from_policy_eval": float(test_eval_row["net_pnl_pips"]),
                "train_trade_count": int(train_metrics["trade_count"]),
                "train_total_net_pnl_pips": float(train_metrics["total_net_pnl_pips"]),
                "train_expectancy_pips": float(train_metrics["expectancy_pips"]),
                "train_profit_factor": train_metrics["profit_factor"],
                "train_annualized_net_pnl_pips": float(train_metrics["annualized_net_pnl_pips"]),
                "test_trade_count": int(test_metrics["trade_count"]),
                "test_total_net_pnl_pips": float(test_metrics["total_net_pnl_pips"]),
                "test_expectancy_pips": float(test_metrics["expectancy_pips"]),
                "test_profit_factor": test_metrics["profit_factor"],
                "test_annualized_net_pnl_pips": float(test_metrics["annualized_net_pnl_pips"]),
                "fold_wfe": fold_wfe,
            }
        )

        fold_policy_table = policy_table.copy()
        fold_policy_table.insert(0, "fold_id", fold.fold_id)
        fold_policy_table.insert(1, "fold_index", int(fold.fold_index))
        fold_policy_table.insert(2, "selected_policy_name", selected_policy_name)
        fold_policy_table.insert(3, "searched_global_threshold", float(global_result["threshold"]))
        policy_tables.append(fold_policy_table)

        fold_evaluation = evaluation.copy()
        fold_evaluation.insert(0, "fold_id", fold.fold_id)
        fold_evaluation.insert(1, "fold_index", int(fold.fold_index))
        fold_evaluation.insert(2, "selected_policy_name", selected_policy_name)
        fold_evaluation["is_selected_policy"] = fold_evaluation["policy_name"].eq(selected_policy_name)
        policy_evaluations.append(fold_evaluation)

        train_trades.append(fold_train_trades)
        test_trades.append(fold_test_trades)

    fold_summary = pd.DataFrame(fold_rows).sort_values("fold_index").reset_index(drop=True)
    policy_table_frame = pd.concat(policy_tables, ignore_index=True) if policy_tables else pd.DataFrame()
    policy_evaluation_frame = pd.concat(policy_evaluations, ignore_index=True) if policy_evaluations else pd.DataFrame()
    train_trade_frame = pd.concat(train_trades, ignore_index=True) if train_trades else pd.DataFrame()
    test_trade_frame = pd.concat(test_trades, ignore_index=True) if test_trades else pd.DataFrame()

    test_trade_frame = add_confidence_quintiles(
        test_trade_frame,
        probability_column="policy_probability",
        bucket_count=backtest_config.confidence_bucket_count,
    )
    test_trade_frame = add_calendar_columns(test_trade_frame, datetime_column="entry_datetime")
    train_trade_frame = add_calendar_columns(train_trade_frame, datetime_column="entry_datetime")

    overall_period_start = pd.Timestamp(fold_summary["scheduled_test_start"].min())
    overall_period_end = pd.Timestamp(fold_summary["scheduled_test_end"].max()) - pd.Timedelta(microseconds=1)
    overall_test_metrics = summarize_trade_performance(
        test_trade_frame,
        period_start=overall_period_start,
        period_end=overall_period_end,
        benchmark_monthly_returns=benchmark_monthly_returns,
    )
    equity_curve = build_equity_curve(
        test_trade_frame,
        timestamp_column="entry_datetime",
        pnl_column="net_pnl_pips",
        sort_columns=("entry_datetime", "source_row_idx", "fold_index"),
    )
    if not equity_curve.empty:
        if "model_id" not in equity_curve.columns:
            equity_curve.insert(0, "model_id", model_id)
        else:
            equity_curve["model_id"] = equity_curve["model_id"].fillna(model_id)
        if "direction" not in equity_curve.columns:
            insert_at = 1 if "model_id" in equity_curve.columns else 0
            equity_curve.insert(insert_at, "direction", direction)
        else:
            equity_curve["direction"] = equity_curve["direction"].fillna(direction)

    breakdowns = {
        "breakdown_by_year": build_group_breakdown(test_trade_frame, group_column="calendar_year"),
        "breakdown_by_quarter": build_group_breakdown(test_trade_frame, group_column="calendar_quarter"),
        "breakdown_by_session": build_group_breakdown(test_trade_frame, group_column="session_regime"),
        "breakdown_by_composite": build_group_breakdown(test_trade_frame, group_column="composite_regime"),
        "breakdown_by_confidence_quintile": build_group_breakdown(test_trade_frame, group_column="confidence_quintile"),
    }
    positive_composite_share = _positive_expectancy_share(breakdowns["breakdown_by_composite"])
    wfe_summary = summarize_walk_forward_efficiency(fold_summary)
    acceptance = {
        "policy_profitable_after_costs": bool(
            float(overall_test_metrics["total_net_pnl_pips"]) > 0.0
            and float(overall_test_metrics["expectancy_pips"]) > 0.0
        ),
        "wfe_above_threshold": bool(
            wfe_summary["overall_wfe"] is not None
            and float(wfe_summary["overall_wfe"]) > float(backtest_config.minimum_wfe)
        ),
        "profitable_quarter_share_above_threshold": bool(
            float(overall_test_metrics["profitable_quarter_share"])
            >= float(backtest_config.minimum_profitable_quarter_share)
        ),
        "positive_composite_expectancy_share_above_threshold": bool(
            positive_composite_share >= float(backtest_config.minimum_positive_composite_share)
        ),
        "largest_single_trade_share_below_limit": bool(
            overall_test_metrics["largest_single_trade_share_of_total_pnl"] is not None
            and float(overall_test_metrics["largest_single_trade_share_of_total_pnl"])
            < float(backtest_config.max_single_trade_share)
        ),
        "max_drawdown_less_than_two_times_average_monthly_profit": bool(
            float(overall_test_metrics["average_positive_month_pnl_pips"]) > 0.0
            and float(overall_test_metrics["max_drawdown_pips"])
            < (2.0 * float(overall_test_metrics["average_positive_month_pnl_pips"]))
        ),
    }

    summary = {
        "model_id": model_id,
        "direction": direction,
        "backend": backend,
        "fold_count": int(len(folds)),
        "available_prediction_start": prepared[threshold_config.datetime_column].min(),
        "available_prediction_end": prepared[threshold_config.datetime_column].max(),
        "selected_policy_counts": test_trade_frame.get("policy_name", pd.Series(dtype="object")).value_counts().to_dict(),
        "overall_test_metrics": overall_test_metrics,
        "walk_forward_efficiency": wfe_summary,
        "positive_composite_expectancy_share": positive_composite_share,
        "acceptance": acceptance,
    }

    return {
        "fold_summary": fold_summary,
        "policy_table": policy_table_frame,
        "policy_evaluation": policy_evaluation_frame,
        "selected_train_trades": train_trade_frame,
        "selected_test_trades": test_trade_frame,
        "selected_test_equity_curve": equity_curve,
        "summary": sanitize_for_json(summary),
        **breakdowns,
    }


def select_walk_forward_policy(
    evaluation: pd.DataFrame,
    *,
    split_name: str,
    min_trades_per_week: float,
) -> dict[str, Any]:
    split_rows = evaluation.loc[evaluation["dataset_split"] == split_name].copy()
    if split_rows.empty:
        raise ValueError(f"Walk-forward policy selection is missing the {split_name!r} split rows.")

    baseline = split_rows.loc[split_rows["policy_name"] == "global_threshold"]
    if baseline.empty:
        raise ValueError("Walk-forward policy selection requires a global_threshold baseline row.")
    baseline_row = baseline.iloc[0]

    qualified = split_rows.loc[
        (split_rows["policy_name"] != "global_threshold")
        & (pd.to_numeric(split_rows["event_f05"], errors="coerce") > float(baseline_row["event_f05"]))
        & (
            pd.to_numeric(split_rows["post_cost_expectancy_pips"], errors="coerce")
            > float(baseline_row["post_cost_expectancy_pips"])
        )
        & (pd.to_numeric(split_rows["trades_per_week"], errors="coerce") >= float(min_trades_per_week))
    ].copy()

    if qualified.empty:
        return {
            "qualified_policy_names": [],
            "selected_policy_name": "global_threshold",
            "selection_reason": "no_non_global_policy_qualified_against_train_baseline",
        }

    qualified = qualified.sort_values(
        ["post_cost_expectancy_pips", "event_f05", "net_pnl_pips"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    selected = qualified.iloc[0]
    return {
        "qualified_policy_names": qualified["policy_name"].astype(str).tolist(),
        "selected_policy_name": str(selected["policy_name"]),
        "selection_reason": "best_train_policy_that_beats_global_on_event_f05_and_post_cost_expectancy",
    }


def build_policy_variant_decisions(
    frame: pd.DataFrame,
    *,
    policy_name: str,
    policy_table: pd.DataFrame,
    threshold_config: ThresholdSearchConfig,
    abstain_config: HardAbstainConfig,
) -> pd.DataFrame:
    if policy_name in {"global_threshold", "global_threshold_plus_abstain"}:
        thresholded = apply_threshold_policy(
            frame,
            config=threshold_config,
            policy_table=policy_table.iloc[0:0].copy(),
            policy_name=policy_name,
            fallback_threshold=threshold_config.global_threshold,
        )
        if policy_name == "global_threshold":
            thresholded["policy_emit_signal"] = thresholded["policy_signal_candidate"]
            thresholded["policy_abstain"] = False
            thresholded["policy_abstain_reason"] = pd.Series(pd.NA, index=thresholded.index, dtype="object")
            return thresholded
        return apply_abstain_policy(thresholded, config=abstain_config)

    thresholded = apply_threshold_policy(
        frame,
        config=threshold_config,
        policy_table=policy_table,
        policy_name=policy_name,
        fallback_threshold=threshold_config.global_threshold,
    )
    if policy_name == "regime_threshold":
        thresholded["policy_emit_signal"] = thresholded["policy_signal_candidate"]
        thresholded["policy_abstain"] = False
        thresholded["policy_abstain_reason"] = pd.Series(pd.NA, index=thresholded.index, dtype="object")
        return thresholded
    if policy_name == "regime_threshold_plus_abstain":
        return apply_abstain_policy(thresholded, config=abstain_config)
    raise ValueError(f"Unsupported policy variant: {policy_name!r}")


def extract_emitted_trades(
    decisions: pd.DataFrame,
    *,
    fold: WalkForwardFold,
    dataset_split: str,
    threshold_config: ThresholdSearchConfig,
    model_id: str | None,
    direction: str,
    backend: str | None,
) -> pd.DataFrame:
    if decisions.empty or "policy_emit_signal" not in decisions.columns:
        return pd.DataFrame()

    emitted = decisions.loc[decisions["policy_emit_signal"].fillna(False).astype(bool)].copy()
    if emitted.empty:
        return pd.DataFrame()

    emitted["entry_datetime"] = pd.to_datetime(emitted[threshold_config.datetime_column], errors="coerce")
    emitted["exit_datetime"] = pd.to_datetime(
        emitted.get(threshold_config.exit_datetime_column, pd.Series(pd.NaT, index=emitted.index)),
        errors="coerce",
    )
    emitted["gross_pnl_pips"] = pd.to_numeric(
        emitted.get(threshold_config.gross_pnl_column, pd.Series(np.nan, index=emitted.index)),
        errors="coerce",
    )
    emitted["net_pnl_pips"] = pd.to_numeric(
        emitted.get(threshold_config.net_pnl_column, pd.Series(np.nan, index=emitted.index)),
        errors="coerce",
    )
    emitted["total_cost_pips"] = pd.to_numeric(
        emitted.get(threshold_config.total_cost_column, pd.Series(np.nan, index=emitted.index)),
        errors="coerce",
    )
    emitted["entry_price"] = pd.to_numeric(
        emitted.get(threshold_config.entry_price_column, pd.Series(np.nan, index=emitted.index)),
        errors="coerce",
    )
    emitted["exit_price"] = pd.to_numeric(
        emitted.get(threshold_config.exit_price_column, pd.Series(np.nan, index=emitted.index)),
        errors="coerce",
    )
    emitted["policy_probability"] = pd.to_numeric(emitted.get("policy_probability"), errors="coerce")
    emitted["policy_threshold"] = pd.to_numeric(emitted.get("policy_threshold"), errors="coerce")
    emitted["source_row_idx"] = pd.to_numeric(emitted.get(threshold_config.position_column), errors="coerce").astype("Int64")
    emitted["target"] = pd.to_numeric(emitted.get(threshold_config.target_column), errors="coerce").astype("Int64")
    emitted = emitted.loc[emitted["net_pnl_pips"].notna()].copy()
    if emitted.empty:
        return pd.DataFrame()
    emitted["trade_outcome"] = np.where(emitted["net_pnl_pips"] > 0.0, "win", np.where(emitted["net_pnl_pips"] < 0.0, "loss", "flat"))
    emitted["fold_id"] = fold.fold_id
    emitted["fold_index"] = int(fold.fold_index)
    emitted["dataset_split"] = dataset_split
    emitted["model_id"] = model_id
    emitted["direction"] = direction
    emitted["backend"] = backend

    columns = [
        "model_id",
        "direction",
        "backend",
        "fold_id",
        "fold_index",
        "dataset_split",
        "entry_datetime",
        "exit_datetime",
        "source_row_idx",
        "target",
        "policy_name",
        "policy_probability",
        "policy_threshold",
        "policy_threshold_source",
        "composite_regime",
        "session_regime",
        "stress_regime",
        "entry_price",
        "exit_price",
        "gross_pnl_pips",
        "total_cost_pips",
        "net_pnl_pips",
        "trade_outcome",
    ]
    available_columns = [column for column in columns if column in emitted.columns]
    return emitted.loc[:, available_columns].sort_values(["entry_datetime", "source_row_idx"]).reset_index(drop=True)


def add_calendar_columns(
    trades: pd.DataFrame,
    *,
    datetime_column: str,
) -> pd.DataFrame:
    if trades.empty or datetime_column not in trades.columns:
        return trades
    working = trades.copy()
    working[datetime_column] = pd.to_datetime(working[datetime_column], errors="coerce")
    working["calendar_year"] = working[datetime_column].dt.year.astype("Int64").astype(str)
    quarter_timestamps = working[datetime_column]
    if quarter_timestamps.dt.tz is not None:
        quarter_timestamps = quarter_timestamps.dt.tz_localize(None)
    working["calendar_quarter"] = quarter_timestamps.dt.to_period("Q").astype(str)
    return working


def add_confidence_quintiles(
    trades: pd.DataFrame,
    *,
    probability_column: str,
    bucket_count: int,
) -> pd.DataFrame:
    if trades.empty or probability_column not in trades.columns:
        return trades

    working = trades.copy()
    probabilities = pd.to_numeric(working[probability_column], errors="coerce")
    valid_mask = probabilities.notna()
    working["confidence_quintile"] = pd.Series(pd.NA, index=working.index, dtype="object")
    if not bool(valid_mask.any()):
        return working

    valid_count = int(valid_mask.sum())
    bins = max(1, min(int(bucket_count), valid_count))
    labels = [f"Q{index}" for index in range(1, bins + 1)]
    ranked = probabilities.loc[valid_mask].rank(method="first")
    working.loc[valid_mask, "confidence_quintile"] = pd.qcut(ranked, q=bins, labels=labels).astype(str)
    return working


def _slice_train_frame(
    frame: pd.DataFrame,
    fold: WalkForwardFold,
    *,
    threshold_config: ThresholdSearchConfig,
) -> pd.DataFrame:
    datetimes = pd.to_datetime(frame[threshold_config.datetime_column], errors="coerce")
    positions = pd.to_numeric(frame[threshold_config.position_column], errors="coerce")
    mask = (datetimes < fold.scheduled_test_start) & (positions <= float(fold.train_position_end))
    return frame.loc[mask].copy()


def _slice_test_frame(
    frame: pd.DataFrame,
    fold: WalkForwardFold,
    *,
    threshold_config: ThresholdSearchConfig,
) -> pd.DataFrame:
    datetimes = pd.to_datetime(frame[threshold_config.datetime_column], errors="coerce")
    mask = (datetimes >= fold.scheduled_test_start) & (datetimes < fold.scheduled_test_end)
    return frame.loc[mask].copy()


def _positive_expectancy_share(breakdown: pd.DataFrame) -> float:
    if breakdown.empty or "expectancy_pips" not in breakdown.columns:
        return 0.0
    expectancy = pd.to_numeric(breakdown["expectancy_pips"], errors="coerce").dropna()
    if expectancy.empty:
        return 0.0
    return float((expectancy > 0.0).mean())


def _lookup_policy_evaluation_row(
    evaluation: pd.DataFrame,
    split_name: str,
    policy_name: str,
) -> pd.Series:
    match = evaluation.loc[
        (evaluation["dataset_split"] == split_name)
        & (evaluation["policy_name"] == policy_name)
    ]
    if match.empty:
        raise ValueError(f"Missing evaluation row for split={split_name!r}, policy={policy_name!r}.")
    return match.iloc[0]


def _next_quarter_start(timestamp: pd.Timestamp) -> pd.Timestamp:
    quarter_month = ((timestamp.month - 1) // 3) * 3 + 1
    quarter_start = pd.Timestamp(
        year=timestamp.year,
        month=quarter_month,
        day=1,
        tz=timestamp.tz if timestamp.tz is not None else None,
    )
    if timestamp > quarter_start:
        quarter_start = quarter_start + pd.DateOffset(months=3)
    return quarter_start
