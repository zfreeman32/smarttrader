from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd


def add_unit_aliases(payload: Mapping[str, Any]) -> dict[str, Any]:
    aliased = dict(payload)
    for key, value in payload.items():
        alias_key = _unit_alias_key(str(key))
        if alias_key is not None:
            aliased[alias_key] = value
    return aliased


def add_unit_alias_columns(frame: pd.DataFrame) -> pd.DataFrame:
    working = frame.copy()
    for column in list(frame.columns):
        alias_column = _unit_alias_key(column)
        if alias_column is not None:
            working[alias_column] = working[column]
    return working


def _timestamp_to_period(timestamp: pd.Timestamp, *, freq: str) -> pd.Period:
    normalized = timestamp.tz_localize(None) if timestamp.tz is not None else timestamp
    return normalized.to_period(freq)


def _timestamps_to_period_series(timestamps: pd.Series, *, freq: str) -> pd.Series:
    normalized = timestamps
    if normalized.dt.tz is not None:
        normalized = normalized.dt.tz_localize(None)
    return normalized.dt.to_period(freq)


def compute_period_pnl_series(
    trades: pd.DataFrame,
    *,
    timestamp_column: str,
    pnl_column: str,
    freq: str,
    period_start: pd.Timestamp | None = None,
    period_end: pd.Timestamp | None = None,
) -> pd.Series:
    start = _coerce_timestamp(period_start)
    end = _coerce_timestamp(period_end)

    if start is None or end is None:
        timestamps = pd.to_datetime(trades.get(timestamp_column), errors="coerce").dropna()
        if timestamps.empty:
            if start is None or end is None:
                return pd.Series(dtype=float)
        else:
            if start is None:
                start = pd.Timestamp(timestamps.min())
            if end is None:
                end = pd.Timestamp(timestamps.max())

    if start is None or end is None:
        return pd.Series(dtype=float)

    if end < start:
        end = start

    full_index = pd.period_range(
        start=_timestamp_to_period(start, freq=freq),
        end=_timestamp_to_period(end, freq=freq),
        freq=freq,
    )
    if len(full_index) == 0:
        return pd.Series(dtype=float)

    if trades.empty or timestamp_column not in trades.columns or pnl_column not in trades.columns:
        return pd.Series(0.0, index=full_index, dtype=float)

    working = trades.loc[:, [timestamp_column, pnl_column]].copy()
    working[timestamp_column] = pd.to_datetime(working[timestamp_column], errors="coerce")
    working[pnl_column] = pd.to_numeric(working[pnl_column], errors="coerce")
    working = working.dropna(subset=[timestamp_column, pnl_column])
    if working.empty:
        return pd.Series(0.0, index=full_index, dtype=float)

    working["_period"] = _timestamps_to_period_series(working[timestamp_column], freq=freq)
    grouped = working.groupby("_period", sort=True)[pnl_column].sum()
    return grouped.reindex(full_index, fill_value=0.0).astype(float)


def compute_buy_hold_period_returns(
    market_frame: pd.DataFrame,
    *,
    datetime_column: str = "datetime",
    price_column: str = "close",
    pip_size: float = 1e-4,
    freq: str = "M",
) -> pd.Series:
    if market_frame.empty or datetime_column not in market_frame.columns or price_column not in market_frame.columns:
        return pd.Series(dtype=float)

    working = market_frame.loc[:, [datetime_column, price_column]].copy()
    working[datetime_column] = pd.to_datetime(working[datetime_column], errors="coerce")
    working[price_column] = pd.to_numeric(working[price_column], errors="coerce")
    working = working.dropna(subset=[datetime_column, price_column]).sort_values(datetime_column)
    if working.empty:
        return pd.Series(dtype=float)

    working["_period"] = _timestamps_to_period_series(working[datetime_column], freq=freq)
    period_prices = working.groupby("_period", sort=True)[price_column].agg(["first", "last"])
    returns = (period_prices["last"] - period_prices["first"]) / float(pip_size)
    return returns.astype(float)


def build_equity_curve(
    trades: pd.DataFrame,
    *,
    timestamp_column: str,
    pnl_column: str,
    sort_columns: tuple[str, ...] = ("entry_datetime", "source_row_idx"),
) -> pd.DataFrame:
    if trades.empty or timestamp_column not in trades.columns or pnl_column not in trades.columns:
        return add_unit_alias_columns(
            pd.DataFrame(
                columns=[
                    timestamp_column,
                    pnl_column,
                    "equity_pips",
                    "running_peak_pips",
                    "drawdown_pips",
                ]
            )
        )

    working = trades.copy()
    working[timestamp_column] = pd.to_datetime(working[timestamp_column], errors="coerce")
    working[pnl_column] = pd.to_numeric(working[pnl_column], errors="coerce")
    working = working.dropna(subset=[timestamp_column, pnl_column]).copy()
    if working.empty:
        return add_unit_alias_columns(
            pd.DataFrame(
                columns=[
                    timestamp_column,
                    pnl_column,
                    "equity_pips",
                    "running_peak_pips",
                    "drawdown_pips",
                ]
            )
        )

    available_sort_columns = [column for column in sort_columns if column in working.columns]
    if timestamp_column not in available_sort_columns:
        available_sort_columns.insert(0, timestamp_column)
    working = working.sort_values(available_sort_columns).reset_index(drop=True)
    working["equity_pips"] = working[pnl_column].cumsum()
    working["running_peak_pips"] = working["equity_pips"].cummax()
    working["drawdown_pips"] = working["equity_pips"] - working["running_peak_pips"]
    return add_unit_alias_columns(working)


def summarize_trade_performance(
    trades: pd.DataFrame,
    *,
    timestamp_column: str = "entry_datetime",
    pnl_column: str = "net_pnl_pips",
    gross_pnl_column: str = "gross_pnl_pips",
    period_start: pd.Timestamp | None = None,
    period_end: pd.Timestamp | None = None,
    benchmark_monthly_returns: pd.Series | None = None,
    effective_trials: int = 1,
) -> dict[str, Any]:
    start = _coerce_timestamp(period_start)
    end = _coerce_timestamp(period_end)
    if start is None or end is None:
        timestamps = pd.to_datetime(trades.get(timestamp_column), errors="coerce").dropna()
        if not timestamps.empty:
            if start is None:
                start = pd.Timestamp(timestamps.min())
            if end is None:
                end = pd.Timestamp(timestamps.max())

    if start is None or end is None:
        start = pd.Timestamp("1970-01-01")
        end = start
    if end < start:
        end = start

    if trades.empty:
        monthly_pnl = compute_period_pnl_series(
            pd.DataFrame(columns=[timestamp_column, pnl_column]),
            timestamp_column=timestamp_column,
            pnl_column=pnl_column,
            freq="M",
            period_start=start,
            period_end=end,
        )
        quarterly_pnl = compute_period_pnl_series(
            pd.DataFrame(columns=[timestamp_column, pnl_column]),
            timestamp_column=timestamp_column,
            pnl_column=pnl_column,
            freq="Q",
            period_start=start,
            period_end=end,
        )
        return _base_performance_summary(
            period_start=start,
            period_end=end,
            monthly_pnl=monthly_pnl,
            quarterly_pnl=quarterly_pnl,
            benchmark_monthly_returns=benchmark_monthly_returns,
            effective_trials=effective_trials,
        )

    working = trades.copy()
    working[timestamp_column] = pd.to_datetime(working[timestamp_column], errors="coerce")
    working[pnl_column] = pd.to_numeric(working.get(pnl_column), errors="coerce")
    if gross_pnl_column in working.columns:
        working[gross_pnl_column] = pd.to_numeric(working[gross_pnl_column], errors="coerce")
    else:
        working[gross_pnl_column] = np.nan
    working = working.dropna(subset=[timestamp_column, pnl_column]).copy()

    if working.empty:
        monthly_pnl = compute_period_pnl_series(
            pd.DataFrame(columns=[timestamp_column, pnl_column]),
            timestamp_column=timestamp_column,
            pnl_column=pnl_column,
            freq="M",
            period_start=start,
            period_end=end,
        )
        quarterly_pnl = compute_period_pnl_series(
            pd.DataFrame(columns=[timestamp_column, pnl_column]),
            timestamp_column=timestamp_column,
            pnl_column=pnl_column,
            freq="Q",
            period_start=start,
            period_end=end,
        )
        return _base_performance_summary(
            period_start=start,
            period_end=end,
            monthly_pnl=monthly_pnl,
            quarterly_pnl=quarterly_pnl,
            benchmark_monthly_returns=benchmark_monthly_returns,
            effective_trials=effective_trials,
        )

    working = working.sort_values(timestamp_column).reset_index(drop=True)
    pnl = working[pnl_column].astype(float)
    gross_pnl = working[gross_pnl_column].astype(float)
    wins = pnl.loc[pnl > 0.0]
    losses = pnl.loc[pnl < 0.0]
    equity_curve = build_equity_curve(working, timestamp_column=timestamp_column, pnl_column=pnl_column)
    drawdown_stats = compute_drawdown_stats(equity_curve, timestamp_column=timestamp_column)

    monthly_pnl = compute_period_pnl_series(
        working,
        timestamp_column=timestamp_column,
        pnl_column=pnl_column,
        freq="M",
        period_start=start,
        period_end=end,
    )
    quarterly_pnl = compute_period_pnl_series(
        working,
        timestamp_column=timestamp_column,
        pnl_column=pnl_column,
        freq="Q",
        period_start=start,
        period_end=end,
    )

    summary = _base_performance_summary(
        period_start=start,
        period_end=end,
        monthly_pnl=monthly_pnl,
        quarterly_pnl=quarterly_pnl,
        benchmark_monthly_returns=benchmark_monthly_returns,
        effective_trials=effective_trials,
    )

    total_net_pnl = float(pnl.sum())
    total_abs_pnl = float(np.abs(pnl).sum())
    trade_count = int(len(working))
    average_win = float(wins.mean()) if not wins.empty else 0.0
    average_loss = float(abs(losses.mean())) if not losses.empty else 0.0
    loss_sum = float(abs(losses.sum()))
    profit_factor = (
        float(wins.sum()) / loss_sum
        if loss_sum > 0.0
        else (None if wins.empty else None)
    )
    largest_trade_share = None
    if trade_count > 0 and abs(total_net_pnl) > 0.0:
        largest_trade_share = float(np.abs(pnl).max() / abs(total_net_pnl))

    concentration = pnl_trade_concentration(pnl)

    summary.update(
        {
            "trade_count": trade_count,
            "total_net_pnl_pips": total_net_pnl,
            "total_gross_pnl_pips": float(gross_pnl.sum()) if not gross_pnl.isna().all() else 0.0,
            "expectancy_pips": float(total_net_pnl / trade_count) if trade_count > 0 else 0.0,
            "trades_per_month": float(trade_count / summary["period_months"]) if float(summary["period_months"]) > 0.0 else 0.0,
            "profit_factor": profit_factor,
            "hit_rate": float(len(wins) / trade_count) if trade_count > 0 else 0.0,
            "average_win_pips": average_win,
            "average_loss_pips": average_loss,
            "average_win_loss_ratio": (
                float(average_win / average_loss)
                if average_loss > 0.0
                else (None if average_win > 0.0 else 0.0)
            ),
            "max_drawdown_pips": float(drawdown_stats["max_drawdown_pips"]),
            "max_drawdown_duration_days": float(drawdown_stats["max_drawdown_duration_days"]),
            "largest_single_trade_share_of_total_pnl": largest_trade_share,
            "pnl_herfindahl_index": concentration["herfindahl_index"],
            "largest_single_trade_share_of_abs_pnl": concentration["largest_abs_share"],
        }
    )
    return add_unit_aliases(summary)


def compute_drawdown_stats(
    equity_curve: pd.DataFrame,
    *,
    timestamp_column: str,
) -> Mapping[str, float]:
    if equity_curve.empty or "drawdown_pips" not in equity_curve.columns or timestamp_column not in equity_curve.columns:
        return add_unit_aliases(
            {
                "max_drawdown_pips": 0.0,
                "max_drawdown_duration_days": 0.0,
            }
        )

    working = equity_curve.loc[:, [timestamp_column, "drawdown_pips"]].copy()
    working[timestamp_column] = pd.to_datetime(working[timestamp_column], errors="coerce")
    working["drawdown_pips"] = pd.to_numeric(working["drawdown_pips"], errors="coerce")
    working = working.dropna(subset=[timestamp_column, "drawdown_pips"]).reset_index(drop=True)
    if working.empty:
        return add_unit_aliases(
            {
                "max_drawdown_pips": 0.0,
                "max_drawdown_duration_days": 0.0,
            }
        )

    max_drawdown = float(abs(min(float(working["drawdown_pips"].min()), 0.0)))
    max_duration_days = 0.0
    underwater_start: pd.Timestamp | None = None
    for _, row in working.iterrows():
        timestamp = pd.Timestamp(row[timestamp_column])
        drawdown = float(row["drawdown_pips"])
        if drawdown < 0.0 and underwater_start is None:
            underwater_start = timestamp
        elif drawdown >= 0.0 and underwater_start is not None:
            duration_days = (timestamp - underwater_start).total_seconds() / 86400.0
            max_duration_days = max(max_duration_days, duration_days)
            underwater_start = None

    if underwater_start is not None:
        last_timestamp = pd.Timestamp(working[timestamp_column].iloc[-1])
        duration_days = (last_timestamp - underwater_start).total_seconds() / 86400.0
        max_duration_days = max(max_duration_days, duration_days)

    return add_unit_aliases(
        {
            "max_drawdown_pips": max_drawdown,
            "max_drawdown_duration_days": float(max_duration_days),
        }
    )


def build_group_breakdown(
    trades: pd.DataFrame,
    *,
    group_column: str,
    timestamp_column: str = "entry_datetime",
    pnl_column: str = "net_pnl_pips",
) -> pd.DataFrame:
    if trades.empty or group_column not in trades.columns:
        return add_unit_alias_columns(
            pd.DataFrame(
                columns=[
                    group_column,
                    "trade_count",
                    "net_pnl_pips",
                    "expectancy_pips",
                    "hit_rate",
                    "profit_factor",
                    "average_win_pips",
                    "average_loss_pips",
                ]
            )
        )

    rows: list[dict[str, Any]] = []
    for group_value, bucket in trades.groupby(group_column, dropna=False, sort=True):
        metrics = summarize_trade_performance(
            bucket,
            timestamp_column=timestamp_column,
            pnl_column=pnl_column,
            period_start=pd.to_datetime(bucket[timestamp_column], errors="coerce").min(),
            period_end=pd.to_datetime(bucket[timestamp_column], errors="coerce").max(),
        )
        rows.append(
            {
                group_column: group_value,
                "trade_count": int(metrics["trade_count"]),
                "net_pnl_pips": metrics["total_net_pnl_pips"],
                "expectancy_pips": metrics["expectancy_pips"],
                "hit_rate": metrics["hit_rate"],
                "profit_factor": metrics["profit_factor"],
                "average_win_pips": metrics["average_win_pips"],
                "average_loss_pips": metrics["average_loss_pips"],
            }
        )
    return add_unit_alias_columns(pd.DataFrame(rows))


def summarize_walk_forward_efficiency(fold_summary: pd.DataFrame) -> dict[str, Any]:
    if fold_summary.empty:
        return add_unit_aliases(
            {
                "fold_count": 0,
                "mean_train_annualized_net_pnl_pips": 0.0,
                "mean_test_annualized_net_pnl_pips": 0.0,
                "overall_wfe": None,
                "median_fold_wfe": None,
                "profitable_test_fold_share": 0.0,
            }
        )

    working = fold_summary.copy()
    working["train_annualized_net_pnl_pips"] = pd.to_numeric(
        working.get("train_annualized_net_pnl_pips"),
        errors="coerce",
    )
    working["test_annualized_net_pnl_pips"] = pd.to_numeric(
        working.get("test_annualized_net_pnl_pips"),
        errors="coerce",
    )
    working["test_total_net_pnl_pips"] = pd.to_numeric(
        working.get("test_total_net_pnl_pips"),
        errors="coerce",
    )
    working["fold_wfe"] = pd.to_numeric(working.get("fold_wfe"), errors="coerce")

    mean_train = float(working["train_annualized_net_pnl_pips"].mean()) if not working.empty else 0.0
    mean_test = float(working["test_annualized_net_pnl_pips"].mean()) if not working.empty else 0.0
    overall_wfe = None
    if abs(mean_train) > 0.0:
        overall_wfe = float(mean_test / mean_train)

    valid_fold_wfe = working["fold_wfe"].replace([np.inf, -np.inf], np.nan).dropna()
    profitable_share = float((working["test_total_net_pnl_pips"] > 0.0).mean()) if not working.empty else 0.0
    return add_unit_aliases(
        {
            "fold_count": int(len(working)),
            "mean_train_annualized_net_pnl_pips": mean_train,
            "mean_test_annualized_net_pnl_pips": mean_test,
            "overall_wfe": overall_wfe,
            "median_fold_wfe": float(valid_fold_wfe.median()) if not valid_fold_wfe.empty else None,
            "profitable_test_fold_share": profitable_share,
        }
    )


def sanitize_for_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): sanitize_for_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [sanitize_for_json(item) for item in value]
    if isinstance(value, tuple):
        return [sanitize_for_json(item) for item in value]
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, pd.Period):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        if not np.isfinite(value):
            return None
        return float(value)
    if pd.isna(value):
        return None
    return value


def pnl_trade_concentration(pnl: pd.Series | np.ndarray) -> dict[str, float]:
    values = np.abs(pd.to_numeric(pd.Series(pnl), errors="coerce").dropna().to_numpy(dtype=float, copy=True))
    if values.size == 0:
        return {"herfindahl_index": 0.0, "largest_abs_share": 0.0}
    total = float(values.sum())
    if total <= 0.0:
        return {"herfindahl_index": 0.0, "largest_abs_share": 0.0}
    weights = values / total
    return {
        "herfindahl_index": float(np.square(weights).sum()),
        "largest_abs_share": float(weights.max()),
    }


def _base_performance_summary(
    *,
    period_start: pd.Timestamp,
    period_end: pd.Timestamp,
    monthly_pnl: pd.Series,
    quarterly_pnl: pd.Series,
    benchmark_monthly_returns: pd.Series | None,
    effective_trials: int,
) -> dict[str, Any]:
    period_days = max((period_end - period_start).total_seconds() / 86400.0, 1.0 / 288.0)
    period_months = max(period_days / 30.4375, 1.0 / 30.4375)
    period_years = max(period_days / 365.25, 1.0 / 365.25)
    total_net = float(monthly_pnl.sum()) if not monthly_pnl.empty else 0.0

    monthly_values = monthly_pnl.to_numpy(dtype=float, copy=True) if not monthly_pnl.empty else np.asarray([], dtype=float)
    quarterly_values = quarterly_pnl.to_numpy(dtype=float, copy=True) if not quarterly_pnl.empty else np.asarray([], dtype=float)
    sharpe = _annualized_sharpe(monthly_values)
    sortino = _annualized_sortino(monthly_values)
    approx_deflated_sharpe = _approximate_deflated_sharpe(
        sharpe,
        observation_count=int(monthly_values.size),
        effective_trials=effective_trials,
    )
    profitable_quarter_share = float((quarterly_values > 0.0).mean()) if quarterly_values.size else 0.0
    profitable_month_share = float((monthly_values > 0.0).mean()) if monthly_values.size else 0.0
    average_monthly_profit = float(monthly_values[monthly_values > 0.0].mean()) if np.any(monthly_values > 0.0) else 0.0

    benchmark_correlation = None
    if benchmark_monthly_returns is not None and not monthly_pnl.empty:
        aligned = pd.concat(
            [
                monthly_pnl.rename("strategy"),
                benchmark_monthly_returns.reindex(monthly_pnl.index).rename("benchmark"),
            ],
            axis=1,
        ).dropna()
        if len(aligned) >= 2 and float(aligned["strategy"].std(ddof=0)) > 0.0 and float(aligned["benchmark"].std(ddof=0)) > 0.0:
            benchmark_correlation = float(aligned["strategy"].corr(aligned["benchmark"]))

    return add_unit_aliases(
        {
        "period_start": period_start,
        "period_end": period_end,
        "period_days": float(period_days),
        "period_months": float(period_months),
        "period_years": float(period_years),
        "trade_count": 0,
        "total_net_pnl_pips": total_net,
        "total_gross_pnl_pips": 0.0,
        "annualized_net_pnl_pips": float(total_net / period_years) if period_years > 0.0 else 0.0,
        "expectancy_pips": 0.0,
        "trades_per_month": 0.0,
        "profit_factor": None,
        "hit_rate": 0.0,
        "average_win_pips": 0.0,
        "average_loss_pips": 0.0,
        "average_win_loss_ratio": 0.0,
        "max_drawdown_pips": 0.0,
        "max_drawdown_duration_days": 0.0,
        "largest_single_trade_share_of_total_pnl": None,
        "pnl_herfindahl_index": 0.0,
        "largest_single_trade_share_of_abs_pnl": 0.0,
        "monthly_sharpe": sharpe,
        "monthly_sortino": sortino,
        "approx_deflated_sharpe": approx_deflated_sharpe,
        "deflated_sharpe_effective_trials": int(max(int(effective_trials), 1)),
        "monthly_observation_count": int(monthly_values.size),
        "profitable_quarter_share": profitable_quarter_share,
        "profitable_month_share": profitable_month_share,
        "average_positive_month_pnl_pips": average_monthly_profit,
        "benchmark_monthly_correlation": benchmark_correlation,
        }
    )


def _annualized_sharpe(values: np.ndarray) -> float | None:
    if values.size == 0:
        return None
    std = float(np.std(values, ddof=0))
    if std <= 0.0:
        return None
    return float(np.mean(values) / std * np.sqrt(12.0))


def _annualized_sortino(values: np.ndarray) -> float | None:
    if values.size == 0:
        return None
    downside = values[values < 0.0]
    downside_std = float(np.std(downside, ddof=0)) if downside.size else 0.0
    if downside_std <= 0.0:
        return None
    return float(np.mean(values) / downside_std * np.sqrt(12.0))


def _approximate_deflated_sharpe(
    sharpe: float | None,
    *,
    observation_count: int,
    effective_trials: int,
) -> float | None:
    if sharpe is None:
        return None

    trials = max(int(effective_trials), 1)
    if observation_count < 2:
        return float(sharpe) if trials <= 1 else None
    if trials <= 1:
        return float(sharpe)

    sample_penalty = min(1.0, np.sqrt(max(observation_count - 1, 0) / float(trials)))
    sharpe_cap = float(np.sqrt(2.0 * np.log(max(trials, 2))))
    if sharpe_cap <= 0.0:
        return float(sharpe)

    capped_ratio = min((float(sharpe) ** 2) / (sharpe_cap ** 2), 1.0)
    cap_penalty = np.sqrt(max(0.0, 1.0 - capped_ratio))
    return float(float(sharpe) * sample_penalty * cap_penalty)


def _coerce_timestamp(value: pd.Timestamp | str | None) -> pd.Timestamp | None:
    if value is None:
        return None
    timestamp = pd.to_datetime(value, errors="coerce")
    if pd.isna(timestamp):
        return None
    return pd.Timestamp(timestamp)


def _unit_alias_key(column: str) -> str | None:
    if "_pips" not in column:
        return None
    return column.replace("_pips", "_units")
