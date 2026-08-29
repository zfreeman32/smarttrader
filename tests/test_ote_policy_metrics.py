from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model_testing.ote_policy_metrics import summarize_trade_performance


def test_summarize_trade_performance_reports_profit_factor_drawdown_and_trade_rate() -> None:
    trades = pd.DataFrame(
        {
            "entry_datetime": pd.to_datetime(
                [
                    "2024-01-05 00:00:00",
                    "2024-01-20 00:00:00",
                    "2024-02-10 00:00:00",
                    "2024-03-15 00:00:00",
                ]
            ),
            "net_pnl_pips": [10.0, -5.0, 15.0, -5.0],
            "gross_pnl_pips": [12.0, -3.0, 17.0, -3.0],
        }
    )

    summary = summarize_trade_performance(
        trades,
        period_start=pd.Timestamp("2024-01-01 00:00:00"),
        period_end=pd.Timestamp("2024-03-31 23:59:59"),
        drawdown_starting_balance_pips=10.0,
    )

    assert int(summary["trade_count"]) == 4
    assert float(summary["total_net_pnl_pips"]) == pytest.approx(15.0)
    assert float(summary["total_net_pnl_units"]) == pytest.approx(15.0)
    assert float(summary["expectancy_pips"]) == pytest.approx(3.75)
    assert float(summary["expectancy_units"]) == pytest.approx(3.75)
    assert float(summary["profit_factor"]) == pytest.approx(2.5)
    assert float(summary["hit_rate"]) == pytest.approx(0.5)
    assert float(summary["max_drawdown_pips"]) == pytest.approx(5.0)
    assert float(summary["max_drawdown_units"]) == pytest.approx(5.0)
    assert float(summary["max_drawdown_pct"]) == pytest.approx(25.0)
    assert float(summary["max_account_drawdown_pct"]) == pytest.approx(25.0)
    assert float(summary["max_profit_retracement_pct"]) == pytest.approx(50.0)
    assert float(summary["drawdown_starting_balance_pips"]) == pytest.approx(10.0)
    assert float(summary["drawdown_starting_balance_units"]) == pytest.approx(10.0)
    assert float(summary["trades_per_month"]) == pytest.approx(4.0 / 2.989726, rel=1e-3)
    assert float(summary["largest_single_trade_share_of_total_pnl"]) == pytest.approx(1.0)
    assert summary["approx_deflated_sharpe"] == summary["monthly_sharpe"]


def test_summarize_trade_performance_deflates_sharpe_when_effective_trials_increase() -> None:
    trades = pd.DataFrame(
        {
            "entry_datetime": pd.to_datetime(
                [
                    "2024-01-05 00:00:00",
                    "2024-02-05 00:00:00",
                    "2024-03-05 00:00:00",
                    "2024-04-05 00:00:00",
                    "2024-05-05 00:00:00",
                    "2024-06-05 00:00:00",
                ]
            ),
            "net_pnl_pips": [8.0, 6.0, -2.0, 9.0, -1.0, 7.0],
            "gross_pnl_pips": [9.0, 7.0, -1.0, 10.0, 0.0, 8.0],
        }
    )

    summary = summarize_trade_performance(
        trades,
        period_start=pd.Timestamp("2024-01-01 00:00:00"),
        period_end=pd.Timestamp("2024-06-30 23:59:59"),
        effective_trials=12,
    )

    assert summary["monthly_sharpe"] is not None
    assert summary["approx_deflated_sharpe"] is not None
    assert float(summary["approx_deflated_sharpe"]) < float(summary["monthly_sharpe"])
    assert int(summary["deflated_sharpe_effective_trials"]) == 12
