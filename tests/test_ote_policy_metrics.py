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
    )

    assert int(summary["trade_count"]) == 4
    assert float(summary["total_net_pnl_pips"]) == pytest.approx(15.0)
    assert float(summary["expectancy_pips"]) == pytest.approx(3.75)
    assert float(summary["profit_factor"]) == pytest.approx(2.5)
    assert float(summary["hit_rate"]) == pytest.approx(0.5)
    assert float(summary["max_drawdown_pips"]) == pytest.approx(5.0)
    assert float(summary["trades_per_month"]) == pytest.approx(4.0 / 2.989726, rel=1e-3)
    assert float(summary["largest_single_trade_share_of_total_pnl"]) == pytest.approx(1.0)
