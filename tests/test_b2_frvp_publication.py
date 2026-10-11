import numpy as np
import pandas as pd

from data.labeling.frvp_labeling_engine import FRVPLabelingParams, _detect_htf_swings


def test_completed_atr_is_prefix_invariant_and_uses_last_completed_hour():
    index = pd.date_range("2026-01-01", periods=600, freq="5min", tz="UTC")
    value = 100 + np.arange(600) * .01
    bars = pd.DataFrame(dict(open=value, high=value+1, low=value-1, close=value, volume=1), index=index)
    params = FRVPLabelingParams(completed_htf_publication=True)
    full = _detect_htf_swings(bars, params)[2]
    prefix = _detect_htf_swings(bars.iloc[:497], params)[2]
    pd.testing.assert_series_equal(prefix, full.iloc[:497])
    changed = bars.copy()
    changed.loc[index[497:504], "high"] += 100
    control = _detect_htf_swings(changed, params)[2]
    pd.testing.assert_series_equal(full.iloc[:503], control.iloc[:503])
    assert full.iloc[503] != control.iloc[503]  # 17:55 bar decides at 18:00.


def test_incomplete_hour_does_not_publish_a_new_atr():
    index = pd.date_range("2026-01-01", periods=600, freq="5min", tz="UTC")
    value = 100 + np.arange(600) * .01
    bars = pd.DataFrame(dict(open=value, high=value+1, low=value-1, close=value, volume=1), index=index)
    bars = bars.drop(index[500])
    atr = _detect_htf_swings(bars, FRVPLabelingParams(completed_htf_publication=True))[2]
    assert atr.loc[index[503]] == atr.loc[index[491]]
