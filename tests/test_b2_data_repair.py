from types import SimpleNamespace
import numpy as np
import pandas as pd
from data.labeling.causal_ote import completed_ote_frames
from data.labeling.frvp_labeling_engine import annotate_confirmed_htf_spacing
from preprocessing.config import PreprocessingConfig
from preprocessing.feature_selection import resolve_feature_columns
from features.config import FeatureBuilderConfig
from features.feature_sets.continuation_pullback import build_continuation_pullback
from data.labeling.causal_ote import completed_continuation_context, publication_frames


def test_ote_publication_only_uses_completed_full_buckets():
    idx = pd.date_range('2026-01-01', periods=40, freq='5min', tz='UTC')
    frame = pd.DataFrame(dict(open=1., high=2., low=0., close=1., volume=1.), index=idx)
    decisions, half, hour = completed_ote_frames(frame)
    assert decisions.index[0] == idx[0] + pd.Timedelta(minutes=5)
    assert half.index[0] == idx[6]
    assert hour.index[0] == idx[12]
    _, prefix_half, prefix_hour = completed_ote_frames(frame.iloc[:17])
    pd.testing.assert_frame_equal(prefix_half, half.loc[:decisions.index[16]], check_freq=False)
    pd.testing.assert_frame_equal(prefix_hour, hour.loc[:decisions.index[16]], check_freq=False)
    _, _, missing = completed_ote_frames(frame.drop(idx[3]))
    assert idx[12] not in missing.index


def test_frvp_gate_requires_confirmation_not_just_past_swing_time():
    t = pd.Timestamp('2026-01-01T10:00Z')
    swing = SimpleNamespace(swing_type='low', swing_time=t, confirm_time=t+pd.Timedelta(hours=1))
    before = SimpleNamespace(swing_type='low', swing_time=t+pd.Timedelta(minutes=15))
    available = SimpleNamespace(swing_type='low', swing_time=t+pd.Timedelta(minutes=55))
    annotate_confirmed_htf_spacing([before, available], [swing], [], base_bar_minutes=5)
    assert before.bars_since_prev_30m_same is None
    assert available.bars_since_prev_30m_same == 12
    assert available.bars_since_prev_1h_same is None


def test_declared_and_fallback_label_helpers_are_not_predictors():
    frame = pd.DataFrame({'safe': [1., 2.], 'htf_confluence_long_frvp_meta': [0, 1]})
    for metadata in ({}, {'feature_columns': list(frame)}):
        assert resolve_feature_columns(frame, metadata, PreprocessingConfig()) == ['safe']


def test_ote_shared_continuation_context_is_prefix_invariant_and_ignores_carried_labels():
    idx = pd.date_range('2026-01-01', periods=380, freq='5min', tz='UTC')
    close = 1.1 + np.arange(len(idx)) * .00001 + np.sin(np.arange(len(idx)) / 5) * .0002
    bars = pd.DataFrame(dict(datetime=idx, open=close-.00001, high=close+.0001, low=close-.0001, close=close, volume=1.))
    context = completed_continuation_context(bars)
    pd.testing.assert_frame_equal(completed_continuation_context(bars.iloc[:373]), context.iloc[:373])
    cfg = FeatureBuilderConfig(htf_bar_timestamp_semantics='bar_open')
    clean = build_continuation_pullback(bars, cfg)
    poisoned = bars.assign(**{c: 999. for c in context})
    pd.testing.assert_frame_equal(build_continuation_pullback(poisoned, cfg), clean)
    pd.testing.assert_frame_equal(build_continuation_pullback(bars.iloc[:373], cfg), clean.iloc[:373])


def test_explicit_close_stamps_and_open_stamps_publish_the_same_completed_bars():
    idx = pd.date_range('2026-01-01', periods=24, freq='5min', tz='UTC')
    opening = pd.DataFrame(dict(open=1., high=2., low=0., close=1., volume=1.), index=idx)
    closing = opening.copy()
    closing.index += pd.Timedelta(minutes=5)
    for left, right in zip(publication_frames(opening, timestamp_semantics='bar_open', bar_minutes=5), publication_frames(closing, timestamp_semantics='bar_close', bar_minutes=5)):
        pd.testing.assert_frame_equal(left, right)


def test_scalar_scan_score_preserves_vectorized_scores_exactly():
    from features.transforms import proximity_score
    from features.feature_sets.continuation_pullback import _scalar_proximity
    values = np.r_[np.random.default_rng(19).normal(size=1000), np.nan, np.inf, -np.inf]
    for threshold in (1.25, 1.5):
        np.testing.assert_array_equal([_scalar_proximity(v, threshold) for v in values], proximity_score(values, threshold).to_numpy())
