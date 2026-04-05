# v1 vs v2 Regime Slice Report Summary

This summary covers the `v1_v2_comparison` regime-slice run produced from:

- `model_testing/reports/ote_regime_slices/v1_v2_comparison/run_summary.json`
- `model_testing/reports/ote_regime_slices/v1_v2_comparison/slice_report.csv`
- `model_testing/reports/ote_regime_slices/v1_v2_comparison/composite_bucket_winners.csv`
- `models/ote_model_registry_v1_v2_candidates.json`

Run timestamp:

- Generated at `2026-04-03T00:13:13.844259+00:00`
- Localized to `2026-04-02 20:13:13 America/New_York`

Run settings:

- models compared: `10`
- slice families: `composite_regime`, `trend_regime`, `vol_regime`, `session_regime`, `stress_regime`, `year`
- bootstrap iterations for AP confidence intervals: `200`
- regime-threshold sufficiency cutoff: `50` positive events

Confidence note:

- A "confident win" below means the winner's AP confidence interval lower bound exceeded the runner-up AP confidence interval upper bound for that slice.

## Executive Summary

- Long side: `long_ote_tcn_v2_candidate` is the clear leader and should advance to threshold and abstain policy search.
- Short side: `short_ote_tcn_v2_candidate` is the clear leader and should advance to threshold and abstain policy search.
- On both sides, `TCN v1` remains the most credible challenger and should stay in the next evaluation pass.
- `XGBoost` remains useful as a baseline, but the current `TCN v2` artifacts dominate the slice report.
- `long_ote_lstm_v2_candidate` regressed materially and should not advance.
- Thin composite buckets should not drive regime-threshold decisions; they should fall back to global thresholds in the next step.

## Long Models

### Overall Long Ranking

| Model | Backend | Test AP | Test Event F0.5 | Test Slice Wins | Confident Test Slice Wins | Notes |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `long_ote_tcn_v2_candidate` | TCN | 0.763 | 0.852 | 27 / 27 | 11 | Best long artifact on disk; strongest across nearly every meaningful slice |
| `long_ote_tcn_v1_candidate` | TCN | 0.699 | 0.801 | 0 | 0 | Main runner-up; second place in 19 / 27 test slices |
| `long_ote_xgb_v1_candidate` | XGBoost | 0.667 | 0.704 | 0 | 0 | Best XGBoost long artifact; occasional runner-up |
| `long_ote_xgb_v2_candidate` | XGBoost | 0.641 | 0.689 | 0 | 0 | Regressed versus XGB v1 |
| `long_ote_lstm_v1_candidate` | LSTM | 0.635 | 0.797 | 0 | 0 | Reasonable threshold benchmark, but weak AP ranking quality |
| `long_ote_lstm_v2_candidate` | LSTM | 0.324 | 0.504 | 0 | 0 | Large regression versus LSTM v1 |

### Long-Side Findings

- `long_ote_tcn_v2_candidate` won all `27` long-side test slices and all `9` long-side test composite buckets.
- `long_ote_tcn_v2_candidate` produced `11` confident test-slice wins.
- `long_ote_tcn_v1_candidate` was the runner-up in `19` of `27` long-side test slices.
- `long_ote_xgb_v1_candidate` and `long_ote_xgb_v2_candidate` only appeared as occasional runner-ups.
- No LSTM artifact was runner-up in any long-side test slice.

### Long-Side Version Results

- `TCN v2` improved over `TCN v1` by `+0.064` test AP and `+0.051` test event F0.5.
- `XGB v2` underperformed `XGB v1` by `-0.026` test AP and `-0.015` test event F0.5.
- `LSTM v2` underperformed `LSTM v1` by `-0.311` test AP and `-0.292` test event F0.5.

### Long-Side Strongest Evidence

The cleanest long-side wins for `long_ote_tcn_v2_candidate` were:

- `composite_regime = strong_down_high`: AP `0.915`, event F0.5 `0.925`, `459` positives
- `composite_regime = strong_down_medium`: AP `0.719`, event F0.5 `0.828`, `546` positives
- `session_regime = london`: AP `0.851`, event F0.5 `0.908`, `359` positives
- `session_regime = overlap`: AP `0.919`, event F0.5 `0.927`, `271` positives
- `trend_regime = strong_down`: AP `0.774`, event F0.5 `0.855`, `997` positives
- `vol_regime = high`: AP `0.889`, event F0.5 `0.912`, `564` positives
- `vol_regime = medium`: AP `0.706`, event F0.5 `0.822`, `628` positives
- `year = 2023`: AP `0.784`, event F0.5 `0.870`, `254` positives
- `year = 2024`: AP `0.759`, event F0.5 `0.847`, `397` positives

### Long-Side Composite Bucket Sufficiency

Composite buckets with enough positive events to support regime-specific threshold search at `min_positive_events = 50`:

- `ranging_high`: `110`
- `ranging_medium`: `102`
- `strong_down_high`: `459`
- `strong_down_low`: `139`
- `strong_down_medium`: `546`

Composite buckets that are too thin and should fall back to the global threshold:

- `ranging_low`: `12`
- `strong_up_high`: `41`
- `strong_up_low`: `1`
- `strong_up_medium`: `21`

### Long-Side Bottom Line

- Advance `long_ote_tcn_v2_candidate`.
- Keep `long_ote_tcn_v1_candidate` in the next pass as the primary challenger.
- Keep `long_ote_xgb_v1_candidate` only as a secondary baseline if extra comparison coverage is desired.
- Do not advance `long_ote_lstm_v2_candidate`.
- Do not prefer `long_ote_xgb_v2_candidate` over `long_ote_xgb_v1_candidate`.

## Short Models

### Overall Short Ranking

| Model | Backend | Test AP | Test Event F0.5 | Test Slice Wins | Confident Test Slice Wins | Notes |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `short_ote_tcn_v2_candidate` | TCN | 0.738 | 0.787 | 26 / 27 | 12 | Best short artifact on disk by AP; dominates slice report |
| `short_ote_tcn_v1_candidate` | TCN | 0.656 | 0.788 | 0 | 0 | Main runner-up; slightly higher headline test F0.5 but weaker slice profile |
| `short_ote_xgb_v1_candidate` | XGBoost | 0.633 | 0.703 | 1 / 27 | 0 | Lone win came in a thin bucket (`ranging_low`) |
| `short_ote_xgb_v2_candidate` | XGBoost | 0.620 | 0.720 | 0 | 0 | Mixed versus XGB v1; still behind TCNs |

### Short-Side Findings

- `short_ote_tcn_v2_candidate` won `26` of `27` short-side test slices and `8` of `9` short-side test composite buckets.
- `short_ote_tcn_v2_candidate` produced `12` confident test-slice wins.
- The only test slice not won by `short_ote_tcn_v2_candidate` was `composite_regime = ranging_low`, won by `short_ote_xgb_v1_candidate`.
- That `ranging_low` short-side bucket only had `18` positives, so it should not be used as promotion evidence.
- `short_ote_tcn_v1_candidate` was the runner-up in `17` of `27` short-side test slices.

### Short-Side Version Results

- `TCN v2` improved over `TCN v1` by `+0.081` test AP while trailing by only `-0.001` on headline test event F0.5.
- `XGB v2` slightly underperformed `XGB v1` on test AP by `-0.013`, while improving test event F0.5 by `+0.017`.
- Short-side promotion should favor AP quality plus slice robustness, which points to `short_ote_tcn_v2_candidate`.

### Short-Side Strongest Evidence

The cleanest short-side wins for `short_ote_tcn_v2_candidate` were:

- `composite_regime = strong_up_medium`: AP `0.727`, event F0.5 `0.752`, `525` positives
- `composite_regime = strong_up_low`: AP `0.445`, event F0.5 `0.500`, `156` positives
- `session_regime = asia`: AP `0.580`, event F0.5 `0.639`, `245` positives
- `session_regime = london`: AP `0.836`, event F0.5 `0.842`, `358` positives
- `session_regime = new_york`: AP `0.691`, event F0.5 `0.735`, `289` positives
- `stress_regime = normal`: AP `0.729`, event F0.5 `0.763`, `1178` positives
- `trend_regime = strong_up`: AP `0.750`, event F0.5 `0.783`, `985` positives
- `vol_regime = low`: AP `0.434`, event F0.5 `0.493`, `166` positives
- `vol_regime = medium`: AP `0.705`, event F0.5 `0.742`, `593` positives
- `year = 2024`: AP `0.746`, event F0.5 `0.789`, `379` positives
- `year = 2025`: AP `0.748`, event F0.5 `0.785`, `403` positives

### Short-Side Composite Bucket Sufficiency

Composite buckets with enough positive events to support regime-specific threshold search at `min_positive_events = 50`:

- `ranging_high`: `90`
- `ranging_medium`: `95`
- `strong_up_high`: `431`
- `strong_up_low`: `156`
- `strong_up_medium`: `525`

Composite buckets that are too thin and should fall back to the global threshold:

- `ranging_low`: `18`
- `strong_down_high`: `36`
- `strong_down_low`: `3`
- `strong_down_medium`: `13`

### Short-Side Bottom Line

- Advance `short_ote_tcn_v2_candidate`.
- Keep `short_ote_tcn_v1_candidate` in the next pass as the primary challenger.
- Keep `short_ote_xgb_v1_candidate` and `short_ote_xgb_v2_candidate` only as secondary baselines.
- Do not promote the lone `short_ote_xgb_v1_candidate` composite win because it came from a thin bucket.

## Recommendations For Next Steps

### 1. Run Threshold And Abstain Policy Search On The TCN Leader Pair

Primary models to advance:

- `long_ote_tcn_v2_candidate`
- `short_ote_tcn_v2_candidate`

Recommended challengers to keep in the same run:

- `long_ote_tcn_v1_candidate`
- `short_ote_tcn_v1_candidate`

### 2. Keep `min_positive_events = 50`

Reasons:

- It matches the regime-slice run assumptions.
- It correctly treats thin composite buckets as global-threshold fallback cases.
- It avoids overfitting thresholds to very small buckets such as `long strong_up_low` and `short strong_down_low`.

### 3. Use The Candidate Registry And Explicit Model IDs

Do not rely on the threshold-search script defaults for this comparison set.

- The defaults select `active` models by role.
- The `v1_v2_comparison` cohort lives in `models/ote_model_registry_v1_v2_candidates.json`.
- The run should therefore pass explicit `--model-id` values.

Recommended command:

```text
python scripts/run_ote_threshold_policy_search.py --regime-report-root model_testing/reports/ote_regime_slices/v1_v2_comparison --registry-path models/ote_model_registry_v1_v2_candidates.json --output-root model_testing/reports/ote_threshold_policies/v1_v2_tcn_focus --model-id long_ote_tcn_v2_candidate --model-id short_ote_tcn_v2_candidate --model-id long_ote_tcn_v1_candidate --model-id short_ote_tcn_v1_candidate --min-positive-events 50 --min-events-per-month 3.0 --min-trades-per-week 3.0
```

### 4. Promotion Guidance After Threshold Search

- If `TCN v2` still leads after threshold search and abstain evaluation, it should become the promotion favorite on both sides.
- `TCN v1` should remain the fallback challenger unless threshold search materially changes the ranking.
- `long_ote_lstm_v2_candidate` should not move forward without a clear retraining fix.
- `long_ote_xgb_v2_candidate` should not be preferred over `long_ote_xgb_v1_candidate`.
- Short-side XGBoost should remain benchmark-only unless a later backtest overturns the current slice evidence.

### 5. Backtest Expectations

After threshold and abstain policy search:

- write any surviving `global_threshold`, `regime_thresholds`, and `abstain_policy` values back to the chosen registry
- run the walk-forward policy backtest on the shortlisted models
- require the threshold-adjusted policies to improve both event quality and post-cost expectancy before promotion

