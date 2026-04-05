# OTE Model Artifact Comparison

This README documents the six on-disk `v1` and `v2` OTE model roots currently present under `models/` and the comparison registry you can use for `model_testing`.

The numbers below are pulled from each artifact's `training_summary.json` and `model_config.json`, not from older planning notes.

## Scope

- Six root folders are present:
  - `ote_full_lstm_v1`
  - `ote_full_lstm_v2`
  - `ote_full_tcn_v1`
  - `ote_full_tcn_v2`
  - `ote_full_xgb_v1`
  - `ote_full_xgb_v2`
- Those six roots contain ten actual model artifacts:
  - LSTM roots are long-only.
  - TCN and XGBoost roots each contain both `long_ote` and `short_ote`.
- The existing `ote_model_registry.json` is your active/promotion-oriented registry.
- The new `ote_model_registry_v1_v2_candidates.json` is the comparison registry for regime slicing and follow-on testing.

## Inventory

| Root | Targets on disk | Backend | Internal `config.output_root` | Notes |
|---|---|---|---|---|
| `ote_full_lstm_v1` | `long_ote` | `torch/lstm` | `models/ote_full_lstm_v1` | Older long-only benchmark snapshot. |
| `ote_full_lstm_v2` | `long_ote` | `torch/lstm` | `models/ote_full_lstm_v2_champion_candidate` | Newer long-only LSTM candidate, but metrics regressed sharply. |
| `ote_full_tcn_v1` | `long_ote`, `short_ote` | `torch/tcn` | `models/ote_full_tcn_v3_tail` | Saved under a `v1` directory but the training summary points at a later internal run name. |
| `ote_full_tcn_v2` | `long_ote`, `short_ote` | `torch/tcn` | `models/ote_full_tcn_v4` | Strongest overall root on disk. |
| `ote_full_xgb_v1` | `long_ote`, `short_ote` | `xgboost` | `models/ote_full_xgb_v2` | Older XGBoost snapshot. |
| `ote_full_xgb_v2` | `long_ote`, `short_ote` | `xgboost` | `models/ote_full_xgb_v3` | Newer XGBoost snapshot. |

## Long OTE Comparison

Feature counts for older `v1` artifacts are derived from `selected_feature_names` because those summaries do not persist `model_config.feature_count`.

| Artifact | Backend | Window | Features | Trials | Training schedule | Threshold | CV AP | CV event F0.5 | Test AP | Test event F0.5 |
|---|---|---:|---:|---:|---|---:|---:|---:|---:|---:|
| `ote_full_tcn_v2/long_ote` | `tcn` | 20 | 59 | 20 | `40 ep, 4/18/10/8` | 0.71 | 0.6254 | 0.7738 | 0.7625 | 0.8518 |
| `ote_full_tcn_v1/long_ote` | `tcn` | 24 | 48 | 20 | `40 ep, 4/18/10/8` | 0.68 | 0.5693 | 0.7231 | 0.6987 | 0.8013 |
| `ote_full_lstm_v1/long_ote` | `lstm` | 28 | 40 | 20 | `40 ep, auto schedule` | 0.50 | 0.5186 | 0.6907 | 0.6353 | 0.7968 |
| `ote_full_xgb_v1/long_ote` | `xgboost` | 8 | 72 | 20 | `18 ep, lag 0/1/3/7, delta 16` | 0.68 | 0.5476 | 0.7068 | 0.6669 | 0.7041 |
| `ote_full_xgb_v2/long_ote` | `xgboost` | 8 | 23 | 20 | `18 ep, lag 0-7, delta 12` | 0.47 | 0.5138 | 0.6372 | 0.6408 | 0.6889 |
| `ote_full_lstm_v2/long_ote` | `lstm` | 20 | 13 | 48 | `56 ep, 6/26/14/10` | 0.29 | 0.2770 | 0.4336 | 0.3243 | 0.5043 |

## Short OTE Comparison

| Artifact | Backend | Window | Features | Trials | Training schedule | Threshold | CV AP | CV event F0.5 | Test AP | Test event F0.5 |
|---|---|---:|---:|---:|---|---:|---:|---:|---:|---:|
| `ote_full_tcn_v2/short_ote` | `tcn` | 28 | 94 | 20 | `40 ep, 4/18/10/8` | 0.62 | 0.6239 | 0.7473 | 0.7375 | 0.7871 |
| `ote_full_tcn_v1/short_ote` | `tcn` | 28 | 96 | 20 | `40 ep, 4/18/10/8` | 0.56 | 0.4975 | 0.6674 | 0.6562 | 0.7878 |
| `ote_full_xgb_v2/short_ote` | `xgboost` | 8 | 16 | 20 | `18 ep, lag 0-7, delta 12` | 0.50 | 0.5544 | 0.6845 | 0.6200 | 0.7201 |
| `ote_full_xgb_v1/short_ote` | `xgboost` | 8 | 72 | 20 | `18 ep, lag 0/1/3/7, delta 16` | 0.68 | 0.5479 | 0.7008 | 0.6330 | 0.7033 |

## Config Takeaways

- `tcn_v2` is the cleanest upgrade path on the long side. It improved every tracked long metric over `tcn_v1`, including `+0.0561` CV AP, `+0.0638` test AP, and `+0.0505` test event F0.5.
- `tcn_v2` is also the strongest short-side candidate by AP, but `tcn_v1` is effectively tied on held-out short test event F0.5 (`0.7878` vs `0.7871`). Regime slicing should decide whether `tcn_v2`'s stronger ranking metrics or `tcn_v1`'s slightly higher thresholded short score is more robust by bucket.
- `xgb_v1` beats `xgb_v2` on every tracked long-side metric, so the newer XGBoost snapshot is not a clear upgrade for long OTE.
- `xgb_v2` is mixed on short OTE: slightly better held-out short event F0.5 than `xgb_v1`, but worse short test AP and worse short CV event F0.5.
- `lstm_v1` remains a far better long benchmark than `lstm_v2`. Despite more trials, more epochs, a larger hidden size, and a longer staged schedule, `lstm_v2` regressed hard across CV and test.
- The saved directory names are the safest source of truth for testing. Several summaries embed later internal run names (`v3`, `v4`, `champion_candidate`) that do not match the folder names currently on disk.

## Recommended Read On Current Artifacts

- Long default candidate: `ote_full_tcn_v2/long_ote`
- Long fallback benchmark: `ote_full_tcn_v1/long_ote`
- Short comparison pair to settle with regime slices:
  - `ote_full_tcn_v2/short_ote`
  - `ote_full_tcn_v1/short_ote`
- XGBoost should still stay in the comparison set:
  - `ote_full_xgb_v1/long_ote`
  - `ote_full_xgb_v1/short_ote`
  - `ote_full_xgb_v2/short_ote`
- LSTM should stay in the comparison set as benchmark context, not as the default production path.

## Comparison Registry

Use the new comparison registry:

- `models/ote_model_registry_v1_v2_candidates.json`

Design choices:

- It keeps `models/ote_model_registry.json` untouched.
- It uses the actual on-disk artifact paths under the six `v1` and `v2` roots.
- All entries are marked `status: "candidate"` so the regime-slice script will include them by default.
- `global_threshold` is seeded from each artifact's saved training threshold.
- `regime_thresholds` and `abstain_policy` remain `null` until post-slice policy work is done.

## First Model Testing Command

Start the next phase with:

```cmd
python scripts/run_ote_regime_slice_report.py ^
  --registry-path models/ote_model_registry_v1_v2_candidates.json ^
  --output-root model_testing/reports/ote_regime_slices/v1_v2_comparison_20260402 ^
  --bootstrap-iterations 200 ^
  --min-positive-events 50
```

If you want to narrow the first pass, add repeated `--model-id` flags for just the TCN and XGBoost short-side candidates.
