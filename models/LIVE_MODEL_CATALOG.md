# Live Model Catalog

Updated on `2026-05-27` using:

- `model_testing/reports/MODEL_LEADERBOARDS_20260527.md`
- best saved threshold/backtest summaries copied into `model_testing/reports/ote_*/*/multifamily_live_v1`

## Champion

These models are in `models/ote_model_registry_live_multifamily.json` with `status="active"` and are the live emitters for the current `ote_live` app.

| Model ID | Deploy path | Family | Why it made the live set |
| --- | --- | --- | --- |
| `short_reversal_xgb_v2_20260525` | `models/live/short_reversal_xgb` | reversal | Post-training leaderboard rank `#1`; strongest short-side live champion. |
| `long_reversal_tcn_v2_20260525_narrow48` | `models/live/long_reversal_tcn` | reversal | Post-training leaderboard rank `#2`; strongest long-side live champion. |
| `short_ote_meta_tcn_champion` | `models/live/short_meta_tcn` | meta tcn | Post-training leaderboard rank `#7`; passed `6/6` gates. |
| `long_ote_union_tcn_candidate_20260523` | `models/live/long_union_tcn` | long union | Post-training leaderboard rank `#8`; best extra long performer outside reversal/meta/breakout. |
| `short_ote_union_tcn_candidate_20260520` | `models/live/short_union_tcn` | short union | Post-training leaderboard rank `#9`; best extra short performer outside reversal/meta. |

Primary models:

- Long primary: `long_reversal_tcn_v2_20260525_narrow48`
- Short primary: `short_reversal_xgb_v2_20260525`

## Candidate

These models are in the live registry with `status="candidate"` so they still load into the runtime, but only as shadow models.

| Model ID | Deploy path | Family | Why it stays shadow-only |
| --- | --- | --- | --- |
| `long_ote_meta_tcn_champion` | `models/live/long_meta_tcn` | meta tcn | Strong training profile, but only `4/6` post-training gates. |
| `long_breakout_tcn_champion` | `models/live/long_breakout_tcn` | long breakout | Best repaired breakout artifact, but still weaker live conversion than the champion set. |
| `long_breakout_xgb_v1` | `models/live/long_breakout_xgb` | long breakout | Included as the second long-breakout comparison model even though post-training performance is weak. |

## Legacy

For the current live deployment, every model artifact not listed in `models/ote_model_registry_live_multifamily.json` should be treated as legacy.

Important examples moved out of the active live set:

- `long_ote_tcn_v2_candidate`
- `short_ote_tcn_v2_candidate`
- `long_reversal_tcn_champion`
- `short_reversal_xgb_v1`
- `short_breakout_tcn_champion`
- `short_reversal_tcn_champion`
- `long_reversal_tcn_v3_20260526_overlap_ny_dd_repair`

Historical artifacts and report roots were kept for auditability, but they are no longer part of the current `ote_live` deployment registry.

## Layout

The live app now loads the deployed artifacts from the flat `models/live/` directory so the runtime no longer depends on dated experiment folder names.
