# FRVP Friction A/B Study

Generated at 2026-07-19 18:06:12Z.

This study holds the saved FRVP model artifacts and saved regime-labeled prediction roots fixed, then reruns only the threshold/backtest economics layer under two spread-cost modes:

- `session_schedule`: use the ES session spread schedule as the spread cost source.
- `feature_proxy`: use the saved `approx_spread` feature when available, with session spreads as fallback.

## Branch Readout

### Long continuation v3 baseline

- Model: `frvp_long_continuation_xgb_v1`
- Targeted filter preset: `frvp_long_continuation_xgb_overlap_composite_prune_v3`
- Session schedule: net `7001.200`, Sharpe `1.187`, mean cost `3.005`.
- Feature proxy: net `1598.400`, Sharpe `0.441`, mean cost `4.645`.
- Delta (`feature - session`): net `-5402.800`, mean cost `1.640`, preferred arm `session_schedule`.
- Notes: Strongest saved long continuation checkpoint under the current FRVP promotion stack.

### Long reversal full-span control

- Model: `frvp_long_reversal_xgb_v1`
- Targeted filter preset: `frvp_long_reversal_xgb_composite_prune_v3`
- Session schedule: net `6027.000`, Sharpe `1.061`, mean cost `2.771`.
- Feature proxy: net `5764.000`, Sharpe `1.023`, mean cost `4.650`.
- Delta (`feature - session`): net `-263.000`, mean cost `1.879`, preferred arm `session_schedule`.
- Notes: Saved full-span long reversal control from the July 19, 2026 Q11 readout.

### Long reversal recent-2y control

- Model: `frvp_long_reversal_xgb_v1`
- Targeted filter preset: `frvp_long_reversal_xgb_recent_regime_prune_v1`
- Session schedule: net `4193.900`, Sharpe `1.429`, mean cost `2.650`.
- Feature proxy: net `4488.550`, Sharpe `1.564`, mean cost `4.650`.
- Delta (`feature - session`): net `294.650`, mean cost `2.000`, preferred arm `feature_proxy`.
- Notes: Saved recent-regime long reversal control from the July 19, 2026 Q11 readout.

### Short meta sentinel

- Model: `frvp_short_meta_xgb_v1`
- Session schedule: net `1994.550`, Sharpe `0.067`, mean cost `2.825`.
- Feature proxy: net `5323.100`, Sharpe `0.202`, mean cost `4.649`.
- Delta (`feature - session`): net `3328.550`, mean cost `1.824`, preferred arm `feature_proxy`.
- Notes: Current short-side FRVP sentinel from the shadow bundle menu.

### Short reversal control

- Model: `frvp_short_reversal_xgb_v1`
- Session schedule: net `-5346.900`, Sharpe `-0.577`, mean cost `2.895`.
- Feature proxy: net `-8151.400`, Sharpe `-0.830`, mean cost `4.643`.
- Delta (`feature - session`): net `-2804.500`, mean cost `1.748`, preferred arm `session_schedule`.
- Notes: Direct short reversal control to test whether costs are the main failure driver.
