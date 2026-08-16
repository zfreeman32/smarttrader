# FRVP Regime-Gated Deployment / Concentration Study

Generated at 2026-07-21 03:21:25Z.

## Frozen controls and candidate runs

- `long_continuation_v3_baseline` (long_continuation_baseline): net `7001.20`, Sharpe `1.187`, WFE `2.344`, profitable-quarter share `0.625`, largest-trade share `0.0643`, accepted `True`.
- `long_reversal_fullspan_baseline` (long_reversal_fullspan_baseline): net `6027.00`, Sharpe `1.061`, WFE `-3.797`, profitable-quarter share `0.600`, largest-trade share `0.0938`, accepted `False`.
- `long_reversal_fullspan_sdm_asia_prune_v1` (long_reversal_fullspan_candidates): net `6245.15`, Sharpe `1.138`, WFE `-4.334`, profitable-quarter share `0.600`, largest-trade share `0.0905`, accepted `False`.
- `long_reversal_fullspan_watch_prune_v1` (long_reversal_fullspan_candidates): net `5009.50`, Sharpe `0.965`, WFE `-3.245`, profitable-quarter share `0.600`, largest-trade share `0.1129`, accepted `False`.
- `long_reversal_recent2y_baseline` (long_reversal_recent2y_baseline): net `4193.90`, Sharpe `1.429`, WFE `2.413`, profitable-quarter share `0.500`, largest-trade share `0.1279`, accepted `False`.
- `long_reversal_recent2y_q10_v1` (long_reversal_recent2y_candidates): net `4216.40`, Sharpe `1.422`, WFE `2.359`, profitable-quarter share `0.500`, largest-trade share `0.1272`, accepted `False`.
- `long_reversal_recent2y_sdh_overlap_prune_v1` (long_reversal_recent2y_candidates): net `3677.05`, Sharpe `1.480`, WFE `2.162`, profitable-quarter share `0.600`, largest-trade share `0.0942`, accepted `True`.
- `long_reversal_recent2y_sdh_overlap_q10_v1` (long_reversal_recent2y_candidates): net `3667.90`, Sharpe `1.478`, WFE `2.152`, profitable-quarter share `0.500`, largest-trade share `0.0944`, accepted `False`.

## Quick reads

### Long reversal full-span frozen control

Worst selected-test pairs:
- `strong_down_medium/asia`: net `-218.15` over `11` trades.
- `ranging_medium/overlap`: net `181.40` over `4` trades.
- `strong_up_high/london`: net `297.35` over `1` trades.
- `ranging_low/asia`: net `302.10` over `6` trades.
- `strong_down_high/overlap`: net `459.90` over `14` trades.
Worst selected-test quarters:
- `2025Q4`: net `-527.55` over `7` trades.
- `2026Q1`: net `-318.20` over `8` trades.
- `2025Q1`: net `-256.30` over `2` trades.
- `2024Q4`: net `-199.85` over `9` trades.
- `2023Q3`: net `-126.90` over `6` trades.
Worst selected-train year/pair pockets:
- `2022/strong_down_medium/new_york`: net `-24460.85` over `369` trades.
- `2022/strong_down_medium/london`: net `-15422.80` over `172` trades.
- `2022/ranging_medium/new_york`: net `-7590.90` over `146` trades.
- `2021/ranging_medium/new_york`: net `-6027.60` over `204` trades.
- `2022/strong_down_high/london`: net `-3394.75` over `15` trades.

### Long reversal recent-2y frozen control

Worst selected-test pairs:
- `ranging_low/overlap`: net `23.35` over `1` trades.
- `ranging_medium/overlap`: net `113.70` over `2` trades.
- `strong_up_high/overlap`: net `206.05` over `3` trades.
- `strong_up_high/london`: net `297.35` over `1` trades.
- `strong_down_medium/overlap`: net `404.50` over `10` trades.
Worst selected-test quarters:
- `2025Q1`: net `-469.20` over `8` trades.
- `2025Q4`: net `-240.90` over `6` trades.
- `2024Q4`: net `-230.20` over `8` trades.
- `2026Q1`: net `-33.60` over `4` trades.
- `2024Q2`: net `645.10` over `6` trades.
Worst selected-train year/pair pockets:
- `2022/strong_down_medium/london`: net `-2249.35` over `19` trades.
- `2022/strong_down_medium/new_york`: net `-1287.80` over `52` trades.
- `2023/strong_down_medium/london`: net `-636.60` over `4` trades.
- `2022/strong_down_high/london`: net `-549.30` over `2` trades.
- `2026/strong_down_high/overlap`: net `-451.30` over `2` trades.
