# FRVP Roll Shadow Validation

- Generated: `2026-07-17T10:33:25.913717+00:00`
- Selected models: `frvp_long_continuation_xgb_v1, frvp_long_reversal_xgb_v1, frvp_short_meta_xgb_v1`
- Roll boundary: `2026-06-17T00:00:00+00:00` from `ESM6` to `ESU6`
- Validation passed: `True`
- Health error count: `0`
- Feature parity attempted: `True`
- Feature parity passed: `True`

## Runtime Transition

- Pre-roll dashboard contract: `ESM6`
- Post-roll dashboard contract: `ESU6`
- Dashboard contract switch recorded: `True`

## Decision Counts

- Pre-roll: `{'shadow': 6}`
- Post-roll: `{'shadow': 9}`
- Total persisted signal decisions: `15`

## Blocking Findings

- None

## Advisories

- This replay validates the FRVP shadow runtime against a historical contract switch using tagged contract_symbol data.
- It does not remove the current live-ops requirement to switch the active IBKR front-month contract explicitly at roll time.
