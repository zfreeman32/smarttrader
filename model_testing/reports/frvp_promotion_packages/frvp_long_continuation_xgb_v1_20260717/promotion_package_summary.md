# FRVP Promotion Package

- Package status: `finalized_without_human_same_contract_signoff`
- Promotion decision: `pending_human_same_contract_signoff`
- Model: `frvp_long_continuation_xgb_v1`
- Generated: `2026-07-17T10:34:13.955655+00:00`

## Promotion Readout

- Paper-trading gate passed: `True`
- Sharpe: `1.226`
- DSR: `1.061`
- WFE: `2.459`
- Max drawdown pct: `9.90`
- Trades: `524`

## Threshold vs Prune

- Static selected policy: `global_threshold`
- Static selector note: `no_non_global_policy_qualified_against_test_baseline`
- Walk-forward dominant policy: `global_threshold`
- Walk-forward static policy share: `0.7920`
- Conclusion: The saved v3 contract is a hard-pruned base global-threshold policy, not a new regime-threshold winner. On the held-out static test, global_threshold stayed positive (+11.49 ticks expectancy, +264.3 ticks net) while regime_threshold was negative (-4.34 ticks expectancy, -681.3 ticks net), so no non-global policy qualified. Walk-forward still mixed in regime_threshold on 109 of 524 selected trades, but the dominant contract was the hard-pruned global threshold (415 trades, 79.20% share). Relative to the unpruned refresh baseline, the targeted prune sharply reduced trade count (1609 -> 524), lifted expectancy (7.75 -> 13.72 ticks), improved Sharpe (0.604 -> 1.226), improved DSR (0.585 -> 1.061), and cut drawdown (10200.85 -> 1540.10 ticks).

## Validation Evidence

- Placebo passed: `True` with gap `0.2368`
- Roll audit passed: `True` with minimum fold gap `10238` bars
- Roll shadow validation passed: `True` at `2026-06-17T00:00:00+00:00`

## Open Items

- Human same-contract TradingView profile signoff is intentionally deferred for now.
- The historical roll replay validated the FRVP shadow runtime across a real contract switch, but the live IBKR front-month handoff still remains an explicit operator action rather than a fully automatic collector roll.
