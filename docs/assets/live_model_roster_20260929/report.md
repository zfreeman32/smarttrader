# Live-app model chart — 2026-09-29

App recommendation: six repaired candidates in Research after diagnostic staging; 21 legacy comparisons behind a toggle; paused S4 and retired FRVP short reversal hidden. This report does not install them.

No candidate is newly cleared for qualified scored collection. Full live-feature parity and collection gates remain pending. Defined outcome contracts and saved thresholds alone are insufficient.

Accuracy is unweighted correct binary labels / all held-out test rows at the saved threshold. It is NOT in-sample fit accuracy, tolerance-based event precision, or trade win rate. Baseline is accuracy from always choosing the more common label.

Current P&L belongs only to the corrected S3 full-policy version. The five other replacements are not backtested. Historical policy results are separate, unverified legacy evidence, not replacement performance.

Legacy P&L uses the atlas's app-linked policy run, not the best sweep. Short OTE breakout uses the named July 14 shadow run because no atlas app-linked curve exists. Rows have different periods and policies; do not rank or sum ticks, pips and dollars.

Legacy training accuracy can be inflated by the data/history issues identified in B2. High OTE accuracy does not certify live profitability or transfer EURUSD OTE results to ES.

Among repaired candidates, S3 has the strongest saved weighted test AUC (0.619) and is the only one above its unweighted majority-label accuracy baseline, but its full-policy backtest loses money. FRVP long continuation/S5 show modest discrimination; TCN/ICT are close to 0.5 AUC. These are research priorities, not trading endorsements.

Coverage cells describe selected representatives, not an assertion that no archived model exists. Setup detectors/family models do not count as dedicated setup specialists. Background/OTE training is not a blanket prerequisite for the six selected ES branches.

| Model | Status | Threshold | Test accuracy | Majority baseline | Weighted AUC | Net P&L |
|---|---|---:|---:|---:|---:|---|
| FRVP Long Continuation XGB | Repaired research candidate | 0.62 | 52.7% | 55.2% | 0.5597 | Not backtested |
| FRVP Long Meta XGB | Legacy diagnostic | 0.50 | 57.3% | 54.0% | 0.5971 | +9,216.05 ticks (legacy) |
| FRVP Long Reversal XGB | Legacy diagnostic | 0.62 | 50.2% | 50.5% | 0.6406 | +3,677.05 ticks (legacy) |
| FRVP Long S1 XGB | Legacy diagnostic | 0.68 | 58.4% | 59.0% | 0.6062 | +178.90 ticks (legacy) |
| FRVP Long S2 XGB | Legacy diagnostic | 0.71 | 58.0% | 60.0% | 0.5483 | +3,499.35 ticks (legacy) |
| FRVP Long S3 XGB | Repaired research candidate | 0.65 | 58.8% | 55.5% | 0.6190 | -$1,460.63 (current full policy) |
| FRVP Long S4 XGB | Paused | 0.23 | 58.0% | 60.0% | 0.5419 | +392.75 ticks (legacy) |
| FRVP Long S5 XGB | Repaired research candidate | 0.62 | 54.3% | 54.7% | 0.5509 | Not backtested |
| FRVP Long S6 XGB | Legacy diagnostic | 0.62 | 55.6% | 53.5% | 0.7007 | -2,897.15 ticks (legacy) |
| FRVP Short Continuation TCN | Repaired research candidate | 0.56 | 50.4% | 50.4% | 0.5206 | Not backtested |
| FRVP Short Meta XGB | Legacy diagnostic | 0.50 | 50.7% | 51.5% | 0.5252 | +1,829.05 ticks (legacy) |
| FRVP Short Reversal XGB | Retired | 0.68 | 52.7% | 51.3% | 0.6693 | -3,476.05 ticks (legacy) |
| ICT Long Continuation XGB | Repaired research candidate | 0.56 | 49.8% | 50.2% | 0.5107 | Not backtested |
| ICT Long Meta XGB | Legacy diagnostic | 0.32 | 57.6% | 65.3% | 0.6749 | +7,461.25 ticks (legacy) |
| ICT Long Reversal XGB | Legacy diagnostic | 0.32 | 63.6% | 70.4% | 0.6827 | +5,471.25 ticks (legacy) |
| ICT Short Continuation XGB | Legacy diagnostic | 0.44 | 50.0% | 53.1% | 0.5532 | +500.75 ticks (legacy) |
| ICT Short IFVG reversal XGB | Legacy diagnostic | 0.35 | 64.5% | 67.0% | 0.6019 | -65.70 ticks (legacy) |
| ICT Short Meta XGB | Legacy diagnostic | 0.32 | 60.2% | 67.1% | 0.6528 | +3,906.70 ticks (legacy) |
| ICT Short Premium/discount XGB | Repaired research candidate | 0.53 | 50.3% | 53.0% | 0.5140 | Not backtested |
| ICT Short Reversal XGB | Legacy diagnostic | 0.29 | 62.5% | 70.3% | 0.6650 | +3,052.15 ticks (legacy) |
| ICT Short Sweep/reclaim XGB | Legacy diagnostic | 0.29 | 67.4% | 74.1% | 0.7147 | +3,049.25 ticks (legacy) |
| OTE Long Breakout TCN | Legacy diagnostic | 0.92 | 99.4% | 97.3% | 0.9996 | -4,704.25 pips (legacy) |
| OTE Long Meta TCN | Legacy diagnostic | 0.86 | 97.5% | 94.0% | 0.9792 | +3,841.45 pips (legacy) |
| OTE Long Reversal TCN | Legacy diagnostic | 0.80 | 98.7% | 98.2% | 0.9766 | +5,468.55 pips (legacy) |
| OTE Long Union TCN | Legacy diagnostic | 0.89 | 97.4% | 94.0% | 0.9780 | +5,658.55 pips (legacy) |
| OTE Short Breakout TCN | Legacy diagnostic | 0.86 | 98.5% | 97.3% | 0.9997 | -872.80 pips (legacy) |
| OTE Short Meta TCN | Legacy diagnostic | 0.89 | 97.5% | 94.2% | 0.9801 | +3,903.55 pips (legacy) |
| OTE Short Reversal XGB | Legacy diagnostic | 0.77 | 98.8% | 98.3% | 0.9775 | +5,453.65 pips (legacy) |
| OTE Short Union TCN | Legacy diagnostic | 0.92 | 97.5% | 94.2% | 0.9797 | +2,480.80 pips (legacy) |

## Long/short coverage

| Strategy | Family/setup | Long | Short |
|---|---|---|---|
| FRVP | Continuation | ID 2: repaired XGB | ID 1: repaired TCN |
| FRVP | Reversal | ID 10: legacy XGB | Retired XGB; no eligible replacement |
| FRVP | Meta (pooled) | ID 3: legacy XGB | ID 4: legacy XGB |
| FRVP | Setup S1 | ID 8: legacy XGB | No retained dedicated specialist |
| FRVP | Setup S2 | ID 5: legacy XGB | No retained dedicated specialist |
| FRVP | Setup S3 | ID 6: repaired XGB | No retained dedicated specialist |
| FRVP | Setup S4 | Paused XGB | No retained dedicated specialist |
| FRVP | Setup S5 | ID 7: repaired XGB | No retained dedicated specialist |
| FRVP | Setup S6 | ID 9: legacy XGB | No retained dedicated specialist |
| ICT | Continuation | ID 11: repaired XGB | ID 14: legacy XGB |
| ICT | Reversal | ID 13: legacy XGB | ID 16: legacy XGB |
| ICT | Meta (pooled) | ID 12: legacy XGB | ID 15: legacy XGB |
| ICT | Premium/discount | No retained specialist | ID 17: repaired XGB |
| ICT | IFVG reversal | No retained specialist | ID 18: legacy XGB |
| ICT | Sweep/reclaim | No retained specialist | ID 19: legacy XGB |
| ICT | Sweep + displacement + FVG | No retained specialist | No retained specialist |
| ICT | OB retest after MSS | No retained specialist | No retained specialist |
| ICT | Session-open pre-IB | No retained specialist | No retained specialist |
| ICT | Session-open post-IB | No retained specialist | No retained specialist |
| ICT | Displacement continuation after raid | No retained specialist | No retained specialist |
| ICT | Classic breaker | Excluded research branch | Excluded research branch |
| OTE | Reversal | ID 22: legacy | ID 26: legacy |
| OTE | Breakout | ID 20: legacy | ID 24: legacy |
| OTE | Meta (pooled) | ID 21: legacy | ID 25: legacy |
| OTE | Union (pooled) | ID 23: legacy | ID 27: legacy |
| OTE | Continuation pullback / entry | Archived experiments; no selected representative | Archived experiments; no selected representative |
