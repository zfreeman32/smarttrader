# FRVP, ICT, and OTE Trading Logic

This note is a trader-facing summary of how the repo thinks about markets.

It is intentionally focused on trading logic, not model training, data prep, or backtesting mechanics.

## 1. The Big Picture

The workflow is built around a simple idea:

- Do not trade random bars.
- Wait for price to reach an area that matters.
- Read whether price is rejecting that area or accepting it.
- Trade either the reversal, the continuation pullback, or the breakout.

In plain English, the stack is trying to answer:

1. Where is price relative to meaningful structure?
2. Is the market balanced, trending, or running stops?
3. Is this a place to fade, follow, or stand aside?

## 2. What "OTE Workflow" Means Here

In this repo, "OTE workflow" is best understood as the decision process around a trade, not just one narrow pattern.

The important executable trade types used in that workflow are:

| Trade type | Plain-English meaning | Typical job |
| --- | --- | --- |
| Reversal | Price overextends into a key level and rejects | Fade the move back into balance or back toward the opposite side |
| Continuation pullback | Price shows strength, pulls back into a good area, then resumes | Join the trend after the pullback |
| Breakout | Price leaves a balance area with commitment | Go with the expansion if acceptance is real |

`Confirmation / confluence` is still important, but it is a filter layer rather than a fourth trade type. It helps decide whether a reversal, continuation, or breakout idea is worth taking.

So the workflow is not "buy because a model said so."

It is:

1. Find context.
2. Find a setup.
3. Check whether price behavior supports that setup.
4. Define the invalidation point.
5. Aim for the next logical target.

## 3. The Core OTE Reading Sequence

This is the simplest trader version of the workflow:

1. Start with context.
   Look at session, higher-timeframe direction, and obvious reference levels.
2. Mark the important prices.
   Prior highs/lows, session levels, value-area edges, fair value gaps, order blocks, and obvious swing points.
3. Decide what type of day it looks like.
   Balanced day, trend day, stop-run day, breakout day, or choppy/noisy day.
4. Wait for one of the three trade types.
   Reversal, continuation pullback, or breakout.
5. Enter near structure, not in the middle.
   Good entries usually happen on a retest, reclaim, pullback, or clean break with follow-through.
6. Define the "wrong" point first.
   If price gets back through the level that made the trade idea valid, the trade is wrong.
7. Target the next magnet.
   That might be POC, VAH/VAL, the next HVN, an opposing liquidity pool, or the next obvious swing level.

## 4. FRVP Logic in Plain English

FRVP stands for Fixed Range Volume Profile.

It is a way of answering:

- Where did the market actually spend the most business?
- Where did it reject price quickly?
- Where is "fair value" versus "too far"?

### FRVP concepts that matter

| Term | Plain-English meaning |
| --- | --- |
| POC | The price where the most business was done |
| VAH / VAL | The upper and lower edges of the main value area |
| HVN | A heavy-volume area; price often gets attracted to it or slows there |
| LVN | A thin-volume area; price often moves through it quickly |
| Balanced profile | The market found agreement and rotated around fair value |
| Trend profile | The market accepted higher or lower prices and did not stay balanced |

### What FRVP is trying to trade

FRVP is mostly about auction behavior:

- If price is still inside value, fade the edges back toward fair value.
- If price cleanly leaves value and accepts higher or lower prices, go with the move.
- If price pokes outside value and fails, fade it back inside.

### Main FRVP setups used here

1. Mean reversion inside a balanced profile.
   If price reaches VAH or VAL and there is no real acceptance outside, the idea is a move back toward POC.
2. Breakout retest after acceptance.
   If price breaks out of value, then comes back and holds the old edge, the idea is continuation.
3. Value-area breakout with force.
   If price leaves value with strong participation, treat it as initiative activity and follow the move.
4. Failed auction.
   If price looks above or below value, cannot hold there, and falls back inside, that is a reversal setup.
5. LVN traverse.
   Thin zones often act like air pockets. If price enters one with momentum, it can travel quickly through it.
6. HVN magnet.
   If price is between accepted volume nodes, the nearest node can act like a pull target.

### The trader's FRVP mindset

Ask:

- Are we rotating inside value or accepting new value?
- Is price being rejected at the edge, or is it building above/below it?
- Is the next move likely back to POC, or onward to the next node?

## 5. ICT Logic in Plain English

ICT logic is built around liquidity, structure, and displacement.

It is trying to answer:

- Where are stops likely sitting?
- Did price run those stops?
- Did price reclaim the move, or did it actually accept the new level?
- Is there a clean imbalance or order block to enter from?

### ICT concepts that matter

| Term | Plain-English meaning |
| --- | --- |
| Liquidity sweep | Price runs a prior high or low to trigger stops |
| Displacement | A strong push away from a level; shows urgency and intent |
| FVG | A fast-move imbalance area that price may revisit |
| Order block | The last meaningful opposing candle area before displacement |
| MSS / CHoCH | A structure shift that hints the move may be changing direction |
| Premium / discount | Expensive area for shorts, cheap area for longs, relative to the current swing |
| Session manipulation | A false move around the open that sets up the real move |

### What ICT is trying to trade

ICT logic is less about "fair value" in the auction sense and more about this sequence:

1. Price attacks a pool of stops.
2. The market reveals whether that raid was a trap or a true breakout.
3. The trade comes from the reclaim, the structure shift, or the pullback into imbalance.

### Main ICT setups used here

1. Sweep reclaim.
   Price runs a high or low, then closes back the other way. That suggests the raid failed.
2. Sweep, displacement, then FVG retrace.
   Stops get run, price launches away, then pulls back into the imbalance for entry.
3. Order block retest after structure shift.
   Structure changes, then price returns to the origin area before continuing.
4. IFVG reversal.
   A failed gap zone flips and becomes a reversal area.
5. Premium/discount continuation pullback.
   The move is already directional; the entry comes from a better-priced retrace, not from chasing the impulse.
6. Session-open manipulation reversal.
   The open runs an obvious level, traps traders, then reverses.
7. Displacement continuation after a raid.
   Price raids one side, then drives hard in the opposite direction and keeps going.

### The trader's ICT mindset

Ask:

- Which side's stops just got taken?
- Did price reclaim the level or keep accepting beyond it?
- Do we have real displacement, or just noise?
- Is the pullback returning into a useful entry zone?

## 6. How FRVP and ICT Work Together

They are different lenses on the same market.

FRVP tells you:

- where business was accepted
- where price is balanced
- where price is likely to rotate or get pulled

ICT tells you:

- where liquidity sits
- whether stops were raided
- whether a move is a trap, reversal, or continuation

When both line up, the trade is usually cleaner.

Examples:

- ICT sweep reversal at the same place FRVP shows rejection from value edge
- ICT continuation pullback into an FVG while FRVP shows acceptance outside value
- Session-open stop raid that also rejects a known HVN or value-area extreme

## 7. What Matters Most to a Trader

If you strip all the tooling away, the important parts are:

- Context first
- Entry near structure
- Clear invalidation
- Trade type clarity
- Patience when the market is in the middle of nowhere

The workflow is strongest when it can clearly say one of these:

- "Price rejected a key area, so fade it."
- "Price accepted a new area, so follow it."
- "Price is pulling back into a good spot inside an existing move, so join it."

If it cannot say one of those clearly, the best trade is often no trade.

## 8. Short Summary

- OTE is the overall trade-selection workflow.
- FRVP is the volume-profile lens: value, rejection, acceptance, node-to-node movement.
- ICT is the liquidity-and-structure lens: sweeps, displacement, gaps, order blocks, and session traps.
- The important executable trade types are reversal, continuation pullback, and breakout; confirmation through confluence is a filter, not a fourth trade type.
- The goal is not to predict every bar. The goal is to wait for a meaningful area, read the reaction, and act only when the story is clean.
