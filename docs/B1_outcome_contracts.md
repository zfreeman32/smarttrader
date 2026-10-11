# B1 outcome contracts (2026-09-16 to 2026-09-29)

Consolidated record of the B1 outcome-contract work, in order: contract design, repository audit, FRVP risk resolution, baseline scope and structural ATR, independent OTE evaluation, the completed matrix (catalogue v3), completion validation and the candidate-specific outcome review. The machine-readable catalogues are `configs/outcome_contracts/b1_v1.json` to `b1_v3.json` (each with a lock). The superseded v1/v2/scope matrix views are kept verbatim at the end.

Contents:
- [B1: frozen evaluation outcomes](#b1_outcome_contract_design_20260916)
- [B1 repository audit and migration report](#b1_outcome_contract_audit_20260916)
- [B1: FRVP S2/S3/S5 and S4 risk resolution](#b1_frvp_risk_resolution_20260917)
- [B1 baseline scope, model/setup diagnostics and structural ATR](#b1_baseline_scope_and_model_setup_audit_20260917)
- [B1 completion: independent OTE evaluation](#b1_ote_evaluation_resolution_20260918)
- [B1 completed contract matrix - 2026-09-18](#b1_outcome_contract_matrix_complete_20260918)
- [B1 completion validation - 2026-09-18](#b1_completion_validation_20260918)
- [B1 candidate-specific historical outcome review — 2026-09-29](#b1_candidate_outcome_review_20260929)
- [B1 active contract matrix - 2026-09-18](#b1_outcome_contract_matrix_20260918)
- [B1 active contract matrix - 2026-09-17](#b1_outcome_contract_matrix_20260917)
- [B1 contract matrix ? 2026-09-16](#b1_outcome_contract_matrix_20260916)


<a id="b1_outcome_contract_design_20260916"></a>

## B1: frozen evaluation outcomes

*Merged 2026-10-09 from `docs/B1_outcome_contract_design_20260916.md`.*

Implemented September 16–17, 2026. This defines evaluation, not a strategy redesign,
model promotion, relabeling exercise, or permission to start scored collection.

The active catalogue is [b1_v3.json](../configs/outcome_contracts/b1_v3.json),
with 45 immutable instrument/setup/side/timeframe entries and a SHA-256 lock.
B1 is complete for the declared baseline: 37 defined risk entries, eight
explicitly excluded classic-breaker/unmatched placeholders, zero unresolved
included bindings. [Full active matrix](#b1_outcome_contract_matrix_complete_20260918).

The [OTE resolution](#b1_ote_evaluation_resolution_20260918) explicitly adopts
independent family economics and the user's 1h reversal / 30m continuation and
breakout ATR choice. [FRVP resolution](#b1_frvp_risk_resolution_20260917) remains
unchanged. [Scope/ATR/model-setup audit](#b1_baseline_scope_and_model_setup_audit_20260917)
keeps classic breaker excluded and **all pooled predictions diagnostic-only**,
even when matched. Source catalogue v1/v2 and scope v1 remain frozen for replay.
New runs use engine `b1-v4`, artifact schema `b1-outcome-artifacts-v2`.
No contract was selected by evaluation P&L. B2/B3 and other activation gates
remain independent prerequisites; B1 does not activate scored collection.

### Four independent questions

| Surface | Meaning | Saved namespace | Prohibited inference |
|---|---|---|---|
| A | Original target used to train the artifact | `training_label_outcomes.csv`; `training_label_outcome`, version/hash/provenance | Positive does not imply target hit, positive markout, or profit |
| B | Independent signal under frozen entry/barriers/timeout | `paper_contract_outcomes.csv`; `paper_contract_outcome` | Target hit does not imply net profit or position admission |
| C | Side-adjusted gross market response, ignoring barriers | `forward_returns.csv` | Positive response is not a trade or P&L |
| D | Position admission, quantity, management, costs, accounting | `simulated_positions.csv`, `simulated_equity_curve.csv` | Rejected positions have no simulated P&L, even when B is TARGET |

`evaluated_signals.csv` joins all four by unique `event_id` for convenience.
The input `SignalEvidence` is also saved verbatim as canonical JSONL. B/C never
read training outcome, probability, quantity, or policy eligibility. D computes
gross P&L from actual entry/exit prices and quantity, never from B's category.

Training labels remain frozen. Legacy label versions/hashes are copied if they
exist; otherwise they remain `UNKNOWN_LEGACY` / null with `unknown_legacy`
provenance. A newly calculated training-summary or registry hash is explicitly a
snapshot hash, **not** a claim to know which historic labeling code produced a
model. `outcome_lineage.inspect_training_lineage` is read-only. B2 must resolve
missing or inconsistent artifact lineage; B1 does not manufacture it.

### Availability and entry

`event_anchor_ts` is historical context. `known_at_ts` is the earliest actual
decision availability proven by the supplied evidence. The evaluator requires:

- Anchor <= known-at; signal-bar close and required confirmation <= known-at.
- Feature maximum source-close <= feature availability <= prediction
  availability <= calibration availability <= policy availability <= known-at.
- Regime, ATR and geometry availability <= known-at; regime must also precede
  policy availability. Every timestamp has an explicit UTC offset.
- A verified availability status and evidence references. Unverified legacy
  predictions produce INVALID, with no executable entry. Timestamp declarations
  enforce the evaluation boundary; they do not replace B2 producer/lineage audits.
- Classic breaker additionally requires source-OB formation < activation <
  retest <= signal close <= decision. The old OB/pivot timestamp cannot stand in
  for breaker availability. Missing lineage prevents even causal diagnostics.

`signal_from_available_inputs` computes the maximum from explicitly supplied
producer timestamps; it cannot convert a target/datetime pair into a causal
signal. For ML predictions a future-dependent *training target* does not by
itself delay an independently causal prediction. A *rule-event replay* that
depends on swing/zigzag confirmation must wait for that confirmation. The
caller must distinguish these cases in its evidence rather than reuse label
timestamps or retrospectively selected entries.

All execution bars have **open timestamps**. Entry is the unadjusted OPEN of
the first independently scheduled bar with `open_ts > known_at_ts`. It is not
the close of the signal bar, next close, or best subsequent price. If a bar is
open-stamped 10:25 and known at its 10:30 close, the 10:30 open is excluded by
the strict inequality: the baseline enters at 10:35. This conservative boundary
choice is deliberate, not a timestamp relabeling. Nonzero decision delay is
retained: a decision recorded at 10:36 cannot enter at 10:35.

Only the first eligible scheduled open can fill (`max_entry_delay_bars=1`). A
missing/corrupt required bar produces DATA_GAP, without looking forward for a
replacement. End of calendar or unavailable calendar coverage produces
NO_ENTRY. Planned maintenance/weekend closures can defer the *scheduled* next
open without turning a genuinely missing expected bar into an acceptable delay.

The native OTE 1-minute research model has its own timeframe contract. It is not
retroactive refinement of a 5-minute decision. B1 supports causal 1-minute
**barrier disambiguation**, not best-price entry refinement. Any future entry
refinement must receive a new explicit version and use bars strictly after the
decision time.

### Frozen risk and management

Resolved B1 risk comes from pre-existing intended geometry, adopted explicitly
as an independent paper baseline:

- FRVP S1/S6: completed 1-hour structural ATR, stop 1 ATR, target 0.8 ATR,
  timeout 60 entry-relative tradable bars. Both the design table and current
  implementation agree on these initial values. They are not the frozen
  August paper trial's 120-bar markout.
- FRVP S2/S3/S5 v2: same causal completed 1-hour ATR, stop 1 ATR, timeout
  96 entry-relative tradable bars. Targets: S2 1 ATR both sides; S3 0.95 long,
  1 short; S5 0.90 long, 0.85 short. Pooled/meta and specialist routing uses
  the selected setup and explicit side, with separate S3/S5 contract IDs.
- FRVP S4 v2: stop 0.90 ATR, target 1 ATR, timeout 48 entry-relative tradable
  bars. This resolves paper economics while its policy-candidacy pause remains.
- OTE reversal: completed 1h Wilder ATR14, stop 2 ATR, target 1 ATR;
  120 5m bars or 600 native-1m bars from actual entry (600 nominal minutes).
- OTE continuation: completed 30m Wilder ATR14, stop 0.7 ATR, target 0.8 ATR;
  36 5m bars (180 nominal minutes).
- OTE breakout and sustained-breakout forecasts: completed 30m Wilder ATR14,
  stop 0.7 ATR, target 0.9 ATR; 24 5m bars (120 nominal minutes). Sustained
  is a forecast target, not future validation or different trade management.
- ICT reversal setups: supplied causal structural stop with distance floor
  max(0.25 signal ATR14, one tick); target is the farther of the supplied
  opposing reference, 2 ATR, and 1.25 times initial risk. Timeout 18 bars.
- ICT session-open pre/post-IB: same initial geometry, timeout 12 bars.
- ICT premium/discount and displacement-after-raid continuation: same stop;
  target floor 2.5 ATR / 1.25R, timeout 24 bars.

ICT values come from `ICTLabelingConfig` and `_resolve_barrier_prices`, but B1
does **not** inherit the labeling engine's volatility-scaled clocks, lunch-clock
freezing, extension, breakeven movement, or trailing after target activation.
This is an explicit fixed-contract migration using its initial parameters.
TARGET here means the initial target was reached; it makes no claim about an
old managed-continuation label or subsequent economic exit.

Stop and target use signal evidence plus the actual entry OPEN. No entry-bar
ATR fallback exists. Invalid ATR identity/availability blocks B. The ATR is
also used as a frozen denominator for C. Futures barriers round outward to
tradable ticks: long stop down/target up, short stop up/target down. This avoids
fractional-tick fills and favorable rounding. FX analytic barriers remain
unrounded; prices are never synthetically spread-adjusted.

All B1 catalogue contracts hold over scheduled closures, matching existing
multi-hour historical windows. Their timeout counts **tradable bars including
the entry bar**, not missing bars or elapsed wall time. The displayed nominal
minutes equal bars × timeframe; actual wall time can be longer across closures.
The engine also supports a separately versioned `exit_before_session_boundary`
contract, exiting at the last scheduled bar's close. It cannot infer a holiday
from missing prices.

Timeout exits at the horizon bar close. Positive/negative/zero side-adjusted
movement produces TIMEOUT_WIN / TIMEOUT_LOSS / TIMEOUT_FLAT. None is TARGET.
At a session-boundary exit the same sign categories apply, and the resolution
method explicitly says `session_boundary_close`.

### Barrier and data-path contract

Open gaps through a stop pay the adverse open. A favorable target gap fills at
the target with no favorable price improvement. Otherwise single-touch bars
resolve at the barrier. When both barriers touch:

1. Accept a complete 1-minute path only when every minute is after the decision,
   all OHLC is valid, coordinates agree, and its aggregate matches the 5-minute
   OHLC. A partial path cannot establish first touch.
2. Use its first touch; a collision inside one minute still resolves stop-first.
3. Without that evidence use adverse-first on the coarse bar, for either side.

`barrier_resolution_method` and `ambiguous_bar_resolution_required` retain the
distinction. There is no TP-first fallback. The enum also reserves AMBIGUOUS for
unresolved imported/future contracts, although this version resolves valid OHLC
collisions adversely instead of excluding their losses.

OHLC is never reindexed/forward-filled to fabricate prices. The tape uses a
separately supplied expected-open schedule with source, version, session IDs,
coverage bounds, and a content hash. That schedule must come from exchange or
captured broker hours, not from the list of bars that happened to arrive. This
avoids extending the application's weekly diagnostic fallback into unverified
holiday coverage. ES/6E require a constant contract-coordinate ID from signal
through resolution. Unknown coordinates, coordinate changes, invalid roll
flags, missing expected bars, and corrupt OHLC censor the affected path.

A paper position already resolved before a later gap retains its result. C
independently censors any longer horizon crossing that gap. An incomplete
holding horizon is DATA_GAP, never an early profitable timeout. Unresolved B
outcomes stay in the denominator of data-quality counts and are excluded from
paper win/target rates. TIMEOUT_FLAT remains in the resolved denominator.

Paper MFE/MAE include the whole exit bar and are explicitly marked
`whole_exit_bar_bounds`; OHLC cannot reveal extrema strictly before an intrabar
exit. They are not an invented tick path.

### Forward diagnostics

For h = 1, 3, 6, 12, 24, 48, the measurement is from entry OPEN to the h-th
tradable bar CLOSE (entry is bar 1), independently of B's exit:

- `forward_return_h`: side × (horizon close − entry), in price units.
- `_units`: the same movement divided by tick/pip increment.
- `_pct`: fractional return (multiply by 100 to display percent).
- `_log`: side × log(close / entry), when prices permit it.
- `_atr`: movement divided by the causal frozen input ATR, if available.
- `_mfe` / `_mae`: maximum favorable (>= 0) / adverse (<= 0) excursion in price
  units over the full horizon; the unsuffixed `forward_mfe`/`forward_mae` use 48.
- `_status`: OK or the specific missing/roll/incomplete-horizon condition.

Costs, targets, labels, policies, and sizing do not change these measurements.
A short's downward response is positive. No field in this namespace is called
P&L.

### Costs and positions

| Instrument | Increment / cash per unit | Session spread: overlap, London, NY, Asia, off-hours | Fixed slippage / round trip | Commission / round trip | ID |
|---|---|---|---|---|---|
| ES | 0.25 / $12.50 per tick per contract | 1, 1, 1, 1.5, 2 ticks **per side** | 0.25 tick | 0.40 tick | `es_session_cost_v1` |
| 6E | 0.00005 / $6.25 per tick per contract | 1, 1, 1, 1.5, 2 ticks **per side** | 0.25 tick | 0.40 tick | `6e_session_cost_v1` |
| EURUSD | 0.0001 / cash conversion unresolved without lot convention | 1, 1.5, 1.5, 2.5, 3 pips **per side** | 0.30 pip | 0.35 pip | `eurusd_session_cost_v1` |

These are the existing configured research assumptions, not a claim about a
broker's actual rates. Code tracing establishes their interpretation: the
legacy evaluator adds **one full configured spread at entry and another at
exit**, and adds the fixed slippage and commission only once. B1 preserves that
conservative full-spread-per-side convention; it does not quietly halve it or
double the round-trip commission. The CLI's spread-dependent slippage
multiplier was zero; B1 freezes zero. There is no volatility or open/close spike
adjustment. Session windows are London 07–16 local, NY 08–17 local, overlap
priority, Asia 19–02 NY, otherwise off-hours, with timezone/DST conversion.

B1 uses the session at **each actual leg**, whereas the legacy fallback often
used the signal's session for both legs. ES/6E preserve the corrected session
cost path, never `approx_spread`. EURUSD explicitly migrates from legacy `auto`
candle-range proxy to the existing session schedule. That migration is part of
the new cost ID and makes old economics non-comparable without rerun.

The separate `PositionContract` freezes policy ID/hash, fixed-barrier simulator
version, maximum positions/quantity, per-model or instrument scope, opposite
position rule, cooldown in minutes, starting equity, and accounting units.
Selection flags/reasons and quantity must be predeclared in input evidence;
the runner does not fit thresholds. B3 must define the prospective policy
comparison and risk restrictions. No portfolio default is invented.

Every in-scope signal can receive B/C independently; excluded/pooled signals
receive C and `NOT_EVALUATED` B, never a D position. D admits chronologically, with
event ID as deterministic tie-breaker. Rejection reasons are retained. A
position whose path becomes unknown is recorded as taken/unresolved with null
P&L and continues consuming capacity; the simulator cannot reject it using
future knowledge of the gap. Only closed positions contribute realized equity.
The manifest reports unresolved exposure so this partial curve cannot be read
as a complete portfolio result.

`gross_pnl = side × (exit − entry) / increment × quantity` (and cash-per-unit
when cash accounting is explicitly available). `total_cost` is the sum of
spread, slippage, commission; `net_pnl = gross_pnl − total_cost`. FX cash mode is
rejected until a lot/currency conversion convention exists. The A7 diagnostic
ledger and production management are separate versioned simulators; their
round-trip spread/sizing assumptions must not be renamed as these B1 results.
No scored A7 registration or frozen paper-trial activation changes here.

### Versioning, artifacts, and running

Changing entry, stop, target, horizon, rounding, costs, management, or calendar
semantics requires a new ID/version and hash. Loading changed bytes against an
existing lock fails. Nested contract fields are immutable; no mutable dicts are
exposed in the frozen types. Future multiple versions should be selected by
explicit ID, never by an implicit "latest" rule.

New output directories must not exist. Each run saves contract and simulator
snapshots, catalogue/lock, original signal evidence, unfilled coarse/fine tape,
expected schedule, calendar metadata, engine source, all four CSV namespaces,
and a manifest with file hashes, version information and a run hash. Old runs
remain reproducible. Replay validates hashes and refuses a different engine;
the archived source must then be used in an isolated checkout.

Input JSON schemas live alongside the catalogue. Run the self-contained
synthetic example (no models or labels are changed):

```powershell
ote_venv/Scripts/python.exe scripts/run_outcome_evaluation.py --signals model_testing/examples/outcome_b1/signals.jsonl --bars model_testing/examples/outcome_b1/bars.csv --schedule model_testing/examples/outcome_b1/schedule.csv --calendar model_testing/examples/outcome_b1/calendar.json --position-contract model_testing/examples/outcome_b1/position_contract.json --output-dir tmp/b1_example
ote_venv/Scripts/python.exe scripts/run_outcome_evaluation.py --replay-run tmp/b1_example --output-dir tmp/b1_example_replay
ote_venv/Scripts/python.exe scripts/audit_outcome_contracts.py
```

The existing shared Phase 7 entrypoint also dispatches B1 directly:

```powershell
ote_venv/Scripts/python.exe scripts/run_ote_policy_backtest.py --b1-input-manifest model_testing/examples/outcome_b1/input_manifest.json --output-root tmp/b1_phase7
```

FRVP and ICT orchestration scripts accept `-OutcomeInputManifest <path>` plus a
new `-BacktestOutputRoot`. This mode evaluates the supplied frozen decisions
directly and does not run regime preprocessing, threshold search or registry
rebuilding. Manifest paths are relative to the manifest file. The direct B1 CLI
also accepts `--input-manifest`.

Historical Phase 6/7 threshold/backtest entrypoints now require explicit
`--legacy-outcomes` (Python APIs: `legacy_outcomes=True`; FRVP/ICT PowerShell
wrappers: `-LegacyOutcomes`). They retain numerical behavior and save a hashed
`legacy_label_horizon_or_breakout_event_v0` snapshot beside every output. New
legacy runs are excluded from B1 promotion and cannot rewrite deployed
threshold policies. Existing archives and trained artifacts are untouched.
Legacy research scripts calling these APIs without opt-in now fail with the
migration command before evaluating; this is intentional enforcement.

### Completion and remaining prerequisites

There are no unresolved included B1 economics bindings. FRVP S2/S3/S5/S4 were
resolved in catalogue v2; independent OTE economics are now resolved in v3.
`scripts/audit_outcome_contracts.py --require-complete` checks all catalogue
bindings and registry targets and fails if an included contract is unresolved.

Classic breaker is excluded and **all** pooled/unmatched predictions remain
diagnostic-only under locked scope v1. Their eight unresolved risk placeholders
are retained for archives, not pending decisions inside this baseline.

Historical label identity and causal feature lineage (including HTF publication
timing), approved market-calendar coverage, FX cash conversion and prospective
full-policy/management comparison remain prerequisites where missing. B2/B3/B4/B5
are not completed here. No qualified collection or frozen paper trial is activated.

UNRESOLVED remains supported for archived or future research contracts. Included
signals can retain C when causal entry evidence is valid, but B is INVALID,
D cannot take a position, and paper win-rate denominators exclude them.


<a id="b1_outcome_contract_audit_20260916"></a>

## B1 repository audit and migration report

*Merged 2026-10-09 from `docs/B1_outcome_contract_audit_20260916.md`.*

**Completion addendum, September 18:** B1 is now complete for its declared scope.
[Independent OTE rules](#b1_ote_evaluation_resolution_20260918) and catalogue v3
resolve the five OTE entries; 37 risk entries are defined and eight placeholders
are explicitly excluded. The user's choice preserves 1h reversal and 30m
continuation/breakout structural ATR. All pooled predictions remain diagnostic-only.
[Current matrix](#b1_outcome_contract_matrix_complete_20260918) and
[completion evidence](audits/b1_completion_20260918.json) supersede earlier open
OTE status below. Earlier audit sections remain a historical record; B2/B3 gates
remain open and no scored collection is activated.

September 18: [baseline scope and ATR audit](#b1_baseline_scope_and_model_setup_audit_20260917)
explicitly excludes classic breaker and makes all pooled predictions diagnostic-only.
The [latest inventory](audits/b1_registry_inventory_20260918.json) records each
model/contract's disposition. The initial ATR audit also missed hour-open
publication of full-hour FRVP ATR; this is now reproduced, corrected only in
the new evaluation producer, and recorded as a B2 historical-lineage issue.

September 17 correction: the initial FRVP barrier audit below omitted the
Phase03 pipeline overrides when recommending 1.2 ATR. Saved control, refresh
and specialist event configuration verifies a 1.0-ATR continuation baseline,
S3-long 0.95, S5-long 0.90, S5-short 0.85, and S4 stop/target 0.90/1.00.
These are now independently frozen in v2. See the
[evidence and resolution](#b1_frvp_risk_resolution_20260917),
[active matrix](#b1_outcome_contract_matrix_20260917) and
[side-aware registry audit](audits/b1_registry_inventory_20260917.json).
The active catalogue has 32 resolved and 13 unresolved entries. B1 remains open.

Audit performed before implementation on September 16, 2026. Scope includes
every root model registry, historical/research registry variants, shared Phase
6/7 scripts, live diagnostic contracts, setup/label engines, cost code, and
design/experiment records. The A1–A11 documents referenced in notes were
consolidated into `docs/live_app_change_journal.md`; their content was inspected
there. No training, label generation, test-period contract tuning, or model
promotion was performed.

The read-only [machine inventory](audits/b1_registry_inventory_20260916.json)
contains **120 model registrations and 28 unique target names**, registry and
record hashes, artifact paths, training-summary snapshots, label metadata,
family bindings, and all matching contract IDs. All 120 have bindings. The
inventory deliberately includes old control/candidate rosters so an old model
cannot silently acquire new label semantics. This is not a count of 120 unique
trained models. Current label hashes are unknown for legacy artifacts unless
explicitly stored; B2 must establish historical producer lineage.

### Inventory of evaluated target families

| Strategy | Registered targets / architecture coverage | Setup routing |
|---|---|---|
| OTE | long/short reversal, breakout, long sustained breakout, long/short `ote`; XGB, TCN, historical LSTM/RNN/challenger variants | reversal, continuation/pullback, breakout, sustained; `ote`/meta/union dispatch to causal setup, otherwise unmatched |
| FRVP | long/short reversal, continuation, meta; XGB/TCN | S1/S4/S6 reversal; S2/S3/S5 continuation |
| FRVP specialists | long reversal_setup1/4/6, long continuation_setup2/3/5 | same-side exact setup contract |
| ICT | long/short reversal, continuation, meta; XGB | eight canonical setup types, each explicit in matrix |
| ICT specialists | short reversal_sweep_reclaim, reversal_ifvg_reversal, continuation_premium_discount_continuation | same-side exact setup contract |
| ICT classic breaker | isolated long/short label/prepared research; no root registered production model | separate classic_breaker contract; no canonical pooling |
| Native 1m research | long reversal TCN candidate and 1m reversal research artifacts | separate 1m identity; not a 5m intrabar price-search path |

ES is the current FRVP/ICT evaluation instrument; 6E has configured transfer
semantics and is explicitly covered without claiming a deployed 6E roster.
EURUSD is the current OTE instrument. MES is an ICT configuration alias but is
not an independently evaluated root registry in this checkout; B1 refuses an
unregistered MES/NQ contract instead of borrowing ES cash/commission units.
FRVP 6b remains deferred/unimplemented, as already recorded in notes.

### Findings by implementation path

| Area | Previous behavior and evidence | B1 action |
|---|---|---|
| Shared Phase 6/7 economics | Both `run_frvp_post_training_eval.ps1` and `run_ict_post_training_eval.ps1` call the OTE threshold/backtest scripts. `attach_forward_trade_outcomes` normally enters at prediction-row close and exits at source_row_idx + `label_max_holding_bars` (default 120). Market loader requests only datetime/close/approx_spread. | Preserve explicit legacy replay; new B1 evaluator requires raw OHLC, independently scheduled opens and availability evidence. No automatic generic horizon conversion. |
| OTE training | `ReversalParams` uses zigzag confirmation and TB 1/2 ATR TP/SL, 120 bars; `assign_entry_labels` searches candidate bars around/before swings, scored using later confirmation/followthrough. Continuation defaults .8/.7 ATR TP/SL, 36 bars; breakout .9/.7, 24 bars; native 1m retunes bar counts. | Labels untouched. Original swing and confirm clocks remain distinct. Label settings are recorded as training semantics, not approved evaluation stops. |
| OTE breakout event enrichment | `ote_breakout_event_metadata.py` joins saved confirm/entry/exit information. `attach_forward_trade_outcomes` can replace the generic close-to-close path with saved event entry/exit. Historical best-entry or label timing alone does not prove causal availability. | Legacy contract records the mixed path. B1 does not treat saved label entry prices as executable; requires decision-time evidence. |
| FRVP setup timestamps | `frvp/setups/detector.py` evaluates current close/profile/participation conditions. Label events use the firing index as both swing and confirm indices, then enter next row's open. Historical profile anchor is contextual, not signal availability. Initial balance requires completion. | Anchor and firing-close/producer availability stored separately; entry follows actual known-at. |
| FRVP barriers | Generic `FRVPLabelingParams`: S2/S3/S5 1.2/1 TP/SL; design 1.5/1. Actual Phase03 pipeline and saved events: S2 1/1, S3 .95/1 long and 1/1 short, S5 .90/1 long and .85/1 short, all 96 bars. S4 actual 1/.9, 48 bars versus design 1.2/.8. | S1/S6 remain frozen v1. S2/S3/S5/S4 resolved in v2 from established configuration with independent B1 execution semantics; side-specific dispatch, no pooled 120-bar substitution. |
| FRVP collision | `_evaluate_event_barriers` checks TP before SL for both sides. | Historical labels preserved; B1 complete causal 1m resolution or adverse-first, with method recorded. |
| FRVP ATR fallback | Labeler can fall back from signal ATR to entry-row ATR, which can contain entry-bar future data. | No such fallback in B1; require causal ATR identity and availability. |
| FRVP hourly ATR publication | `_resample_ohlcv` uses hour-open labels; `map_htf_atr_to_ltf` forward maps the full-hour value immediately. Later 5m constituents can therefore affect earlier rows of the same hour. Units are price, structural timeframe 1h, Wilder ATR14. | Preserve labels/source; new B1 producer uses complete hourly buckets, close/arrival availability and provenance, with no missing-bar fill. Require proof for paper geometry and ATR normalization. B2 must trace historical impact. |
| ICT structure and events | `ict/structure/swings.py` shifts pivot confirmation forward by window. Detectors keep causal IDs and retest state. `ict/setups/detector.py` emits event_time at signal row; source OB/swing times are older. | Validate known-at after confirming close and all producers. Source structure is never an executable decision. |
| ICT classic breaker | Detector records source OB formation, activation, retest, age/count; requires activation before eligible retest and usually rejection close. Isolated research label/prepared branch exists; canonical pooling explicitly disabled. | Explicit baseline exclusion in scope v1; retain detector/artifacts/logging and lineage checks for diagnostics. No invented economics or promotion. |
| ICT label economics | Label engine enters next-row open, initial structural barriers/floors, base 18/12/24 bars; volatility scale .67–1.75; lunch-clock freezing; continuation target activates trailing/extension up to 24 extra bars. `require_intrabar_resolution` can exclude ambiguous bars. | B1 explicitly freezes initial geometry and base entry-relative tradable-bar timeout; no extension/trailing/partial management in B. Those would require a separate D simulator version. |
| ICT label fallback/path | Signal ATR may fall back to entry ATR. Horizon indices count rows and lunch handling; intrabar order accepts nonempty subsets and does not itself prove full minute coverage. Roll-spanning labels are excluded. | Require causal ATR, full finer-bar coverage and coarse agreement, independent schedule, no unknown execution gaps, constant coordinate. |
| Costs | `evaluation_costs.py` sets ES/6E tick increment/value, .25 fixed slip, .40 commission per trade; FX .30/.35. `resolve_spread_cost_mode(auto)` uses session_schedule for ES/6E, feature_proxy for FX. `microstructure.py` returns candle range for FX; futures proxy is clipped to 1–2 ticks. CLI sets spread-slippage multiplier zero. | Futures corrected session costs preserved; FX explicitly moves to the existing session schedule under new ID. Each cost component saved; source prices unchanged. |
| Cost leg interpretation | Shared evaluator adds entry spread + exit spread + fixed slippage + commission; both fallback spreads often use signal session. `ThresholdSearchConfig` library default multiplier .25 differs from CLI zero. | B1 explicitly full spread per leg, fixed costs once, zero multiplier, actual entry/exit sessions. Legacy snapshots retain whatever config was used. |
| Cooldown/overlap | `HardAbstainConfig` normally uses four source-row bars; ICT setup detector cooldown normally six bars plus session fire caps; FRVP label S6 cooldown is configurable. Shared economics emits independently overlapping event records without portfolio admission. | Existing detector labels unchanged; every supplied signal gets independent B/C. D separately freezes eligibility, cooldown, concurrent count, quantity and opposite-side constraints. |
| Gaps/rolls | Shared close-only lookup cannot see missing intrahorizon execution paths. No assertion that a row offset represents contiguous tradable time. FRVP continuous profile slices and ICT label exclusions already protect many roll windows. | Do not weaken those safeguards; B1 additionally censors unknown/mixed coordinates and every missing expected path bar. Planned closures are explicit calendar evidence. |
| Live inference | A4 stores bar/prediction recording time and rejects delayed/backfilled eligibility; A5 immutable setup history; A6 separates scores, crossings, matched decisions and full-policy qualification. Older prediction CSVs have source-row alignment but generally not all availability attestations. | New input schema requires the missing evidence rather than inferring it from labels. A1 collection boundaries and A11 ICT quarantine remain in force. |
| A7 diagnostic simulator | Independent frozen ResearchContract; next-open/actual-time entry, position limits, censoring, round-trip cost fields; diagnostic only pending B1/B3. | Remains a separately named simulator with distinct economics. B1 does not silently bind its draft tick stops to mixed families or activate scored collection. |

### Causally impossible or economically ambiguous prior interpretations

An OTE swing label or retrospectively selected best-entry bar cannot be used as
the entry time of a rule requiring later swing confirmation. A close-derived
FRVP/ICT signal cannot receive an executable fill at that same historical close.
The existing generic shared evaluator also cannot substantiate “setup TP hit”:
it usually evaluates an independently chosen label-length close markout instead.

An independently causal ML prediction *may* precede the future event it was
trained to predict. B1 does not incorrectly move all such predictions to the
future label confirmation. It requires evidence of the actual inputs and
prediction availability; old source_row_idx alone is insufficient.

The FRVP design/implementation risk conflicts and ICT managed-label versus
generic-backtest horizon mismatch are economic ambiguities, not grounds to
relabel trained models. ICT target activation, final managed exit, label quality,
and net trade profit were never logically interchangeable. B1 names them
separately and does not import a trailing-label TP flag as a fixed trade profit.

### Reports that are not B1-comparable

Existing files remain unchanged. The following report families retain their
original meaning and must not be merged into a new B1 baseline:

- `model_testing/reports/ote_threshold_policies/`, `ote_policy_backtests/`:
  label-horizon or saved breakout-event entry/exit, historical cost path.
- `model_testing/reports/frvp_threshold_policies/`, `frvp_backtests/`,
  `frvp_friction_studies/`: mostly pooled 120-bar economics, sometimes explicit
  cost experiments, not each setup's frozen barriers.
- `model_testing/reports/ict_threshold_policies/`, `ict_backtests/` and
  probability-quantile sweeps: generic shared economic path, not ICT
  fixed/managed event-label outcomes.
- Frozen FRVP `frvp_es_paper_signal_20260816`: explicitly signal-close to close
  after 120 completed 5m bars, independent overlap, no stops/targets, no orders.
  Its readiness/trial/activation state is unchanged.
- Frozen ICT `ict_es_paper_signal_20260813`: the long-meta ledger in
  `ote_live/storage/ict_paper_signal.py` explicitly uses signal-close entry and
  a **20-bar** close markout, no stop/target, independently overlapping events.
  It charges two signal-session spreads plus .25 tick slip and .40 tick
  commission. That is not B1's per-setup 12/18/24-bar contract. Its frozen
  runtime manifest and activation state remain unchanged.
- A7 diagnostic replays retain their own contract hashes and cost/admission
  definitions. A7 source examples remain diagnostic examples, not B1/B3
  approved collections.
- The September live audit's positive setup markouts and label-oriented
  reports remain diagnostics; they cannot establish position-level net profit.

Legacy replay entrypoints are explicitly opt-in and add contract IDs/hashes,
training-label aliases and cost components without changing numeric outputs.
Their new B1-ineligibility metadata is intentional; old archived promotion
decisions are not rewritten. New runs cannot overwrite saved run summaries.

### Validation and remaining restrictions

`tests/test_outcome_contracts.py` uses checked-in synthetic long/short OHLC
fixtures and exercises all 14 requested invariants: causal clocks, future input
rejection, fixed geometry, collision ordering, gap/roll censoring, surface
independence, label preservation, accounting reconciliation, saved hashes,
version tamper rejection, and deterministic replay. Additional cases cover
session/weekend holding and exits, incomplete fine data, adverse stop gaps,
unresolved-capacity admission, target-hit/net-loss disagreement, structural
barriers, legacy opt-in, and CLI replay. The existing threshold/backtest/abstain
tests provide numerical compatibility coverage. Exact commands/results are
recorded in the live change journal.

B1 infrastructure is implemented, but B1 remains in progress while included
families have unresolved outcome rules. Finalization requires resolving those
rules or explicitly documenting the affected families' exclusion from the
intended baseline. Implementing a fail-closed UNRESOLVED state is not itself a
resolution of strategy economics. The September 17 FRVP decision resolves only
S2/S3/S5/S4; it does not approve remaining proposals in the design document, repair B2
model lineage, choose B3 portfolio policy, start C1, or change paper-trial
permissions. No production economic result was rerun to choose these rules.


<a id="b1_frvp_risk_resolution_20260917"></a>

## B1: FRVP S2/S3/S5 and S4 risk resolution

*Merged 2026-10-09 from `docs/B1_FRVP_risk_resolution_20260917.md`.*

September 18 addendum: the numerical freeze below remains unchanged. The
[new scope and ATR decision](#b1_baseline_scope_and_model_setup_audit_20260917)
makes all pooled models diagnostic-only, excludes classic breaker/unmatched
predictions, and requires completed-hour ATR provenance. The historical FRVP
label path's hour-open publication issue is recorded for B2. Use the
[current matrix](#b1_outcome_contract_matrix_20260918) for membership.

Decision: freeze the established Phase03 pipeline configuration as **independent
paper-contract v2 economics**, using B1 causal entry and execution semantics.
This resolves these four setups for ES and its configured 6E transfer variant.
It does not assert that their training labels, paper outcomes, forward returns
or simulated net P&L agree. B1 as a whole remains in progress.

### Evidence and decision

The initial B1 audit compared two layers and missed the third:

| Setup / side | Original design: stop / target / bars | Generic labeler | Established pipeline and all three saved event roots | Adopted v2 |
|---|---|---|---|---|
| S2 both | 1 / 1.5 / 96 | 1 / 1.2 / 96 | 1 / 1 / 96 | 1 / 1 / 96 |
| S3 long | 1 / 1.5 / 96 | 1 / 1.2 / 96 | 1 / 0.95 / 96 | 1 / 0.95 / 96 |
| S3 short | 1 / 1.5 / 96 | 1 / 1.2 / 96 | 1 / 1 / 96 | 1 / 1 / 96 |
| S5 long | 1 / 1.5 / 96 | 1 / 1.2 / 96 | 1 / 0.90 / 96 | 1 / 0.90 / 96 |
| S5 short | 1 / 1.5 / 96 | 1 / 1.2 / 96 | 1 / 0.85 / 96 | 1 / 0.85 / 96 |
| S4 both | 0.8 / 1.2 / 48 | 0.9 / 1 / 48 | 0.9 / 1 / 48 | 0.9 / 1 / 48 |

Stop/target coefficients are multiples of the **completed 1-hour structural
ATR available by known-at**, in instrument price units. Horizons count tradable
5-minute bars from actual entry, including the entry bar: 96 = nominal 480
minutes, 48 = nominal 240 minutes. Closures can extend wall time.

Evidence, in order of relevance:

1. `frvp/pipelines/es_primary_phase04.py::_default_phase03_labeling_params`
   explicitly supplies continuation TP 1.00, S3-long 0.95, S5-long 0.90,
   S5-short 0.85, and enables failed-auction labels. The existing
   `test_default_phase03_labeling_params_include_winning_continuation_overrides`
   already asserts these settings. Its historical name does not justify a new
   performance-based choice in B1.
2. `data/labeling/frvp_labeling_engine.py::_barrier_spec` applies these
   setup/side overrides before falling back to generic continuation settings.
   The inherited stops/horizons are 1 ATR/96 bars and 0.9 ATR/48 bars for S4.
3. Read-only configuration inspection of
   `artifacts/frvp_es_primary_current/phase03/es_primary_frvp_events.csv`,
   `artifacts/frvp_es_primary_refresh_20260701/phase03/es_primary_frvp_events.csv`
   and
   `artifacts/frvp_es_primary_setup_targets_20260829/phase03/es_primary_frvp_events.csv`
   gives exactly the values above. Only setup, side, TP/SL coefficients,
   holding bars and exclusion-reason columns were parsed. Returns, labels and
   evaluation P&L were not used to select this decision.
4. The four August specialist training summaries point to the same setup-target
   prepared root. Their generic `label_horizon_assumptions` includes 120 bars;
   that is not the observed event horizon and is not adopted for B1. These
   pointers and hashes do not prove historic training-label provenance; B2
   retains that responsibility.
5. Historical revisions `31e32ba` (July 8) and `a04f7e9` (August 29) already
   contain the same pipeline overrides. E01 establishes the July refresh as
   the research control; E13/E14 reuse its event/label tape for specialists.
   The older design table remains the original proposal, not the effective
   current pipeline configuration. The older audit itself also records the
   pipeline overrides in its implementation snapshot.

The [configuration evidence](audits/b1_frvp_risk_evidence_20260917.json) records
file SHA-256 hashes, unique per-side risk tuples, prepared-root pointers, and
the original v1 catalogue hashes. Each new contract pins that evidence file's
hash in `risk.source`. To repeat this audit without mutating historical inputs:

```powershell
ote_venv/Scripts/python.exe scripts/audit_frvp_outcome_risk.py --output tmp/frvp_risk_reaudit.json
```

The freeze is explicit and independent: future training-label parameter changes
do not update paper contracts automatically. Any later evaluation change needs
a new contract version. No barriers were swept or selected against evaluation
results for this task. 6E uses the same existing instrument-parameterized
pipeline and the design's transfer semantics; no saved 6E performance or model
validation is claimed from the ES evidence.

### Signal availability and execution

`frvp/setups/detector.py` supplies the causal setup semantics:

| Setup | Event anchor | Setup becomes knowable |
|---|---|---|
| S2 | Retest/hold firing bar open | Its close, after an earlier same-session breakout; never that earlier breakout's timestamp |
| S3 | Outside-value displacement firing bar open | Its close, once displacement, overshoot and completed volume expansion are available |
| S4 | Return-inside-value firing bar open | Its close, after the prior outside run and the sweep, reentry-depth and quiet-reentry checks |
| S5 | LVN displacement firing bar open | Its close, once the known LVN proximity and directional displacement are confirmed |

For every model, `known_at_ts` is at least that setup close and the maximum of
all required profile/HTF, ATR, feature, regime, prediction, calibration and
policy availability. Historical profile anchors are context, never executable
decision timestamps. Existing selected-event/tiebreaker semantics stay intact;
pooled/meta inputs need the causal selected setup and actual signal side.

Entry is the first scheduled unadjusted OPEN **strictly after** known-at. With
open-stamped bars and a 10:30 decision, the 10:30 open is excluded; the default
enters at 10:35. Stop/target offsets are frozen from this entry and the supplied
causal ATR. Futures stop/target levels round outward to valid ticks. This
rounding can make small-ATR outcomes coincide across coefficients; the
coefficient contracts still differ. `min_target_rr` belongs to structural
contracts and does not impose 1.25R on these ATR contracts.

Paper management is fixed barriers and timeout; all signals evaluate
independently. Intrabar collisions use a complete consistent causal 1-minute
path if available, otherwise adverse-first. Missing required entry/holding
bars or invalid roll coordinates yield DATA_GAP without forward fill or a
replacement entry. Planned closures use the supplied calendar; holding is
permitted. Timeout uses the horizon close and distinguishes win/loss/flat.

Costs remain separate: `es_session_cost_v1` / `6e_session_cost_v1`, full
configured spread at each leg, 0.25 tick slippage and 0.40 tick commission per
round trip. No candle-range spread proxy is introduced. Optional position
simulation applies its own versioned admission/sizing rules and records gross,
each cost and net P&L. A resolved paper contract does not supply a missing
model-specific policy or permission to trade.

### S4: three separate questions

- **Economics:** resolved at stop 0.90 ATR, target 1.00 ATR, 48 bars.
- **Historical label enablement:** the old control has 1,002 S4 rows with
  `failed_auction_setup_disabled`; July refresh and August specialist tapes
  have zero. E01 explicitly records re-enablement. Keep these old exclusions
  and training labels intact; never recompute them to match B1.
- **Qualified decision candidacy:** remains paused by A9/A10. The current
  provisional policy falls back to S2 and is not a reviewed S4 policy. S4 can
  receive independent paper diagnostics with verified causal input evidence;
  that does not approve live/scored candidacy or populate a new policy.

### Versioning and compatibility

The explicit active release is [b1_v2.json](../configs/outcome_contracts/b1_v2.json)
with its lock. Eight unresolved FRVP v1 entries become twelve resolved v2
entries: S2/S4 each bind both sides; S3/S5 each split long/short for ES and 6E.
The other 33 contracts retain their identical IDs, values and hashes.
The active catalogue totals **45 entries, 32 resolved and 13 unresolved**.

The [active matrix](#b1_outcome_contract_matrix_20260917) includes every
family. Archived [v1](../configs/outcome_contracts/b1_v1.json) and its lock are
byte-identical to the audit snapshot. Explicit `--catalog` in direct CLI mode,
or `catalog` in an input manifest, selects an archive. Replay uses the saved
catalogue and refuses engine-source mismatch; an old run must use its archived
source in an isolated checkout. The new engine identifier is `b1-v2`; artifact
schema remains `b1-outcome-artifacts-v1` because columns are compatible.

Routing now requires side for split contracts, including saved catalogues
containing only one side. Direct evaluation rejects a mismatched side too.
Pooled continuation/reversal, meta and specialist bindings all obey the same
setup-side mapping. IDs, versions and hashes accompany all four surfaces.

Older v1 unresolved diagnostics remain reproducible. Historical close-to-close
120-bar reports, training TP labels, and the frozen August paper-signal trial
are not interchangeable with v2 outcomes. No historical report, model,
calibration, target, label file, roster, policy or trial activation is rewritten.

### Validation and remaining work

Regression coverage pins every instrument/setup/side for pooled, meta and
specialist dispatch; geometry and entry-relative timeout; opposite-side and
missing-side rejection; archived contract preservation; all four artifact
namespaces and deterministic replay; and differing paper targets with identical
forward returns. Existing FRVP pipeline/labeler tests verify compatibility.
Commands and final results are in the [live change journal](live_app_change_journal.md).

OTE independent evaluation rules, classic-breaker economics/scope and unmatched
pooled-signal treatment remain open. Scored collection, B2 artifact lineage,
B3 full policy and the existing promotion gates are not completed by this FRVP
risk decision. `notes.txt` keeps B1 unchecked and checks off this bounded subtask.


<a id="b1_baseline_scope_and_model_setup_audit_20260917"></a>

## B1 baseline scope, model/setup diagnostics and structural ATR

*Merged 2026-10-09 from `docs/B1_baseline_scope_and_model_setup_audit_20260917.md`.*

**Completion addendum, September 18:** B1 is now complete for its declared scope.
[Independent OTE rules](#b1_ote_evaluation_resolution_20260918) and catalogue v3
resolve the five OTE entries; 37 risk entries are defined and eight placeholders
are explicitly excluded. The user's choice preserves 1h reversal and 30m
continuation/breakout structural ATR. All pooled predictions remain diagnostic-only.
[Current matrix](#b1_outcome_contract_matrix_complete_20260918) and
[completion evidence](audits/b1_completion_20260918.json) supersede earlier open
OTE status below. Earlier audit sections remain a historical record; B2/B3 gates
remain open and no scored collection is activated.

User decision September 17; implementation and verification completed
September 18, 2026. This supersedes the earlier open classic-breaker and
unmatched-prediction scope decisions. B1's independent OTE economics remains open.

### Frozen baseline membership

`configs/outcome_contracts/baseline_scope_v1.json` and its SHA-256 lock are the
explicit baseline policy. Membership is versioned separately from setup risk
because the same setup contract can bind a specialist and a pooled model.
The existing v1/v2 economics catalogues and locks remain byte-identical.

| Input | Baseline treatment | Retained evidence |
|---|---|---|
| Classic breaker, either instrument/side | Excluded | Existing detector/research artifacts/logging, causal signal evidence, forward responses where available |
| Every FRVP or ICT pooled reversal/continuation/meta prediction | Diagnostics-only, even with a selected same-side setup and a passing policy | Raw/calibrated score, thresholds/rejections, all observed setup relationships, context and forward responses |
| OTE pooled `ote`, `meta`, `union` predictions | Diagnostics-only | Same diagnostic surfaces; no unresolved OTE geometry is invented |
| Unmatched predictions, including native 1m | Diagnostics-only | Explicit no-match or unknown-coverage state and causal forward responses |
| Setup specialists / named OTE setup families | Their own economics contract, subject to existing prerequisite gates | Four separated outcome surfaces; unresolved OTE risk still fails closed |

This is an explicit **all-pooled-models** decision, not just an unmatched-signal
filter. A selected setup does not promote a pooled prediction into a paper trade.
FRVP specialists such as `continuation_setup3` are distinct from pooled
`continuation`; ICT specialists such as `reversal_sweep_reclaim` are distinct
from pooled `reversal`. OTE named reversal/continuation/breakout targets retain
their unresolved family contracts; `ote/meta/union` are the pooled bindings.
New unregistered family spellings still fail contract dispatch.

For causally valid diagnostic rows, `paper_contract_outcome=NOT_EVALUATED`,
`baseline_scope=diagnostics_only`, and `baseline_scope_reason` identifies the
decision. Stop, target and paper exit remain null. The gross response entry
timestamp/price is only a common measurement origin; it is not a position fill.
Unverified clocks or missing entry bars retain INVALID/NO_ENTRY/DATA_GAP rather
than manufacturing a valid diagnostic. These rows do not enter paper win rates.
Explicitly excluded rows are counted separately from unresolved included paper
outcomes; data-quality failures remain visible in outcome counts.

The position simulator checks scope independently of both `policy_eligible`
and imported paper-result fields. It cannot admit a pooled or classic-breaker
position even when those fields claim an otherwise favorable trade. No label,
threshold, detector, saved research artifact or frozen trial contract is changed.
All current S4, ICT/B2, promotion and activation restrictions remain.

The catalogue still contains 45 frozen risk entries: 32 have risk geometry;
13 deliberately retain UNRESOLVED economics. Of those 13, eight are classic
breaker/unmatched entries now explicitly outside the baseline; five are OTE
risk contracts still requiring a decision. Membership also excludes pooled
bindings of otherwise resolved contracts. See the [current matrix](#b1_outcome_contract_matrix_20260918).

### Data for a later model-versus-setup audit

The live application already persists every ES shadow prediction, including
below-threshold/rejected observations, and immutable links to selected,
non-selected and opposite-side setup candidates within the collected strategy.
It preserves revisions instead of replacing what was known at prediction time.
Research detector candidates, including any emitted classic-breaker research
events, continue through this existing observation path. No detector is removed
or newly activated by this task.

Live shadow evaluation now also saves `model_setup_audit_context` with source
time, prediction recording time, source-bar version, collection identity and
trend/volatility/composite/session/stress regimes **even when those particular
policy filters are disabled**. Missing fields stay unknown. Raw/calibrated
scores, threshold provenance, all policy checks and rejection reasons already
persist separately. The policy file's source hash participates in collection
identity, so subsequent collections distinguish the new context surface.
Historical context is not retroactively guessed.

For immutable read-only exports:

```powershell
ote_venv/Scripts/python.exe scripts/export_model_setup_audit.py --database <saved-live.sqlite> --collection <exact-collection-id> --output-dir <new-directory>
```

The exporter opens SQLite in read-only mode, snapshots a single transaction,
and never initializes/migrates the source database. It retains all predictions,
feature snapshots, runtime manifests, collection manifests, decisions, setup
events/revisions and prediction links as JSONL. A CSV flattens model/setup
relationships without dropping low scores, opposite sides or non-selected
setups. No linked setup is reported as `no_persisted_setup_link`, **not** proof
that no setup existed. Recorded setup/prediction clocks retain known/unknown/
late availability. Seven days of preceding raw price snapshots and up to 60
observed follow-up bars after the final source bar are exported per instrument,
including across weekends; ingestion gap records are preserved too.

Observed rows alone do not prove a complete expected schedule or full model
roster. The exported collection manifest supplies the declared roster when
available. Latest database OHLC snapshots are not proof of historical causal
availability, roll coordinates or complete bars. Those must be established
before importing them into B1; absent forward history remains censored. No
automatic conversion of old live scores into executable signals is performed.

For B1 evaluation input, optional `SignalEvidence.audit_context` carries the
collection, opportunity and source prediction IDs; declared expected models;
pre-filter coverage; every available setup candidate with direction, selected
state, known-at and detector version; model/calibration/policy identities;
scores, thresholds, rejections and regimes. It has an explicit availability
timestamp. A missing context produces an unknown-coverage record rather than
being dropped or called a no-setup opportunity.

Each new B1 run writes:

| Artifact | Purpose |
|---|---|
| `model_setup_observations.csv` | Each supplied model observation crossed with its recorded setup candidates; unmatched/unknown records retained; model-side gross responses and provenance attached |
| `model_observation_coverage.csv` | Expected versus actually supplied model observations at each collection/opportunity; absent scores are not invented |
| `model_setup_response_summary.csv` | Fixed descriptive slices by setup relationship, session, NY decision hour, trend and volatility; valid/missing horizon counts, positive-response counts, mean/median tick/pip movement, and ATR-response coverage |

Collections, model versions/hashes, contract/scope hashes and sides remain
separate. Different collections cannot be evaluated as one baseline. Setup
observations becoming known after the prediction are tagged and excluded from
response aggregates. Multiple candidates of the same family do not multiply
one prediction's contribution to the same slice. Across different setup
families observations can overlap: their counts cannot be added as independent
trades. Opposite-side rows keep the **model's** signed response, with the setup
side stored explicitly. They are never silently relabeled favorable for the
opposite direction.

These outputs make later comparisons possible; they do not pick a winner now.
A defensible audit must compare models on common opportunities and equivalent
decision/entry clocks, include setup-only controls, report sample and missingness
counts and time stability, and account for searching many slices. A missing
score or a positive forward return is not a losing trade or a profitable trade.
No threshold, hour filter or contract is optimized from these summaries.

### ATR audit: timeframe, units and a historical timing defect

**FRVP risk intends structural 1-hour ATR, measured in price units, mapped onto
5-minute decisions.** `FRVPLabelingParams` inherits `structural_atr_tf="1hr"`,
`atr_period=14`, and `atr_smoothing="ema"` from `ReversalParams`. Here `ema`
means Wilder smoothing: alpha=1/14, `adjust=False`, 14-hour warmup; it is not
the usual alpha=2/(14+1) EMA. `_detect_htf_swings` computes this on hourly OHLC.
Multipliers in the frozen FRVP contracts refer to that hourly ATR, not 5-minute
ATR, ticks, percentages or pips. For ES, ATR 10 means 10 index points = 40 ticks;
for 6E, ATR .00100 means 20 ticks at .00005. Tick conversion occurs afterwards.

However, the old FRVP `_resample_ohlcv` uses pandas' default hour-open labels,
and `map_htf_atr_to_ltf` immediately forward-maps those labels. A 15:55 range
spike can therefore change the ATR assigned to 15:00, before the hour completes.
The helper's "causal" docstring does not correct the caller's timestamp
semantics. A synthetic regression reproduces this exact defect without
regenerating labels. This corrects the initial B1 audit, which caught the
entry-row ATR fallback but did not yet expose this earlier publication issue.

The new evaluation producer `model_testing/outcome_atr.py`:

- Aggregates twelve exact completed 5-minute constituents per full hour
  (or sixty 1-minute constituents for a native 1m tape).
- Publishes at the **hour close**, incorporating recorded late arrival times
  when supplied; maps the most recent available completed value to known-at.
  Thus 15:05 may use the hour closed at 15:00, never the hour closing at 16:00.
- Computes price-unit true range and Wilder ATR14; requires 14 valid full
  hours. Incomplete/corrupt/provisional/mixed-coordinate hours reset warmup;
  coordinate changes reset state. Fully scheduled closures create no fake OHLC.
- Saves value, units, timeframe, period, smoothing, source close, history start,
  actual availability, coordinate and a hash of contributing raw constituents.
  Future bars cannot change an earlier value or its source hash.

The engine requires this structural-ATR provenance for FRVP paper geometry.
Wrong units/period/timeframe fail schema validation; missing/mismatched/future
provenance blocks paper evaluation and leaves ATR-normalized markouts null.
Gross raw/tick/percentage diagnostics remain available when causal entry/path
evidence is valid. No 5-minute ATR or entry-bar ATR fallback is substituted.
Supplied provenance is an explicit producer attestation, not a claim that the
engine has independently reconstructed its source bytes. Prefer derivation
from the saved raw history and independently sourced calendar:

```powershell
ote_venv/Scripts/python.exe scripts/run_outcome_evaluation.py --signals <signals.jsonl> --bars <bars-with-history.csv> --schedule <schedule-with-history.csv> --calendar <calendar.json> --derive-structural-atr --output-dir <new-directory>
```

Input manifests accept `"structural_atr_mode": "derive_from_tape"`; otherwise
the explicit `supplied_evidence` mode requires `atr_provenance` for structural
ATR use. Replay saves and repeats the selected mode. Derivation changes only
the evaluation copy of ATR evidence, never caller objects or training labels.

**This is not a claim that every ATR in the repo is 1-hour ATR.** ICT's existing
risk path `_resolve_atr_like` takes signal-bar `atr_14` (or computes it from the
input bar timeframe), and its frozen B1 contract says `signal_bar_atr14`.
FRVP setup-feature distances also use existing feature ATRs; those are not the
paper risk denominator. Replacing those with hourly ATR would change strategy
semantics and requires a separate versioned decision. OTE economics remains
unresolved. No such strategy redesign was made here.

### Compatibility and remaining work

New runs use engine `b1-v3`, artifact schema `b1-outcome-artifacts-v2`, the
unchanged risk catalogue and separately pinned `b1_baseline_scope_v1`.
Scope IDs/hashes/status travel in all four surfaces and manifests. Source
snapshots now include the scope, ATR and audit modules. Old runs must use their
archived engine; replay refuses changed source. The older synthetic pooled
example now illustrates diagnostics-only under this new scope, rather than
its original three-paper-target example. Its inputs remain unmodified.

Historical FRVP labels, saved event files, trained models and detector source
are preserved. B2 must trace the confirmed hourly-publication defect into
historical labels/features and deployed artifacts, then decide which corrected
datasets/labels/models are required. This task neither certifies those models
nor performs that corrective training. S4 stays paused; frozen paper trials,
thresholds, policies, scored collection and broker state are not activated.

Classic-breaker baseline exclusion and the all-pooled diagnostic decision are
complete. B1's remaining economics decision is OTE. Validation and any local
limitations are recorded in the [live change journal](live_app_change_journal.md).


<a id="b1_ote_evaluation_resolution_20260918"></a>

## B1 completion: independent OTE evaluation

*Merged 2026-10-09 from `docs/B1_OTE_evaluation_resolution_20260918.md`.*

Decision and implementation: September 18, 2026. Active economics catalogue:
[`b1_v3.json`](../configs/outcome_contracts/b1_v3.json), with its SHA-256 lock.
Engine `b1-v4`; artifact schema `b1-outcome-artifacts-v2`; membership remains
`b1_baseline_scope_v1`. The user explicitly chose **1-hour reversal ATR and
30-minute continuation/breakout ATR**, preserving the framework's family distinction.

B1 is complete for its declared baseline: 45 contract entries, 37 defined risk
entries, eight deliberately excluded classic-breaker/unmatched placeholders,
and **zero unresolved included bindings**. All 120 registry records / 28 target
names have contract bindings. This completes outcome definitions, not model
lineage verification, prospective policy selection, or permission to activate
scored collection. B2/B3/B4/B5 and existing strategy restrictions remain open.

### Audit before the decision

Inspected `docs/papers/EURUSD_OTE_Framework.docx` sections 3.2–3.6 and execution
limitations, `docs/FRVP_ICT_reference_and_audits.md#frvp_ict_ote_trading_logic`, the OTE model-promotion and
native-1m plans, the three OTE label engines, `reversal_label_deep_analysis.py`,
`model_testing/ote_breakout_event_metadata.py`, shared Phase 6/7 evaluators,
registered targets, and saved sustained-breakout training configuration.

| Evidence | Previous behavior / ambiguity | B1 decision |
|---|---|---|
| Framework §3.2 and `ReversalParams` | Reversal Wilder ATR14 on 1h; continuation and breakout subclasses specify 30m | Preserve these different structural clocks; units are raw price, not pips, ticks or local execution-bar ATR |
| Framework §3.4; family parameter classes | Training triple-barrier distances/durations: reversal stop/target 2/1 ATR, 120 five-minute bars; continuation 0.7/0.8, 36; breakout 0.7/0.9, 24 | Explicitly adopt these pre-existing family distances/durations as the initial independent evaluation baseline. This is a new evaluation decision, not evidence of previously specified live execution |
| `reversal_label_deep_analysis.BacktestConfig` and `run_label_backtest` | Separate analysis defaults 2/1 ATR, 120 bars; buys/sells a retrospectively chosen entry-label close, applies overlapping-position limits and sizing | Retain reversal's existing geometry only. Reject label-derived decision/entry, inherited position filtering and historical close fill |
| `assign_entry_labels`, trend scan and quality validation | Historical zones/best entries may precede confirmation and depend on future validation | A label remains surface A; it can never provide executable availability for B/C/D |
| Breakout event metadata resolver | Derives entry and exit prices/times from labeled event rows and `tb_bars_held` | Never import those execution times, realized holding periods or prices into B1 |
| Sustained-breakout research registry and saved summary | An opt-in refined forecast target; generic `label_max_holding_bars=120` is training metadata, not a specified sustained position horizon | Evaluate the same causal breakout opportunity with the ordinary breakout geometry and 24-bar timeout; keep a separate contract ID. No future persistence gate or label exit |
| Native-1m reversal and `retune_bar_count_params` | Bar counts scale from a five-minute reference | Preserve the reversal duration: 600 native 1m tradable bars = 600 nominal minutes. This is not retrospective entry refinement |
| Shared cost evaluator | EURUSD session spreads plus explicit round-trip slippage and commission; cash lot conversion absent | Preserve `eurusd_session_cost_v1` and pip accounting; no synthetic OHLC adjustments or invented cash conversion |

The geometries deliberately match the existing pre-evaluation family parameters.
They were adopted explicitly to preserve strategy semantics without inventing
or optimizing new distances. **Equal numerical parameters do not make training
labels and paper outcomes equivalent.** The entry, ATR publication clock,
starting point of the timeout, eligibility, collision/gap treatment and outcome
categories are independently specified here. The evaluator does not import a
label engine, training summary, triple-barrier result or realized label horizon.
Changing any historical label parameter cannot change the locked evaluation
config. No evaluation-period P&L was used to select these rules.

### Frozen OTE rules

Both sides mirror around the unadjusted actual entry open. `A` is the latest
fully completed and available structural Wilder ATR14 at `known_at_ts`, in price
units. It is frozen through the position, even if a later ATR appears before a
delayed scheduled entry. There is no local-ATR fallback, dynamic target,
structural target replacement, 1.25R floor, trailing, breakeven or partial exit.

| Setup/model family | Structural ATR | Stop distance | Target distance | Timeout from actual entry | Contract ID |
|---|---|---|---|---|---|
| Reversal 5m | Completed 1h | 2 A | 1 A | 120 bars / 600 nominal minutes | `eurusd_ote_reversal_eval_v2` |
| Reversal native 1m | Completed 1h | 2 A | 1 A | 600 bars / 600 nominal minutes | `eurusd_ote_reversal_1m_eval_v2` |
| Continuation/pullback 5m | Completed 30m | 0.7 A | 0.8 A | 36 bars / 180 nominal minutes | `eurusd_ote_continuation_eval_v2` |
| Breakout 5m | Completed 30m | 0.7 A | 0.9 A | 24 bars / 120 nominal minutes | `eurusd_ote_breakout_eval_v2` |
| Sustained-breakout forecast 5m | Completed 30m | 0.7 A | 0.9 A | 24 bars / 120 nominal minutes | `eurusd_ote_breakout_sustained_eval_v2` |

Reversal's target/stop ratio is 0.5; B1 does not silently change it to an assumed
universal 1R or 2R. Its target-hit rate alone cannot establish expectancy.
Continuation remains covered as a research family even though it is absent
from the current eight-model mission. Sustained is a forecast model, not a
second management strategy. OTE `ote/meta/union` remain **diagnostics-only even
when matched**. These economic definitions do not override that exclusion.

#### Availability and entry

Keep `event_anchor_ts` separate from the actual decision clock. For a replayed
reversal rule, the anchor can be the earlier swing extreme, but availability
requires its later zigzag retracement confirmation close. Continuation requires
confirmed trend-aligned pullback resumption and its completed trend context.
Breakout requires the completed compression/range-break signal using shifted
prior ranges; a retest variant must wait for the actual retest. No future label
quality, trend-scan pass or sustained-target success is an availability input.

For an online forecast, use actual feature/regime/prediction/calibration/policy
completion, along with every setup input it really uses. A forecast does not
wait for its eventual training-label outcome. A claimed retrospective setup
match must have been knowable then; later matches cannot manufacture entries.
`SignalEvidence` requires the source and availability clocks and evidence refs;
`signal_from_available_inputs` takes their maximum. Supplied attestations are
not independently reconstructed feature proofs; raw historical score CSVs
without that evidence remain ineligible. B2 owns feature/artifact verification.

Entry is the first independently scheduled open **strictly after** known-at.
With open-stamped 5m bars, a decision at 10:30:00 first fills 10:35:00, not the
10:25 bar's close or the simultaneous 10:30 open. A native-1m decision at 10:30
first fills 10:31. No best-price search inside a completed signal bar exists.

#### ATR production and units

`completed_structural_atr` accepts explicit 60/30-minute periods. The 1h producer
identity is preserved; 30m has `b1_structural_30m_wilder14_v1` and a separate
validated evidence type. Both use Wilder alpha 1/14, adjust=False, seeded with
the first valid true range and at least 14 complete structural periods. They
publish at structural close or the latest recorded source arrival, whichever
is later. Six exact 5m constituents are required for 30m; twelve for 1h; sixty
native-1m bars for a 1h reversal input. Warmup, partial periods, corrupt,
provisional or missing constituents invalidate the value and reset warmup.
Planned fully closed periods never fabricate OHLC.

EURUSD spot tape may omit futures coordinate IDs. If a spot series ID is
supplied, it must be consistent; mixed series invalidate a structural period.
Futures still require actual contract coordinates and retain roll invalidation.
Each proof records definition, producer, source close, availability, history
start, period, smoothing, price units, coordinate and contributing-data hash.
The CLI's `--derive-structural-atr` (manifest `structural_atr_mode=derive_from_tape`)
derives both periods independently in mixed-family OTE runs. Supplied evidence
is validated but remains a caller attestation; derivation is preferred.

For example, ATR 0.0020 is 20 EURUSD pips. Reversal stop/target distances are
0.0040/0.0020 (40/20 pips). A 30m continuation ATR of 0.0010 gives 7/8-pip
distances. C's ATR-normalized return uses that same frozen family ATR. Gross
price/pip/percentage/log diagnostics remain available if normalization lacks
valid ATR evidence; unresolved paper geometry cannot become a trade.

#### Resolution, costs and the four surfaces

All valid signals receive independent fixed paper outcomes, without portfolio
or sizing filters. Complete causal 1m paths can resolve a 5m collision;
otherwise stop-first. Native 1m collisions also resolve stop-first. Stop gaps
exit at the adverse open, favorable target gaps receive the target price, and
timeout exits use the timeout bar close. Timeouts count tradable bars including
entry; profit/loss/flat remain distinct from target/stop. Planned closures allow
holding; a weekend extends wall-clock duration, not the tradable-bar budget.
Missing, corrupt or explicitly provisional expected entry/holding bars produce
DATA_GAP with no forward fill or substitute entry. Calendar exhaustion before
entry produces NO_ENTRY.

Forward 1/3/6/12/24/48-bar measurements ignore stop/target management, remain
gross and side-adjusted, and are not P&L. Native-1m horizons are native bars,
not disguised five-minute horizons. Pooled observations retain these and the
model/setup relationship, context and coverage audit introduced earlier.

Only explicitly requested position simulation computes P&L. The existing
EURUSD cost version charges the full configured spread **at each leg**:
overlap 1.0, London 1.5, New York 1.5, Asia 2.5, off-hours 3.0 pips. Sessions use
their existing local London/NY clock with DST. Add 0.30-pip round-trip slippage
and 0.35-pip round-trip commission, multiplied by quantity. These are precisely
the code's interpretations, not a broker fee claim. No volatility adjustment
or candle-range spread is introduced. `gross_pnl - spread_cost - slippage_cost
- commission_cost = net_pnl`. Cash conversion and swap/financing are not in
this version; reports must identify its pip accounting and cost scope.

### Migration, completion and verification

Five unresolved OTE v1 contracts are replaced by five v2 contracts in catalogue
v3. The other 40 contracts, costs and scope are identical; archived catalogues
v1/v2 and locks remain byte-identical. Legacy training labels, model artifacts,
detectors, features and old numerical reports are untouched. Earlier label-close,
generic-120-bar or realized-breakout-label-exit backtests are **not comparable**
to these next-open paper/position results; do not splice their equity curves
or rank them as a common experiment. Replay old runs with their saved engine,
catalogue, scope, tape and calendar. New source changes deliberately require
the archived engine to reproduce old run hashes.

Machine completion gate:

```powershell
ote_venv/Scripts/python.exe scripts/audit_outcome_contracts.py --require-complete
```

It checks every catalogue/model-family binding, including research families,
and every registered target. An unresolved included binding or missing registry
binding fails; explicitly excluded risk placeholders do not. Running against
archived v2 correctly fails on five OTE bindings. This gate does not claim that
120 historical registrations have verified features, label hashes or readiness.

Evidence: [completion audit](audits/b1_completion_20260918.json),
[source/decision evidence](audits/b1_ote_resolution_20260918.json),
[updated registry inventory](audits/b1_registry_inventory_complete_20260918.json),
and [full contract matrix](#b1_outcome_contract_matrix_complete_20260918).
Regression tests cover OTE long/short barriers, actual-entry timeouts, native
1m scaling, ATR units/causal prefix/warmup/delays/gaps, future-input rejection,
training/sizing independence, forward-return independence, separate gross/net
results, frozen versions, excluded pooled rows, completion failure and replay.
[Final validation](#b1_completion_validation_20260918): 398 tests passed;
the shared Phase 7 OTE example and direct replay have identical manifests/hashes.

No B1 economics decision remains open inside the declared baseline. Future
classic-breaker economics, pooled-model trade admission, alternative management
and execution refinements require new explicit contracts/scope versions. B2
must trace OTE HTF source timestamps as well as the already demonstrated FRVP
hour-open publication issue: shared forward-fill mapping is causal only when
its HTF inputs are stamped at actual completion. B1 does not rewrite those
historical sources or certify them from their names.


<a id="b1_outcome_contract_matrix_complete_20260918"></a>

## B1 completed contract matrix - 2026-09-18

*Merged 2026-10-09 from `docs/B1_outcome_contract_matrix_complete_20260918.md`.*

Active catalogue: `b1_v3.json`; engine `b1-v4`; scope `b1_baseline_scope_v1`.
45 entries: 37 defined risk and eight explicitly excluded risk placeholders;
zero unresolved included bindings. [OTE decision/audit](#b1_ote_evaluation_resolution_20260918).
All pooled models remain diagnostic-only even when a setup matches. The baseline
includes named setup families/specialists subject to B2/B3 and existing gates;
no model promotion or scored-collection activation follows from this table.

Known-at: **F** firing close plus profile/HTF and actual feature/regime/prediction/calibration/policy availability; **F2** retest/hold after prior breakout; **F3** confirmed outside-value displacement/volume expansion; **F4** quiet reentry after prior outside run; **F5** LVN displacement; **I** causal setup confirmation and actual pipeline availability; **B** OB formation < activation < retest; **OR** reversal confirmation for rule replay or actual forecast decision; **OC** confirmed trend-aligned pullback resumption; **OB** completed range-break signal; **OS** OB forecast, without future persistence/label validation. All require their real source availability; event anchors remain separate.

**N** = first independently scheduled unadjusted open strictly after known-at.
**G** = no forward fill or replacement entry; missing expected entry/holding bar
or invalid roll => DATA_GAP; exhausted entry calendar => NO_ENTRY; independently
scheduled closures permit holding. Timeouts count tradable bars including entry.
**A1h/A30m** = completed structural Wilder ATR14 in price units, frozen at decision;
**Ai** = ICT signal-bar ATR14. **S** = farther of supplied causal stop, 0.25 Ai and
one tick; **T2/T2.5** = farther of supplied target, 2/2.5 Ai and 1.25R. Futures
barriers round outward. Costs apply only in separate position simulation.

| instrument | strategy | setup/model family | known_at | earliest entry | stop | target | timeout | paper management | cost model | missing-bar rule | contract_id |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ES | frvp 5m | setup1 (long_or_short); included: reversal_setup1; diagnostic: reversal, meta | F | N | 1 A1h | 0.8 A1h | 60 bars / 300m | fixed barriers | es_session_cost_v1 | G | `es_frvp_setup1_eval_v1` |
| ES | frvp 5m | setup2 (long_or_short); included: continuation_setup2; diagnostic: continuation, meta | F2 | N | 1 A1h | 1 A1h | 96 bars / 480m | fixed barriers | es_session_cost_v1 | G | `es_frvp_setup2_eval_v2` |
| ES | frvp 5m | setup3 (long); included: continuation_setup3; diagnostic: continuation, meta | F3 | N | 1 A1h | 0.95 A1h | 96 bars / 480m | fixed barriers | es_session_cost_v1 | G | `es_frvp_setup3_long_eval_v2` |
| ES | frvp 5m | setup3 (short); included: continuation_setup3; diagnostic: continuation, meta | F3 | N | 1 A1h | 1 A1h | 96 bars / 480m | fixed barriers | es_session_cost_v1 | G | `es_frvp_setup3_short_eval_v2` |
| ES | frvp 5m | setup4 (long_or_short); included: reversal_setup4; diagnostic: reversal, meta | F4 | N | 0.9 A1h | 1 A1h | 48 bars / 240m | fixed barriers | es_session_cost_v1 | G | `es_frvp_setup4_eval_v2` |
| ES | frvp 5m | setup5 (long); included: continuation_setup5; diagnostic: continuation, meta | F5 | N | 1 A1h | 0.9 A1h | 96 bars / 480m | fixed barriers | es_session_cost_v1 | G | `es_frvp_setup5_long_eval_v2` |
| ES | frvp 5m | setup5 (short); included: continuation_setup5; diagnostic: continuation, meta | F5 | N | 1 A1h | 0.85 A1h | 96 bars / 480m | fixed barriers | es_session_cost_v1 | G | `es_frvp_setup5_short_eval_v2` |
| ES | frvp 5m | setup6 (long_or_short); included: reversal_setup6; diagnostic: reversal, meta | F | N | 1 A1h | 0.8 A1h | 60 bars / 300m | fixed barriers | es_session_cost_v1 | G | `es_frvp_setup6_eval_v1` |
| ES | frvp 5m | unmatched (long_or_short); included: none; diagnostic: reversal, continuation, meta | F | N | excluded; no paper trade | excluded; no paper trade | excluded; no paper trade | excluded; no paper trade | es_session_cost_v1 | G | `es_frvp_unmatched_eval_v1` |
| 6E | frvp 5m | setup1 (long_or_short); included: reversal_setup1; diagnostic: reversal, meta | F | N | 1 A1h | 0.8 A1h | 60 bars / 300m | fixed barriers | 6e_session_cost_v1 | G | `6e_frvp_setup1_eval_v1` |
| 6E | frvp 5m | setup2 (long_or_short); included: continuation_setup2; diagnostic: continuation, meta | F2 | N | 1 A1h | 1 A1h | 96 bars / 480m | fixed barriers | 6e_session_cost_v1 | G | `6e_frvp_setup2_eval_v2` |
| 6E | frvp 5m | setup3 (long); included: continuation_setup3; diagnostic: continuation, meta | F3 | N | 1 A1h | 0.95 A1h | 96 bars / 480m | fixed barriers | 6e_session_cost_v1 | G | `6e_frvp_setup3_long_eval_v2` |
| 6E | frvp 5m | setup3 (short); included: continuation_setup3; diagnostic: continuation, meta | F3 | N | 1 A1h | 1 A1h | 96 bars / 480m | fixed barriers | 6e_session_cost_v1 | G | `6e_frvp_setup3_short_eval_v2` |
| 6E | frvp 5m | setup4 (long_or_short); included: reversal_setup4; diagnostic: reversal, meta | F4 | N | 0.9 A1h | 1 A1h | 48 bars / 240m | fixed barriers | 6e_session_cost_v1 | G | `6e_frvp_setup4_eval_v2` |
| 6E | frvp 5m | setup5 (long); included: continuation_setup5; diagnostic: continuation, meta | F5 | N | 1 A1h | 0.9 A1h | 96 bars / 480m | fixed barriers | 6e_session_cost_v1 | G | `6e_frvp_setup5_long_eval_v2` |
| 6E | frvp 5m | setup5 (short); included: continuation_setup5; diagnostic: continuation, meta | F5 | N | 1 A1h | 0.85 A1h | 96 bars / 480m | fixed barriers | 6e_session_cost_v1 | G | `6e_frvp_setup5_short_eval_v2` |
| 6E | frvp 5m | setup6 (long_or_short); included: reversal_setup6; diagnostic: reversal, meta | F | N | 1 A1h | 0.8 A1h | 60 bars / 300m | fixed barriers | 6e_session_cost_v1 | G | `6e_frvp_setup6_eval_v1` |
| 6E | frvp 5m | unmatched (long_or_short); included: none; diagnostic: reversal, continuation, meta | F | N | excluded; no paper trade | excluded; no paper trade | excluded; no paper trade | excluded; no paper trade | 6e_session_cost_v1 | G | `6e_frvp_unmatched_eval_v1` |
| ES | ict 5m | sweep_reclaim (long_or_short); included: reversal_sweep_reclaim; diagnostic: reversal, meta | I | N | S | T2 | 18 bars / 90m | fixed barriers | es_session_cost_v1 | G | `es_ict_sweep_reclaim_eval_v1` |
| ES | ict 5m | sweep_displacement_fvg (long_or_short); included: reversal_sweep_displacement_fvg; diagnostic: reversal, meta | I | N | S | T2 | 18 bars / 90m | fixed barriers | es_session_cost_v1 | G | `es_ict_sweep_displacement_fvg_eval_v1` |
| ES | ict 5m | ob_retest_after_mss (long_or_short); included: reversal_ob_retest_after_mss; diagnostic: reversal, meta | I | N | S | T2 | 18 bars / 90m | fixed barriers | es_session_cost_v1 | G | `es_ict_ob_retest_after_mss_eval_v1` |
| ES | ict 5m | ifvg_reversal (long_or_short); included: reversal_ifvg_reversal; diagnostic: reversal, meta | I | N | S | T2 | 18 bars / 90m | fixed barriers | es_session_cost_v1 | G | `es_ict_ifvg_reversal_eval_v1` |
| ES | ict 5m | premium_discount_continuation (long_or_short); included: continuation_premium_discount_continuation; diagnostic: continuation, meta | I | N | S | T2.5 | 24 bars / 120m | fixed barriers | es_session_cost_v1 | G | `es_ict_premium_discount_continuation_eval_v1` |
| ES | ict 5m | session_open_manipulation_pre_ib (long_or_short); included: reversal_session_open_manipulation_pre_ib; diagnostic: reversal, meta | I | N | S | T2 | 12 bars / 60m | fixed barriers | es_session_cost_v1 | G | `es_ict_session_open_manipulation_pre_ib_eval_v1` |
| ES | ict 5m | session_open_manipulation_post_ib (long_or_short); included: reversal_session_open_manipulation_post_ib; diagnostic: reversal, meta | I | N | S | T2 | 12 bars / 60m | fixed barriers | es_session_cost_v1 | G | `es_ict_session_open_manipulation_post_ib_eval_v1` |
| ES | ict 5m | displacement_continuation_after_raid (long_or_short); included: continuation_displacement_continuation_after_raid; diagnostic: continuation, meta | I | N | S | T2.5 | 24 bars / 120m | fixed barriers | es_session_cost_v1 | G | `es_ict_displacement_continuation_after_raid_eval_v1` |
| ES | ict 5m | classic_breaker (long_or_short); included: none; diagnostic: classic_breaker | B | N | excluded; no paper trade | excluded; no paper trade | excluded; no paper trade | excluded; no paper trade | es_session_cost_v1 | G | `es_ict_classic_breaker_eval_v1` |
| ES | ict 5m | unmatched (long_or_short); included: none; diagnostic: reversal, continuation, meta | I | N | excluded; no paper trade | excluded; no paper trade | excluded; no paper trade | excluded; no paper trade | es_session_cost_v1 | G | `es_ict_unmatched_eval_v1` |
| 6E | ict 5m | sweep_reclaim (long_or_short); included: reversal_sweep_reclaim; diagnostic: reversal, meta | I | N | S | T2 | 18 bars / 90m | fixed barriers | 6e_session_cost_v1 | G | `6e_ict_sweep_reclaim_eval_v1` |
| 6E | ict 5m | sweep_displacement_fvg (long_or_short); included: reversal_sweep_displacement_fvg; diagnostic: reversal, meta | I | N | S | T2 | 18 bars / 90m | fixed barriers | 6e_session_cost_v1 | G | `6e_ict_sweep_displacement_fvg_eval_v1` |
| 6E | ict 5m | ob_retest_after_mss (long_or_short); included: reversal_ob_retest_after_mss; diagnostic: reversal, meta | I | N | S | T2 | 18 bars / 90m | fixed barriers | 6e_session_cost_v1 | G | `6e_ict_ob_retest_after_mss_eval_v1` |
| 6E | ict 5m | ifvg_reversal (long_or_short); included: reversal_ifvg_reversal; diagnostic: reversal, meta | I | N | S | T2 | 18 bars / 90m | fixed barriers | 6e_session_cost_v1 | G | `6e_ict_ifvg_reversal_eval_v1` |
| 6E | ict 5m | premium_discount_continuation (long_or_short); included: continuation_premium_discount_continuation; diagnostic: continuation, meta | I | N | S | T2.5 | 24 bars / 120m | fixed barriers | 6e_session_cost_v1 | G | `6e_ict_premium_discount_continuation_eval_v1` |
| 6E | ict 5m | session_open_manipulation_pre_ib (long_or_short); included: reversal_session_open_manipulation_pre_ib; diagnostic: reversal, meta | I | N | S | T2 | 12 bars / 60m | fixed barriers | 6e_session_cost_v1 | G | `6e_ict_session_open_manipulation_pre_ib_eval_v1` |
| 6E | ict 5m | session_open_manipulation_post_ib (long_or_short); included: reversal_session_open_manipulation_post_ib; diagnostic: reversal, meta | I | N | S | T2 | 12 bars / 60m | fixed barriers | 6e_session_cost_v1 | G | `6e_ict_session_open_manipulation_post_ib_eval_v1` |
| 6E | ict 5m | displacement_continuation_after_raid (long_or_short); included: continuation_displacement_continuation_after_raid; diagnostic: continuation, meta | I | N | S | T2.5 | 24 bars / 120m | fixed barriers | 6e_session_cost_v1 | G | `6e_ict_displacement_continuation_after_raid_eval_v1` |
| 6E | ict 5m | classic_breaker (long_or_short); included: none; diagnostic: classic_breaker | B | N | excluded; no paper trade | excluded; no paper trade | excluded; no paper trade | excluded; no paper trade | 6e_session_cost_v1 | G | `6e_ict_classic_breaker_eval_v1` |
| 6E | ict 5m | unmatched (long_or_short); included: none; diagnostic: reversal, continuation, meta | I | N | excluded; no paper trade | excluded; no paper trade | excluded; no paper trade | excluded; no paper trade | 6e_session_cost_v1 | G | `6e_ict_unmatched_eval_v1` |
| EURUSD | ote 5m | reversal (long_or_short); included: reversal; diagnostic: meta, ote, union | OR | N | 2 A1h | 1 A1h | 120 bars / 600m | fixed barriers | eurusd_session_cost_v1 | G | `eurusd_ote_reversal_eval_v2` |
| EURUSD | ote 1m | reversal (long_or_short); included: reversal; diagnostic: meta, ote, union | OR | N | 2 A1h | 1 A1h | 600 bars / 600m | fixed barriers | eurusd_session_cost_v1 | G | `eurusd_ote_reversal_1m_eval_v2` |
| EURUSD | ote 5m | continuation (long_or_short); included: continuation; diagnostic: meta, ote, union | OC | N | 0.7 A30m | 0.8 A30m | 36 bars / 180m | fixed barriers | eurusd_session_cost_v1 | G | `eurusd_ote_continuation_eval_v2` |
| EURUSD | ote 5m | breakout (long_or_short); included: breakout; diagnostic: meta, ote, union | OB | N | 0.7 A30m | 0.9 A30m | 24 bars / 120m | fixed barriers | eurusd_session_cost_v1 | G | `eurusd_ote_breakout_eval_v2` |
| EURUSD | ote 5m | breakout_sustained (long_or_short); included: breakout_sustained; diagnostic: meta, ote, union | OS | N | 0.7 A30m | 0.9 A30m | 24 bars / 120m | fixed barriers | eurusd_session_cost_v1 | G | `eurusd_ote_breakout_sustained_eval_v2` |
| EURUSD | ote 5m | unmatched (long_or_short); included: none; diagnostic: reversal, continuation, meta, ote, union, breakout, breakout_sustained | actual forecast clock | N | excluded; no paper trade | excluded; no paper trade | excluded; no paper trade | excluded; no paper trade | eurusd_session_cost_v1 | G | `eurusd_ote_unmatched_eval_v1` |
| EURUSD | ote 1m | unmatched (long_or_short); included: none; diagnostic: reversal, continuation, meta, ote, union, breakout, breakout_sustained | actual forecast clock | N | excluded; no paper trade | excluded; no paper trade | excluded; no paper trade | excluded; no paper trade | eurusd_session_cost_v1 | G | `eurusd_ote_unmatched_1m_eval_v1` |


<a id="b1_completion_validation_20260918"></a>

## B1 completion validation - 2026-09-18

*Merged 2026-10-09 from `docs/B1_completion_validation_20260918.md`.*

Active catalogue `b1_v3.json`, engine `b1-v4`, artifact schema
`b1-outcome-artifacts-v2`, unchanged `b1_baseline_scope_v1`.

**398 distinct tests passed:**

- 186 B1 contract/scope/ATR/OTE tests, 16.32 seconds. Includes the final
  provisional-entry/holding-bar hardening and native-1m resolution diagnostics.
- 212 live-history, collection, causal-feature, research-execution, audit-replay,
  policy, threshold, abstain, setup-event, shadow-policy, Phase04, FRVP-labeling
  and classic-breaker regressions, 79.12 seconds.

Commands:

```powershell
ote_venv/Scripts/python.exe -m pytest tests/test_outcome_contracts.py tests/test_outcome_scope_audit_atr.py tests/test_outcome_ote_contracts.py -q
ote_venv/Scripts/python.exe -m pytest tests/test_ote_live_setup_history_runtime.py tests/test_es_collection_baseline.py tests/test_es_causal_feature_contract.py tests/test_ote_live_research_execution.py tests/test_ote_live_audit_replay.py tests/test_ote_policy_backtest.py tests/test_ote_threshold_policy.py tests/test_ote_abstain_policy.py tests/test_ote_live_setup_event_history.py tests/test_ote_live_shadow_policy.py tests/test_es_primary_phase04_pipeline.py tests/test_frvp_labeling.py tests/test_ict_classic_breaker_phase3_audit.py -q
ote_venv/Scripts/python.exe scripts/audit_outcome_contracts.py --require-complete
```

Completion gate: 45 entries; 37 defined risk entries; eight excluded
classic-breaker/unmatched placeholders; zero unresolved included bindings;
120 registry records, 28 target names and zero missing bindings. Testing the
same gate against v2 correctly identifies its five unresolved included OTE
bindings. This is contract coverage, not model-lineage/readiness certification.

Synthetic Phase 7 execution and deterministic replay:

```powershell
ote_venv/Scripts/python.exe scripts/run_ote_policy_backtest.py --b1-input-manifest model_testing/examples/outcome_b1/ote_input_manifest.json --output-root tmp/b1_ote_completed_final_run_20260918
ote_venv/Scripts/python.exe scripts/run_outcome_evaluation.py --replay-run tmp/b1_ote_completed_final_run_20260918 --output-dir tmp/b1_ote_completed_final_replay_20260918
```

Both saved manifests are identical. Run hash:
`5fd9728b38e23be305451a0aec33718122a5e94787dc060d42967278e9745757`.
Five synthetic observations: four independent named-family paper targets,
one pooled NOT_EVALUATED diagnostic, one admitted position, three independent
paper targets rejected by the position-concurrency fixture. Both completed
structural ATR clocks are derived from raw synthetic history; training labels,
forward responses, paper exits and costed pip P&L remain separate.

Fixtures are not empirical trading results. No model was trained, labels
regenerated, threshold fitted, collector started, or activation gate cleared.
Archived catalogue v1/v2 and all other active contracts/costs remain unchanged.


<a id="b1_candidate_outcome_review_20260929"></a>

## B1 candidate-specific historical outcome review — 2026-09-29

*Merged 2026-10-09 from `docs/B1_candidate_outcome_review_20260929.md`.*

This review closes the candidate-outcome item for the five remaining proposed
ES branches from the B2 candidate-history repair. It binds the review to
`b1_v3`, `baseline_scope_v1`, and the predeclared B3/B4 comparison
`b3_b4_candidate_review_v1_20260929`.

The corrected S3 specialist (ID 6) is not repeated here because its separate
frozen B1 comparison is already recorded in the S3 full-policy v2 review.

The economic comparison is deliberately separate from model engineering. Saved
artifact replay, score tolerance, AUC, tolerance-based event precision and
training completion establish technical lineage only. None is a profitability
or promotion pass.

### Disposition

| ID | Branch | B1 disposition | Historical outcome evidence | Research-collection treatment |
|---:|---|---|---|---|
| 1 | FRVP short continuation | **Excluded** | Not scored as a promotion candidate | Diagnostic-only; pooled FRVP family under `baseline_scope_v1` |
| 2 | FRVP long continuation | **Excluded** | Not scored as a promotion candidate | Diagnostic-only; pooled FRVP family under `baseline_scope_v1` |
| 11 | ICT long continuation | **Excluded** | Not scored as a promotion candidate | Diagnostic-only; pooled ICT family under `baseline_scope_v1` |
| 7 | FRVP long continuation S5 | **Reviewed, not promotion-cleared** | 490 common opportunities; setup-only 347 trades / **-$8,756.88**; frozen threshold 101 / **-$12,358.13**; full policy insufficient evidence | May proceed to prospective research only after live parity and B5–B7/A12/A13 gates |
| 17 | ICT short premium/discount continuation | **Reviewed, not promotion-cleared** | 185 common opportunities; setup-only 86 closed / **+$3,901.25**; frozen threshold 52 closed / **-$1,678.75**; one unresolved; full policy insufficient evidence | May proceed to prospective research only after live parity and B5–B7/A12/A13 gates |

IDs 1, 2 and 11 are not failures of their classifiers; they are excluded from
the B1 qualified baseline because pooled families are diagnostic-only even when
the detector can provide a matching setup. Their raw scores, setup relations
and future diagnostic observations remain retained.

For ID 7, two replacement test endpoints did not have a contemporaneous S5
detector row and were explicitly excluded from the named-S5 common population;
they were not relabeled as S5. ID 17 used the exact named ICT event match. The
setup-only and frozen-threshold legs use the same event IDs, the same B1 tape,
the same entry/exit/cost rules and the same one-position accounting. The
full-policy leg is `insufficient_evidence` unless an exact replacement-version
policy binding is frozen; it is never represented as zero trades or a pass.

The observed historical result is diagnostic evidence, not a promotion rule:
ID 7 loses after B1 costs in both legs, while ID 17's frozen threshold loses
despite a positive setup-only control. These results do not authorize a
threshold change or a policy search on the inspected sample.

The detailed counts, net/gross P&L, costs, resolution status, run hashes and
replay check are in
[`b1_candidate_outcome_review_20260929.json`](audits/b1_candidate_outcome_review_20260929.json).
The reproducible review runner is
[`review_b1_candidate_outcomes.py`](../scripts/review_b1_candidate_outcomes.py).

### Clean evaluation boundary

The replacement models were fitted without the held-out endpoints used here.
The review evaluates those endpoints once under the frozen B1 contracts; it
does not use labels, future outcomes, threshold tuning or policy search to
admit opportunities. Technical B2 replay inspection remains technical evidence
only.

The clean prospective sample is predeclared rather than waiting for its future
outcomes: the C1 sample begins at the first healthy post-B7 collection
timestamp under a new immutable collection identity, with full eligible ES
session coverage and selected/rejected same-side named setup opportunities
retained. The roster, artifact hashes, threshold/calibration, B1 contract,
matcher, policy binding (or explicit no-policy status), cost stress and review
target must be frozen before that timestamp. Future C1 outcomes are therefore
not a prerequisite for starting an otherwise eligible research collection.

This review does not install a model, alter a registry, activate qualified
collection, start a paper trial, or authorize orders. All existing pooled-model,
S4, retired-short-reversal, economics, drawdown, concentration and promotion
restrictions remain binding.


## Superseded versions (kept verbatim)


<a id="b1_outcome_contract_matrix_20260918"></a>

### B1 active contract matrix - 2026-09-18

*Merged 2026-10-09 from `docs/B1_outcome_contract_matrix_20260918.md`.*

Economics: unchanged `b1_v2.json`. Membership: locked `baseline_scope_v1.json`. [Scope, diagnostics and ATR decision](#b1_baseline_scope_and_model_setup_audit_20260917). All pooled-model bindings are diagnostics-only even when a setup matches. Classic breaker and unmatched predictions are explicitly outside the baseline. Specialist/named setup bindings retain their economics, subject to prerequisite gates.

FRVP uses completed 1h Wilder ATR14 in price units mapped at the decision clock; structural provenance is required. ICT remains signal-bar ATR14. New engine b1-v3 stores scope identity independently of economics. Risk catalogue still has 32 resolved / 13 unresolved entries; eight unresolved entries are now explicitly excluded and five OTE entries remain open.

Known-at: **F** = firing close + completed profile/HTF and actual feature/regime/prediction/calibration/policy availability; **F2** = F at retest/hold after prior breakout; **F3** = F at confirmed outside-value displacement and volume expansion; **F4** = F at confirmed quiet reentry after prior outside run; **F5** = F at LVN displacement. Causal structural ATR must also be available. **I** = confirmed sweep/MSS/FVG/retest close + actual ML/policy availability; **B** = source OB formation < activation < retest/rejection close + actual ML/policy availability; **O** = causal ML/policy availability, plus swing/zigzag or breakout confirmation for rule-event replay. Preserve event anchors separately.
Entry **N** = first independently scheduled unadjusted open strictly after known-at. Missing rule **G** = no forward fill or replacement entry; missing expected entry/holding bar or invalid roll coordinates => DATA_GAP; exhausted entry calendar => NO_ENTRY; planned closures follow the frozen schedule, holding permitted. **S** = structural stop at least max(0.25 ATR, one tick); **T2/T2.5** = farther of supplied opposing target, 2/2.5 ATR and 1.25R. Futures barriers round outward to ticks. Nominal minutes count tradable bars, including entry. Costs apply only in separate position simulation.

| instrument | strategy | setup/model family | known_at | earliest entry | stop | target | timeout | paper management | cost model | missing-bar rule | contract_id |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ES | frvp 5m | setup1 (both sides); included: reversal_setup1; diagnostic: reversal, meta | F | N | 1 ATR | 0.8 ATR | 60 bars / 300m | fixed barriers | es_session_cost_v1 | G | `es_frvp_setup1_eval_v1` |
| ES | frvp 5m | setup2 (both sides); included: continuation_setup2; diagnostic: continuation, meta | F2 | N | 1 ATR | 1 ATR | 96 bars / 480m | fixed barriers | es_session_cost_v1 | G | `es_frvp_setup2_eval_v2` |
| ES | frvp 5m | setup3 (long); included: continuation_setup3; diagnostic: continuation, meta | F3 | N | 1 ATR | 0.95 ATR | 96 bars / 480m | fixed barriers | es_session_cost_v1 | G | `es_frvp_setup3_long_eval_v2` |
| ES | frvp 5m | setup3 (short); included: continuation_setup3; diagnostic: continuation, meta | F3 | N | 1 ATR | 1 ATR | 96 bars / 480m | fixed barriers | es_session_cost_v1 | G | `es_frvp_setup3_short_eval_v2` |
| ES | frvp 5m | setup4 (both sides); included: reversal_setup4; diagnostic: reversal, meta | F4 | N | 0.9 ATR | 1 ATR | 48 bars / 240m | fixed barriers | es_session_cost_v1 | G | `es_frvp_setup4_eval_v2` |
| ES | frvp 5m | setup5 (long); included: continuation_setup5; diagnostic: continuation, meta | F5 | N | 1 ATR | 0.9 ATR | 96 bars / 480m | fixed barriers | es_session_cost_v1 | G | `es_frvp_setup5_long_eval_v2` |
| ES | frvp 5m | setup5 (short); included: continuation_setup5; diagnostic: continuation, meta | F5 | N | 1 ATR | 0.85 ATR | 96 bars / 480m | fixed barriers | es_session_cost_v1 | G | `es_frvp_setup5_short_eval_v2` |
| ES | frvp 5m | setup6 (both sides); included: reversal_setup6; diagnostic: reversal, meta | F | N | 1 ATR | 0.8 ATR | 60 bars / 300m | fixed barriers | es_session_cost_v1 | G | `es_frvp_setup6_eval_v1` |
| ES | frvp 5m | unmatched (both sides); diagnostic: reversal, continuation, meta | F | N | excluded | excluded | excluded | diagnostics only | not applied | G | `es_frvp_unmatched_eval_v1` |
| 6E | frvp 5m | setup1 (both sides); included: reversal_setup1; diagnostic: reversal, meta | F | N | 1 ATR | 0.8 ATR | 60 bars / 300m | fixed barriers | 6e_session_cost_v1 | G | `6e_frvp_setup1_eval_v1` |
| 6E | frvp 5m | setup2 (both sides); included: continuation_setup2; diagnostic: continuation, meta | F2 | N | 1 ATR | 1 ATR | 96 bars / 480m | fixed barriers | 6e_session_cost_v1 | G | `6e_frvp_setup2_eval_v2` |
| 6E | frvp 5m | setup3 (long); included: continuation_setup3; diagnostic: continuation, meta | F3 | N | 1 ATR | 0.95 ATR | 96 bars / 480m | fixed barriers | 6e_session_cost_v1 | G | `6e_frvp_setup3_long_eval_v2` |
| 6E | frvp 5m | setup3 (short); included: continuation_setup3; diagnostic: continuation, meta | F3 | N | 1 ATR | 1 ATR | 96 bars / 480m | fixed barriers | 6e_session_cost_v1 | G | `6e_frvp_setup3_short_eval_v2` |
| 6E | frvp 5m | setup4 (both sides); included: reversal_setup4; diagnostic: reversal, meta | F4 | N | 0.9 ATR | 1 ATR | 48 bars / 240m | fixed barriers | 6e_session_cost_v1 | G | `6e_frvp_setup4_eval_v2` |
| 6E | frvp 5m | setup5 (long); included: continuation_setup5; diagnostic: continuation, meta | F5 | N | 1 ATR | 0.9 ATR | 96 bars / 480m | fixed barriers | 6e_session_cost_v1 | G | `6e_frvp_setup5_long_eval_v2` |
| 6E | frvp 5m | setup5 (short); included: continuation_setup5; diagnostic: continuation, meta | F5 | N | 1 ATR | 0.85 ATR | 96 bars / 480m | fixed barriers | 6e_session_cost_v1 | G | `6e_frvp_setup5_short_eval_v2` |
| 6E | frvp 5m | setup6 (both sides); included: reversal_setup6; diagnostic: reversal, meta | F | N | 1 ATR | 0.8 ATR | 60 bars / 300m | fixed barriers | 6e_session_cost_v1 | G | `6e_frvp_setup6_eval_v1` |
| 6E | frvp 5m | unmatched (both sides); diagnostic: reversal, continuation, meta | F | N | excluded | excluded | excluded | diagnostics only | not applied | G | `6e_frvp_unmatched_eval_v1` |
| ES | ict 5m | sweep_reclaim (both sides); included: reversal_sweep_reclaim; diagnostic: reversal, meta | I | N | S | T2 | 18 bars / 90m | fixed barriers | es_session_cost_v1 | G | `es_ict_sweep_reclaim_eval_v1` |
| ES | ict 5m | sweep_displacement_fvg (both sides); included: reversal_sweep_displacement_fvg; diagnostic: reversal, meta | I | N | S | T2 | 18 bars / 90m | fixed barriers | es_session_cost_v1 | G | `es_ict_sweep_displacement_fvg_eval_v1` |
| ES | ict 5m | ob_retest_after_mss (both sides); included: reversal_ob_retest_after_mss; diagnostic: reversal, meta | I | N | S | T2 | 18 bars / 90m | fixed barriers | es_session_cost_v1 | G | `es_ict_ob_retest_after_mss_eval_v1` |
| ES | ict 5m | ifvg_reversal (both sides); included: reversal_ifvg_reversal; diagnostic: reversal, meta | I | N | S | T2 | 18 bars / 90m | fixed barriers | es_session_cost_v1 | G | `es_ict_ifvg_reversal_eval_v1` |
| ES | ict 5m | premium_discount_continuation (both sides); included: continuation_premium_discount_continuation; diagnostic: continuation, meta | I | N | S | T2.5 | 24 bars / 120m | fixed barriers | es_session_cost_v1 | G | `es_ict_premium_discount_continuation_eval_v1` |
| ES | ict 5m | session_open_manipulation_pre_ib (both sides); included: reversal_session_open_manipulation_pre_ib; diagnostic: reversal, meta | I | N | S | T2 | 12 bars / 60m | fixed barriers | es_session_cost_v1 | G | `es_ict_session_open_manipulation_pre_ib_eval_v1` |
| ES | ict 5m | session_open_manipulation_post_ib (both sides); included: reversal_session_open_manipulation_post_ib; diagnostic: reversal, meta | I | N | S | T2 | 12 bars / 60m | fixed barriers | es_session_cost_v1 | G | `es_ict_session_open_manipulation_post_ib_eval_v1` |
| ES | ict 5m | displacement_continuation_after_raid (both sides); included: continuation_displacement_continuation_after_raid; diagnostic: continuation, meta | I | N | S | T2.5 | 24 bars / 120m | fixed barriers | es_session_cost_v1 | G | `es_ict_displacement_continuation_after_raid_eval_v1` |
| ES | ict 5m | classic_breaker (both sides); diagnostic: classic_breaker | B | N | excluded | excluded | excluded | diagnostics only | not applied | G | `es_ict_classic_breaker_eval_v1` |
| ES | ict 5m | unmatched (both sides); diagnostic: reversal, continuation, meta | I | N | excluded | excluded | excluded | diagnostics only | not applied | G | `es_ict_unmatched_eval_v1` |
| 6E | ict 5m | sweep_reclaim (both sides); included: reversal_sweep_reclaim; diagnostic: reversal, meta | I | N | S | T2 | 18 bars / 90m | fixed barriers | 6e_session_cost_v1 | G | `6e_ict_sweep_reclaim_eval_v1` |
| 6E | ict 5m | sweep_displacement_fvg (both sides); included: reversal_sweep_displacement_fvg; diagnostic: reversal, meta | I | N | S | T2 | 18 bars / 90m | fixed barriers | 6e_session_cost_v1 | G | `6e_ict_sweep_displacement_fvg_eval_v1` |
| 6E | ict 5m | ob_retest_after_mss (both sides); included: reversal_ob_retest_after_mss; diagnostic: reversal, meta | I | N | S | T2 | 18 bars / 90m | fixed barriers | 6e_session_cost_v1 | G | `6e_ict_ob_retest_after_mss_eval_v1` |
| 6E | ict 5m | ifvg_reversal (both sides); included: reversal_ifvg_reversal; diagnostic: reversal, meta | I | N | S | T2 | 18 bars / 90m | fixed barriers | 6e_session_cost_v1 | G | `6e_ict_ifvg_reversal_eval_v1` |
| 6E | ict 5m | premium_discount_continuation (both sides); included: continuation_premium_discount_continuation; diagnostic: continuation, meta | I | N | S | T2.5 | 24 bars / 120m | fixed barriers | 6e_session_cost_v1 | G | `6e_ict_premium_discount_continuation_eval_v1` |
| 6E | ict 5m | session_open_manipulation_pre_ib (both sides); included: reversal_session_open_manipulation_pre_ib; diagnostic: reversal, meta | I | N | S | T2 | 12 bars / 60m | fixed barriers | 6e_session_cost_v1 | G | `6e_ict_session_open_manipulation_pre_ib_eval_v1` |
| 6E | ict 5m | session_open_manipulation_post_ib (both sides); included: reversal_session_open_manipulation_post_ib; diagnostic: reversal, meta | I | N | S | T2 | 12 bars / 60m | fixed barriers | 6e_session_cost_v1 | G | `6e_ict_session_open_manipulation_post_ib_eval_v1` |
| 6E | ict 5m | displacement_continuation_after_raid (both sides); included: continuation_displacement_continuation_after_raid; diagnostic: continuation, meta | I | N | S | T2.5 | 24 bars / 120m | fixed barriers | 6e_session_cost_v1 | G | `6e_ict_displacement_continuation_after_raid_eval_v1` |
| 6E | ict 5m | classic_breaker (both sides); diagnostic: classic_breaker | B | N | excluded | excluded | excluded | diagnostics only | not applied | G | `6e_ict_classic_breaker_eval_v1` |
| 6E | ict 5m | unmatched (both sides); diagnostic: reversal, continuation, meta | I | N | excluded | excluded | excluded | diagnostics only | not applied | G | `6e_ict_unmatched_eval_v1` |
| EURUSD | ote 5m | reversal (both sides); included: reversal; diagnostic: meta, ote, union | O | N | UNRESOLVED | UNRESOLVED | UNRESOLVED | fixed barriers | eurusd_session_cost_v1 | G | `eurusd_ote_reversal_eval_v1` |
| EURUSD | ote 1m | reversal (both sides); included: reversal; diagnostic: meta, ote, union | O | N | UNRESOLVED | UNRESOLVED | UNRESOLVED | fixed barriers | eurusd_session_cost_v1 | G | `eurusd_ote_reversal_1m_eval_v1` |
| EURUSD | ote 5m | continuation (both sides); included: continuation; diagnostic: meta, ote, union | O | N | UNRESOLVED | UNRESOLVED | UNRESOLVED | fixed barriers | eurusd_session_cost_v1 | G | `eurusd_ote_continuation_eval_v1` |
| EURUSD | ote 5m | breakout (both sides); included: breakout; diagnostic: meta, ote, union | O | N | UNRESOLVED | UNRESOLVED | UNRESOLVED | fixed barriers | eurusd_session_cost_v1 | G | `eurusd_ote_breakout_eval_v1` |
| EURUSD | ote 5m | breakout_sustained (both sides); included: breakout_sustained; diagnostic: meta, ote, union | O | N | UNRESOLVED | UNRESOLVED | UNRESOLVED | fixed barriers | eurusd_session_cost_v1 | G | `eurusd_ote_breakout_sustained_eval_v1` |
| EURUSD | ote 5m | unmatched (both sides); diagnostic: reversal, continuation, meta, ote, union, breakout, breakout_sustained | O | N | excluded | excluded | excluded | diagnostics only | not applied | G | `eurusd_ote_unmatched_eval_v1` |
| EURUSD | ote 1m | unmatched (both sides); diagnostic: reversal, continuation, meta, ote, union, breakout, breakout_sustained | O | N | excluded | excluded | excluded | diagnostics only | not applied | G | `eurusd_ote_unmatched_1m_eval_v1` |

For diagnostic bindings, N is a gross market-response origin only; there is no paper trade, position or cost deduction. Same-side setup matching cannot release a pooled model. The five remaining OTE risk entries still require independent economics. Historical v1/v2 catalogues and earlier matrices are retained.


<a id="b1_outcome_contract_matrix_20260917"></a>

### B1 active contract matrix - 2026-09-17

*Merged 2026-10-09 from `docs/B1_outcome_contract_matrix_20260917.md`.*

Historical membership view: [September 18 matrix](#b1_outcome_contract_matrix_20260918)
applies the user-approved diagnostic/exclusion scope. Numerical risk contracts
below remain frozen; pooled eligibility has been superseded.

Active release: `configs/outcome_contracts/b1_v2.json` and its lock. 45 entries: 32 resolved, 13 unresolved. Archived v1 remains unchanged. [FRVP decision and evidence](#b1_frvp_risk_resolution_20260917).

Pooled and specialist models use the causal selected setup and actual side. S3/S5 have separate long/short contracts; architecture never selects risk settings. 6E is a configured transfer variant, not a claim of validated models. S4 policy candidacy remains paused.

Known-at: **F** = firing close + completed profile/HTF and actual feature/regime/prediction/calibration/policy availability; **F2** = F at retest/hold after prior breakout; **F3** = F at confirmed outside-value displacement and volume expansion; **F4** = F at confirmed quiet reentry after prior outside run; **F5** = F at LVN displacement. Causal structural ATR must also be available. **I** = confirmed sweep/MSS/FVG/retest close + actual ML/policy availability; **B** = source OB formation < activation < retest/rejection close + actual ML/policy availability; **O** = causal ML/policy availability, plus swing/zigzag or breakout confirmation for rule-event replay. Preserve event anchors separately.

Entry **N** = first independently scheduled unadjusted open strictly after known-at. Missing rule **G** = no forward fill or replacement entry; missing expected entry/holding bar or invalid roll coordinates => DATA_GAP; exhausted entry calendar => NO_ENTRY; planned closures follow the frozen schedule, holding permitted. **S** = structural stop at least max(0.25 ATR, one tick); **T2/T2.5** = farther of supplied opposing target, 2/2.5 ATR and 1.25R. Futures barriers round outward to ticks. Nominal minutes count tradable bars, including entry. Costs apply only in separate position simulation.

| instrument | strategy | setup/model family | known_at | earliest entry | stop | target | timeout | paper management | cost model | missing-bar rule | contract_id |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ES | frvp 5m | setup1 (both sides); reversal, meta, reversal_setup1 | F | N | 1 ATR | 0.8 ATR | 60 bars / 300m | fixed barriers | es_session_cost_v1 | G | `es_frvp_setup1_eval_v1` |
| ES | frvp 5m | setup2 (both sides); continuation, meta, continuation_setup2 | F2 | N | 1 ATR | 1 ATR | 96 bars / 480m | fixed barriers | es_session_cost_v1 | G | `es_frvp_setup2_eval_v2` |
| ES | frvp 5m | setup3 (long); continuation, meta, continuation_setup3 | F3 | N | 1 ATR | 0.95 ATR | 96 bars / 480m | fixed barriers | es_session_cost_v1 | G | `es_frvp_setup3_long_eval_v2` |
| ES | frvp 5m | setup3 (short); continuation, meta, continuation_setup3 | F3 | N | 1 ATR | 1 ATR | 96 bars / 480m | fixed barriers | es_session_cost_v1 | G | `es_frvp_setup3_short_eval_v2` |
| ES | frvp 5m | setup4 (both sides); reversal, meta, reversal_setup4 | F4 | N | 0.9 ATR | 1 ATR | 48 bars / 240m | fixed barriers | es_session_cost_v1 | G | `es_frvp_setup4_eval_v2` |
| ES | frvp 5m | setup5 (long); continuation, meta, continuation_setup5 | F5 | N | 1 ATR | 0.9 ATR | 96 bars / 480m | fixed barriers | es_session_cost_v1 | G | `es_frvp_setup5_long_eval_v2` |
| ES | frvp 5m | setup5 (short); continuation, meta, continuation_setup5 | F5 | N | 1 ATR | 0.85 ATR | 96 bars / 480m | fixed barriers | es_session_cost_v1 | G | `es_frvp_setup5_short_eval_v2` |
| ES | frvp 5m | setup6 (both sides); reversal, meta, reversal_setup6 | F | N | 1 ATR | 0.8 ATR | 60 bars / 300m | fixed barriers | es_session_cost_v1 | G | `es_frvp_setup6_eval_v1` |
| ES | frvp 5m | unmatched (both sides); reversal, continuation, meta | F | N | UNRESOLVED | UNRESOLVED | UNRESOLVED | fixed barriers | es_session_cost_v1 | G | `es_frvp_unmatched_eval_v1` |
| 6E | frvp 5m | setup1 (both sides); reversal, meta, reversal_setup1 | F | N | 1 ATR | 0.8 ATR | 60 bars / 300m | fixed barriers | 6e_session_cost_v1 | G | `6e_frvp_setup1_eval_v1` |
| 6E | frvp 5m | setup2 (both sides); continuation, meta, continuation_setup2 | F2 | N | 1 ATR | 1 ATR | 96 bars / 480m | fixed barriers | 6e_session_cost_v1 | G | `6e_frvp_setup2_eval_v2` |
| 6E | frvp 5m | setup3 (long); continuation, meta, continuation_setup3 | F3 | N | 1 ATR | 0.95 ATR | 96 bars / 480m | fixed barriers | 6e_session_cost_v1 | G | `6e_frvp_setup3_long_eval_v2` |
| 6E | frvp 5m | setup3 (short); continuation, meta, continuation_setup3 | F3 | N | 1 ATR | 1 ATR | 96 bars / 480m | fixed barriers | 6e_session_cost_v1 | G | `6e_frvp_setup3_short_eval_v2` |
| 6E | frvp 5m | setup4 (both sides); reversal, meta, reversal_setup4 | F4 | N | 0.9 ATR | 1 ATR | 48 bars / 240m | fixed barriers | 6e_session_cost_v1 | G | `6e_frvp_setup4_eval_v2` |
| 6E | frvp 5m | setup5 (long); continuation, meta, continuation_setup5 | F5 | N | 1 ATR | 0.9 ATR | 96 bars / 480m | fixed barriers | 6e_session_cost_v1 | G | `6e_frvp_setup5_long_eval_v2` |
| 6E | frvp 5m | setup5 (short); continuation, meta, continuation_setup5 | F5 | N | 1 ATR | 0.85 ATR | 96 bars / 480m | fixed barriers | 6e_session_cost_v1 | G | `6e_frvp_setup5_short_eval_v2` |
| 6E | frvp 5m | setup6 (both sides); reversal, meta, reversal_setup6 | F | N | 1 ATR | 0.8 ATR | 60 bars / 300m | fixed barriers | 6e_session_cost_v1 | G | `6e_frvp_setup6_eval_v1` |
| 6E | frvp 5m | unmatched (both sides); reversal, continuation, meta | F | N | UNRESOLVED | UNRESOLVED | UNRESOLVED | fixed barriers | 6e_session_cost_v1 | G | `6e_frvp_unmatched_eval_v1` |
| ES | ict 5m | sweep_reclaim (both sides); reversal, meta, reversal_sweep_reclaim | I | N | S | T2 | 18 bars / 90m | fixed barriers | es_session_cost_v1 | G | `es_ict_sweep_reclaim_eval_v1` |
| ES | ict 5m | sweep_displacement_fvg (both sides); reversal, meta, reversal_sweep_displacement_fvg | I | N | S | T2 | 18 bars / 90m | fixed barriers | es_session_cost_v1 | G | `es_ict_sweep_displacement_fvg_eval_v1` |
| ES | ict 5m | ob_retest_after_mss (both sides); reversal, meta, reversal_ob_retest_after_mss | I | N | S | T2 | 18 bars / 90m | fixed barriers | es_session_cost_v1 | G | `es_ict_ob_retest_after_mss_eval_v1` |
| ES | ict 5m | ifvg_reversal (both sides); reversal, meta, reversal_ifvg_reversal | I | N | S | T2 | 18 bars / 90m | fixed barriers | es_session_cost_v1 | G | `es_ict_ifvg_reversal_eval_v1` |
| ES | ict 5m | premium_discount_continuation (both sides); continuation, meta, continuation_premium_discount_continuation | I | N | S | T2.5 | 24 bars / 120m | fixed barriers | es_session_cost_v1 | G | `es_ict_premium_discount_continuation_eval_v1` |
| ES | ict 5m | session_open_manipulation_pre_ib (both sides); reversal, meta, reversal_session_open_manipulation_pre_ib | I | N | S | T2 | 12 bars / 60m | fixed barriers | es_session_cost_v1 | G | `es_ict_session_open_manipulation_pre_ib_eval_v1` |
| ES | ict 5m | session_open_manipulation_post_ib (both sides); reversal, meta, reversal_session_open_manipulation_post_ib | I | N | S | T2 | 12 bars / 60m | fixed barriers | es_session_cost_v1 | G | `es_ict_session_open_manipulation_post_ib_eval_v1` |
| ES | ict 5m | displacement_continuation_after_raid (both sides); continuation, meta, continuation_displacement_continuation_after_raid | I | N | S | T2.5 | 24 bars / 120m | fixed barriers | es_session_cost_v1 | G | `es_ict_displacement_continuation_after_raid_eval_v1` |
| ES | ict 5m | classic_breaker (both sides); classic_breaker | B | N | UNRESOLVED | UNRESOLVED | UNRESOLVED | fixed barriers | es_session_cost_v1 | G | `es_ict_classic_breaker_eval_v1` |
| ES | ict 5m | unmatched (both sides); reversal, continuation, meta | I | N | UNRESOLVED | UNRESOLVED | UNRESOLVED | fixed barriers | es_session_cost_v1 | G | `es_ict_unmatched_eval_v1` |
| 6E | ict 5m | sweep_reclaim (both sides); reversal, meta, reversal_sweep_reclaim | I | N | S | T2 | 18 bars / 90m | fixed barriers | 6e_session_cost_v1 | G | `6e_ict_sweep_reclaim_eval_v1` |
| 6E | ict 5m | sweep_displacement_fvg (both sides); reversal, meta, reversal_sweep_displacement_fvg | I | N | S | T2 | 18 bars / 90m | fixed barriers | 6e_session_cost_v1 | G | `6e_ict_sweep_displacement_fvg_eval_v1` |
| 6E | ict 5m | ob_retest_after_mss (both sides); reversal, meta, reversal_ob_retest_after_mss | I | N | S | T2 | 18 bars / 90m | fixed barriers | 6e_session_cost_v1 | G | `6e_ict_ob_retest_after_mss_eval_v1` |
| 6E | ict 5m | ifvg_reversal (both sides); reversal, meta, reversal_ifvg_reversal | I | N | S | T2 | 18 bars / 90m | fixed barriers | 6e_session_cost_v1 | G | `6e_ict_ifvg_reversal_eval_v1` |
| 6E | ict 5m | premium_discount_continuation (both sides); continuation, meta, continuation_premium_discount_continuation | I | N | S | T2.5 | 24 bars / 120m | fixed barriers | 6e_session_cost_v1 | G | `6e_ict_premium_discount_continuation_eval_v1` |
| 6E | ict 5m | session_open_manipulation_pre_ib (both sides); reversal, meta, reversal_session_open_manipulation_pre_ib | I | N | S | T2 | 12 bars / 60m | fixed barriers | 6e_session_cost_v1 | G | `6e_ict_session_open_manipulation_pre_ib_eval_v1` |
| 6E | ict 5m | session_open_manipulation_post_ib (both sides); reversal, meta, reversal_session_open_manipulation_post_ib | I | N | S | T2 | 12 bars / 60m | fixed barriers | 6e_session_cost_v1 | G | `6e_ict_session_open_manipulation_post_ib_eval_v1` |
| 6E | ict 5m | displacement_continuation_after_raid (both sides); continuation, meta, continuation_displacement_continuation_after_raid | I | N | S | T2.5 | 24 bars / 120m | fixed barriers | 6e_session_cost_v1 | G | `6e_ict_displacement_continuation_after_raid_eval_v1` |
| 6E | ict 5m | classic_breaker (both sides); classic_breaker | B | N | UNRESOLVED | UNRESOLVED | UNRESOLVED | fixed barriers | 6e_session_cost_v1 | G | `6e_ict_classic_breaker_eval_v1` |
| 6E | ict 5m | unmatched (both sides); reversal, continuation, meta | I | N | UNRESOLVED | UNRESOLVED | UNRESOLVED | fixed barriers | 6e_session_cost_v1 | G | `6e_ict_unmatched_eval_v1` |
| EURUSD | ote 5m | reversal (both sides); reversal, meta, ote, union | O | N | UNRESOLVED | UNRESOLVED | UNRESOLVED | fixed barriers | eurusd_session_cost_v1 | G | `eurusd_ote_reversal_eval_v1` |
| EURUSD | ote 1m | reversal (both sides); reversal, meta, ote, union | O | N | UNRESOLVED | UNRESOLVED | UNRESOLVED | fixed barriers | eurusd_session_cost_v1 | G | `eurusd_ote_reversal_1m_eval_v1` |
| EURUSD | ote 5m | continuation (both sides); continuation, meta, ote, union | O | N | UNRESOLVED | UNRESOLVED | UNRESOLVED | fixed barriers | eurusd_session_cost_v1 | G | `eurusd_ote_continuation_eval_v1` |
| EURUSD | ote 5m | breakout (both sides); breakout, meta, ote, union | O | N | UNRESOLVED | UNRESOLVED | UNRESOLVED | fixed barriers | eurusd_session_cost_v1 | G | `eurusd_ote_breakout_eval_v1` |
| EURUSD | ote 5m | breakout_sustained (both sides); breakout_sustained, meta, ote, union | O | N | UNRESOLVED | UNRESOLVED | UNRESOLVED | fixed barriers | eurusd_session_cost_v1 | G | `eurusd_ote_breakout_sustained_eval_v1` |
| EURUSD | ote 5m | unmatched (both sides); reversal, continuation, meta, ote, union, breakout, breakout_sustained | O | N | UNRESOLVED | UNRESOLVED | UNRESOLVED | fixed barriers | eurusd_session_cost_v1 | G | `eurusd_ote_unmatched_eval_v1` |
| EURUSD | ote 1m | unmatched (both sides); reversal, continuation, meta, ote, union, breakout, breakout_sustained | O | N | UNRESOLVED | UNRESOLVED | UNRESOLVED | fixed barriers | eurusd_session_cost_v1 | G | `eurusd_ote_unmatched_1m_eval_v1` |

UNRESOLVED entries retain causal forward diagnostics where possible, produce no resolved paper result, and cannot be admitted by the B1 position simulator. OTE, classic breaker and unmatched pooled treatment remain open B1 decisions. B2/B3 and activation restrictions continue to apply even to resolved contracts.


<a id="b1_outcome_contract_matrix_20260916"></a>

### B1 contract matrix ? 2026-09-16

*Merged 2026-10-09 from `docs/B1_outcome_contract_matrix_20260916.md`.*

Historical v1 matrix. The [active v2 matrix](#b1_outcome_contract_matrix_20260917)
resolves FRVP S2/S3/S5/S4, including side-specific S3/S5 contracts. The v1
catalogue and lock remain unchanged; rows below describe that archive.

Both sides share each listed v1 contract. XGB/TCN/LSTM/RNN architecture does not change setup economics. Pooled models dispatch using the causally matched setup; per-setup targets are bound in the JSON catalogue. UNRESOLVED rows produce diagnostics but cannot produce resolved paper outcomes or positions.

Known-at: **F** = firing close + completed profiles/HTF + actual ML/policy availability; **I** = confirmed sweep/MSS/FVG/retest close + actual ML/policy availability; **B** = failed OB formation < activation < retest/rejection close + actual ML/policy availability; **O** = online ML/policy availability, and swing/zigzag or breakout confirmation for rule-event replay. Always retain the separate historical anchor.

Entry **N** = first independently scheduled open strictly after known-at. Missing rule **G** = no forward fill or delayed search; missing expected open/holding bar or coordinate change => DATA_GAP; exhausted entry calendar => NO_ENTRY; planned closures follow the frozen schedule, holding permitted. **S** = structural stop at least max(0.25 ATR, one tick); **T2/T2.5** = farther of supplied opposing target, 2/2.5 ATR and 1.25R. Costs below are separate position-simulation contracts.

| instrument | strategy | setup/model family | known_at | earliest entry | stop | target | timeout | paper management | cost model | missing-bar rule | contract_id |
|---|---|---|---|---|---|---|---|---|---|---|
| ES | frvp 5m | setup1; reversal, meta, reversal_setup1 | F | N | 1 ATR | 0.8 ATR | 60 bars / 300m | fixed barriers | es_session_cost_v1 | G | `es_frvp_setup1_eval_v1` |
| ES | frvp 5m | setup2; continuation, meta, continuation_setup2 | F | N | UNRESOLVED | UNRESOLVED | UNRESOLVED | fixed barriers | es_session_cost_v1 | G | `es_frvp_setup2_eval_v1` |
| ES | frvp 5m | setup3; continuation, meta, continuation_setup3 | F | N | UNRESOLVED | UNRESOLVED | UNRESOLVED | fixed barriers | es_session_cost_v1 | G | `es_frvp_setup3_eval_v1` |
| ES | frvp 5m | setup4; reversal, meta, reversal_setup4 | F | N | UNRESOLVED | UNRESOLVED | UNRESOLVED | fixed barriers | es_session_cost_v1 | G | `es_frvp_setup4_eval_v1` |
| ES | frvp 5m | setup5; continuation, meta, continuation_setup5 | F | N | UNRESOLVED | UNRESOLVED | UNRESOLVED | fixed barriers | es_session_cost_v1 | G | `es_frvp_setup5_eval_v1` |
| ES | frvp 5m | setup6; reversal, meta, reversal_setup6 | F | N | 1 ATR | 0.8 ATR | 60 bars / 300m | fixed barriers | es_session_cost_v1 | G | `es_frvp_setup6_eval_v1` |
| ES | frvp 5m | unmatched; reversal, continuation, meta | F | N | UNRESOLVED | UNRESOLVED | UNRESOLVED | fixed barriers | es_session_cost_v1 | G | `es_frvp_unmatched_eval_v1` |
| 6E | frvp 5m | setup1; reversal, meta, reversal_setup1 | F | N | 1 ATR | 0.8 ATR | 60 bars / 300m | fixed barriers | 6e_session_cost_v1 | G | `6e_frvp_setup1_eval_v1` |
| 6E | frvp 5m | setup2; continuation, meta, continuation_setup2 | F | N | UNRESOLVED | UNRESOLVED | UNRESOLVED | fixed barriers | 6e_session_cost_v1 | G | `6e_frvp_setup2_eval_v1` |
| 6E | frvp 5m | setup3; continuation, meta, continuation_setup3 | F | N | UNRESOLVED | UNRESOLVED | UNRESOLVED | fixed barriers | 6e_session_cost_v1 | G | `6e_frvp_setup3_eval_v1` |
| 6E | frvp 5m | setup4; reversal, meta, reversal_setup4 | F | N | UNRESOLVED | UNRESOLVED | UNRESOLVED | fixed barriers | 6e_session_cost_v1 | G | `6e_frvp_setup4_eval_v1` |
| 6E | frvp 5m | setup5; continuation, meta, continuation_setup5 | F | N | UNRESOLVED | UNRESOLVED | UNRESOLVED | fixed barriers | 6e_session_cost_v1 | G | `6e_frvp_setup5_eval_v1` |
| 6E | frvp 5m | setup6; reversal, meta, reversal_setup6 | F | N | 1 ATR | 0.8 ATR | 60 bars / 300m | fixed barriers | 6e_session_cost_v1 | G | `6e_frvp_setup6_eval_v1` |
| 6E | frvp 5m | unmatched; reversal, continuation, meta | F | N | UNRESOLVED | UNRESOLVED | UNRESOLVED | fixed barriers | 6e_session_cost_v1 | G | `6e_frvp_unmatched_eval_v1` |
| ES | ict 5m | sweep_reclaim; reversal, meta, reversal_sweep_reclaim | I | N | S | T2 | 18 bars / 90m | fixed barriers | es_session_cost_v1 | G | `es_ict_sweep_reclaim_eval_v1` |
| ES | ict 5m | sweep_displacement_fvg; reversal, meta, reversal_sweep_displacement_fvg | I | N | S | T2 | 18 bars / 90m | fixed barriers | es_session_cost_v1 | G | `es_ict_sweep_displacement_fvg_eval_v1` |
| ES | ict 5m | ob_retest_after_mss; reversal, meta, reversal_ob_retest_after_mss | I | N | S | T2 | 18 bars / 90m | fixed barriers | es_session_cost_v1 | G | `es_ict_ob_retest_after_mss_eval_v1` |
| ES | ict 5m | ifvg_reversal; reversal, meta, reversal_ifvg_reversal | I | N | S | T2 | 18 bars / 90m | fixed barriers | es_session_cost_v1 | G | `es_ict_ifvg_reversal_eval_v1` |
| ES | ict 5m | premium_discount_continuation; continuation, meta, continuation_premium_discount_continuation | I | N | S | T2.5 | 24 bars / 120m | fixed barriers | es_session_cost_v1 | G | `es_ict_premium_discount_continuation_eval_v1` |
| ES | ict 5m | session_open_manipulation_pre_ib; reversal, meta, reversal_session_open_manipulation_pre_ib | I | N | S | T2 | 12 bars / 60m | fixed barriers | es_session_cost_v1 | G | `es_ict_session_open_manipulation_pre_ib_eval_v1` |
| ES | ict 5m | session_open_manipulation_post_ib; reversal, meta, reversal_session_open_manipulation_post_ib | I | N | S | T2 | 12 bars / 60m | fixed barriers | es_session_cost_v1 | G | `es_ict_session_open_manipulation_post_ib_eval_v1` |
| ES | ict 5m | displacement_continuation_after_raid; continuation, meta, continuation_displacement_continuation_after_raid | I | N | S | T2.5 | 24 bars / 120m | fixed barriers | es_session_cost_v1 | G | `es_ict_displacement_continuation_after_raid_eval_v1` |
| ES | ict 5m | classic_breaker; classic_breaker | B | N | UNRESOLVED | UNRESOLVED | UNRESOLVED | fixed barriers | es_session_cost_v1 | G | `es_ict_classic_breaker_eval_v1` |
| ES | ict 5m | unmatched; reversal, continuation, meta | I | N | UNRESOLVED | UNRESOLVED | UNRESOLVED | fixed barriers | es_session_cost_v1 | G | `es_ict_unmatched_eval_v1` |
| 6E | ict 5m | sweep_reclaim; reversal, meta, reversal_sweep_reclaim | I | N | S | T2 | 18 bars / 90m | fixed barriers | 6e_session_cost_v1 | G | `6e_ict_sweep_reclaim_eval_v1` |
| 6E | ict 5m | sweep_displacement_fvg; reversal, meta, reversal_sweep_displacement_fvg | I | N | S | T2 | 18 bars / 90m | fixed barriers | 6e_session_cost_v1 | G | `6e_ict_sweep_displacement_fvg_eval_v1` |
| 6E | ict 5m | ob_retest_after_mss; reversal, meta, reversal_ob_retest_after_mss | I | N | S | T2 | 18 bars / 90m | fixed barriers | 6e_session_cost_v1 | G | `6e_ict_ob_retest_after_mss_eval_v1` |
| 6E | ict 5m | ifvg_reversal; reversal, meta, reversal_ifvg_reversal | I | N | S | T2 | 18 bars / 90m | fixed barriers | 6e_session_cost_v1 | G | `6e_ict_ifvg_reversal_eval_v1` |
| 6E | ict 5m | premium_discount_continuation; continuation, meta, continuation_premium_discount_continuation | I | N | S | T2.5 | 24 bars / 120m | fixed barriers | 6e_session_cost_v1 | G | `6e_ict_premium_discount_continuation_eval_v1` |
| 6E | ict 5m | session_open_manipulation_pre_ib; reversal, meta, reversal_session_open_manipulation_pre_ib | I | N | S | T2 | 12 bars / 60m | fixed barriers | 6e_session_cost_v1 | G | `6e_ict_session_open_manipulation_pre_ib_eval_v1` |
| 6E | ict 5m | session_open_manipulation_post_ib; reversal, meta, reversal_session_open_manipulation_post_ib | I | N | S | T2 | 12 bars / 60m | fixed barriers | 6e_session_cost_v1 | G | `6e_ict_session_open_manipulation_post_ib_eval_v1` |
| 6E | ict 5m | displacement_continuation_after_raid; continuation, meta, continuation_displacement_continuation_after_raid | I | N | S | T2.5 | 24 bars / 120m | fixed barriers | 6e_session_cost_v1 | G | `6e_ict_displacement_continuation_after_raid_eval_v1` |
| 6E | ict 5m | classic_breaker; classic_breaker | B | N | UNRESOLVED | UNRESOLVED | UNRESOLVED | fixed barriers | 6e_session_cost_v1 | G | `6e_ict_classic_breaker_eval_v1` |
| 6E | ict 5m | unmatched; reversal, continuation, meta | I | N | UNRESOLVED | UNRESOLVED | UNRESOLVED | fixed barriers | 6e_session_cost_v1 | G | `6e_ict_unmatched_eval_v1` |
| EURUSD | ote 5m | reversal; reversal, meta, ote, union | O | N | UNRESOLVED | UNRESOLVED | UNRESOLVED | fixed barriers | eurusd_session_cost_v1 | G | `eurusd_ote_reversal_eval_v1` |
| EURUSD | ote 1m | reversal; reversal, meta, ote, union | O | N | UNRESOLVED | UNRESOLVED | UNRESOLVED | fixed barriers | eurusd_session_cost_v1 | G | `eurusd_ote_reversal_1m_eval_v1` |
| EURUSD | ote 5m | continuation; continuation, meta, ote, union | O | N | UNRESOLVED | UNRESOLVED | UNRESOLVED | fixed barriers | eurusd_session_cost_v1 | G | `eurusd_ote_continuation_eval_v1` |
| EURUSD | ote 5m | breakout; breakout, meta, ote, union | O | N | UNRESOLVED | UNRESOLVED | UNRESOLVED | fixed barriers | eurusd_session_cost_v1 | G | `eurusd_ote_breakout_eval_v1` |
| EURUSD | ote 5m | breakout_sustained; breakout_sustained, meta, ote, union | O | N | UNRESOLVED | UNRESOLVED | UNRESOLVED | fixed barriers | eurusd_session_cost_v1 | G | `eurusd_ote_breakout_sustained_eval_v1` |
| EURUSD | ote 5m | unmatched; reversal, continuation, meta, ote, union, breakout, breakout_sustained | O | N | UNRESOLVED | UNRESOLVED | UNRESOLVED | fixed barriers | eurusd_session_cost_v1 | G | `eurusd_ote_unmatched_eval_v1` |
| EURUSD | ote 1m | unmatched; reversal, continuation, meta, ote, union, breakout, breakout_sustained | O | N | UNRESOLVED | UNRESOLVED | UNRESOLVED | fixed barriers | eurusd_session_cost_v1 | G | `eurusd_ote_unmatched_1m_eval_v1` |
