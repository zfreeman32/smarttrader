# OTE Live Stack

The live Dash app now serves both operator views from one process:

- `OTE`: the existing EURUSD / OTE live view.
- `FRVP`: an ES `5m` shadow view fed by IBKR and rendered from persisted runtime state.
- `ICT`: an ES `5m` controlled paper-signal view fed by the same IBKR ES stream and rendered from persisted runtime state.

The FRVP and ICT tabs do not talk to IBKR directly. The shared ES collector writes completed ES bars, model outputs, and FRVP/ICT chart state into the shared SQLite store, and the dashboard reads that store.

FRVP and ICT also retain an append-only setup history beyond the chart's latest
40 events, including rejected candidates, prediction links, source corrections
and diagnostic follow-up candles. See the [A5 storage contract](../docs/live_app_change_journal.md#imported-a5)
for query interfaces, retention and outcome limitations.

## Install

```powershell
pip install -r requirements.txt
```

For IBKR, separately install the official `ibapi` client from the TWS API
`source/pythonclient` directory; see the setup guide below.

## Example Env

Use `ote_live/.env.example` as the starting point.

Minimum OTE-only env:

```dotenv
FMP_API_KEY=replace_me
OTE_LIVE_DB_PATH=ote_live/runtime_data/live_market_data.sqlite3
```

Example FRVP / IBKR additions:

```dotenv
FRVP_LIVE_ASSET=ES
FRVP_LIVE_TIMEFRAME=5m
FRVP_LIVE_DATA_SUPPLIER=IBKR
FRVP_LIVE_DB_PATH=ote_live/runtime_data/live_market_data.sqlite3
FRVP_LIVE_REGISTRY_PATH=models/frvp_es_shadow_live_registry_20260721.json
FRVP_LIVE_LONG_RUNTIME_MANIFEST_PATH=ote_live/runtime_manifests/frvp_es_shadow_20260721/live_runtime_manifest_long.json
FRVP_LIVE_SHORT_RUNTIME_MANIFEST_PATH=ote_live/runtime_manifests/frvp_es_shadow_20260721/live_runtime_manifest_short.json

ICT_LIVE_ASSET=ES
ICT_LIVE_TIMEFRAME=5m
ICT_LIVE_DATA_SUPPLIER=IBKR
ICT_LIVE_DB_PATH=ote_live/runtime_data/live_market_data.sqlite3
ICT_LIVE_REGISTRY_PATH=models/ict_es_paper_signal_registry_20260813.json
ICT_LIVE_LONG_RUNTIME_MANIFEST_PATH=ote_live/runtime_manifests/ict_es_paper_signal_20260813/live_runtime_manifest_long.json
ICT_LIVE_SHORT_RUNTIME_MANIFEST_PATH=ote_live/runtime_manifests/ict_es_paper_signal_20260813/live_runtime_manifest_short.json
ES_LIVE_ALL_MODELS_ACTIVE=false
ICT_PAPER_SIGNAL_TRIAL_ENABLED=false

IBKR_ENABLED=false
IBKR_HOST=127.0.0.1
IBKR_PORT=7497
IBKR_CLIENT_ID=21
IBKR_ACCOUNT_MODE=paper
IBKR_MARKET_DATA_TYPE=live
IBKR_ALLOW_DELAYED_FALLBACK=false
IBKR_ES_SYMBOL=ES
IBKR_ES_SECURITY_TYPE=FUT
IBKR_ES_EXCHANGE=CME
IBKR_ES_CURRENCY=USD
IBKR_ES_MULTIPLIER=50
IBKR_ES_TRADING_CLASS=ES
IBKR_ES_BAR_SIZE=5 mins
IBKR_ES_HISTORY_DURATION=2 D
IBKR_ES_USE_RTH=false
IBKR_DELAYED_HISTORY_REFRESH_SECONDS=240
IBKR_ES_ROLL_POLICY=cme_calendar
IBKR_ES_ROLL_TIME_ET=09:30
```

Keep `FRVP_LIVE_DB_PATH` and `ICT_LIVE_DB_PATH` aligned with `OTE_LIVE_DB_PATH` if you want the combined dashboard to show all three tabs from the same store.

## Signal Notifications

`OTE_LIVE_ALERT_EMAIL_RECIPIENTS` and `OTE_LIVE_ALERT_SMS_RECIPIENTS` are the
shared defaults for the OTE, FRVP, and ICT live collectors. The FRVP-only
collector can override them with `FRVP_LIVE_ALERT_*`; the shared FRVP + ICT ES
collector can override them with `ES_LIVE_ALERT_*`.

Notifications are sent only when an active model emits an actionable signal.
Candidate models retain shadow status and remain visible in the audit trail,
but do not notify when their calibrated probability clears a live threshold.
Automatic signal chart-image saving is temporarily disabled.
SMS bodies include the model id so the firing OTE, FRVP, or ICT model is
identifiable.

## Run Commands

Combined dashboard with tabs:

```powershell
python -m ote_live.scripts.run_live_dashboard
```

Existing OTE collector:

```powershell
python -m ote_live.scripts.run_live_collector --dashboard-url http://127.0.0.1:8050
```

Ongoing shared ES IBKR feed with ICT held out:

```powershell
python -m ote_live.scripts.run_es_live_collector --exclude-ict --dashboard-url http://127.0.0.1:8050
```

The controlled ICT bundle is fail-closed. While its readiness audit is blocked,
an operator may restore and verify the shared paper feed without evaluating ICT:

```powershell
python -m ote_live.scripts.run_es_live_collector --exclude-ict --max-cycles 1 --dashboard-url http://127.0.0.1:8050
python scripts/audit_ict_paper_signal_readiness.py --preflight --allow-clean-stopped-handoff
```

The one-cycle bootstrap must exit normally. Its terminal heartbeat is eligible
for handoff only when it says `stopped`, remains healthy and non-stale, matches
the exact shared ES `5m` service identity, and is no more than 120 seconds old.
Cancelled, failed, unhealthy, stale, wrong-identity, and older snapshots remain
blocked. The default audit still requires a running collector; the explicit
`--allow-clean-stopped-handoff` flag is required for this bounded transition.

Keep `ICT_PAPER_SIGNAL_TRIAL_ENABLED=false` during preflight. Resolve every
reported blocker until the audit returns `status=ready_except_enable_switch`;
that status exits successfully and identifies the switch as the sole remaining
activation step. Then set `ICT_PAPER_SIGNAL_TRIAL_ENABLED=true` and, within the
same 120-second window, launch:

```powershell
python -m ote_live.scripts.run_es_live_collector --include-ict --allow-ict-clean-handoff --dashboard-url http://127.0.0.1:8050
```

The collector reruns the complete clean-handoff audit before constructing any
ICT runtime and accepts only `ready_to_start` for the exact August 13 bundle.
The 28-day clock starts only at the final collector's first healthy `running`
heartbeat, never at bootstrap or from the stopped handoff snapshot. The trial
additionally requires `IBKR_ENABLED=true`, an authenticated paper endpoint on
`4002` or `7497`, and `IBKR_ALLOW_DELAYED_FALLBACK=false`. Saved feature parity
and confirmation-ledger readiness are also required.

The app launcher supports the same non-overlapping flow:

```powershell
.\app_start_cmd.ps1 -PrepareIctHandoff
# Wait for the one-cycle ES bootstrap to exit cleanly, then run the preflight above.
# Set ICT_PAPER_SIGNAL_TRIAL_ENABLED=true only after ready_except_enable_switch.
.\app_start_cmd.ps1 -IncludeIct
```

`-IncludeIct` verifies the exact fresh clean stopped snapshot and the final
audit before it stops or starts any app process. A normal launcher run keeps ICT
excluded. Renamed, copied, or custom ICT manifest paths fail closed; use
`--exclude-ict` when intentionally running the shared feed without this trial.

For dashboard-only testing on delayed IBKR data, use the explicit shadow mode:

```powershell
.\app_start_cmd.ps1 -IncludeIctDelayedTest
```

This starts ICT without the clean-handoff paper-trial audit, forces ICT models
to shadow decisions, and disables ICT paper-signal ledger writes. It is for
fresh dashboard overlays on delayed data; it does not authorize the controlled
paper-signal trial.

For operator preflight and launch, run the audit with its default bundle,
validation, heartbeat-age, and heartbeat paths. Its path and age CLI overrides
exist for isolated tests and diagnostics; they do not relax the collector's
exact-bundle launch guard.

The FRVP decision bundle is separately fail-closed. The normal shared-collector
defaults remain on the July 21 all-shadow bundle until the controlled trial is
ready, so a healthy one-cycle shadow run can provide the bounded handoff:

```powershell
python -m ote_live.scripts.run_es_live_collector --exclude-ict --max-cycles 1 --dashboard-url http://127.0.0.1:8050
python scripts/audit_frvp_paper_signal_readiness.py --preflight --allow-clean-stopped-handoff
```

Keep `FRVP_PAPER_SIGNAL_TRIAL_ENABLED=false` until preflight reports
`ready_except_enable_switch`. Then set it to `true` and, inside the same
120-second clean-handoff window, launch the exact reversal-only bundle:

```powershell
python -m ote_live.scripts.run_es_live_collector --include-frvp --exclude-ict --allow-frvp-clean-handoff `
  --frvp-long-runtime-manifest-path ote_live/runtime_manifests/frvp_es_paper_signal_20260816/live_runtime_manifest_long.json `
  --frvp-short-runtime-manifest-path ote_live/runtime_manifests/frvp_es_paper_signal_20260816/live_runtime_manifest_short.json `
  --dashboard-url http://127.0.0.1:8050
```

Point the separate dashboard process at the same controlled bundle so its
status badges and policy copy show reversal as the only active-weight model:

```powershell
python -m ote_live.scripts.run_live_dashboard `
  --frvp-long-runtime-manifest-path ote_live/runtime_manifests/frvp_es_paper_signal_20260816/live_runtime_manifest_long.json `
  --frvp-short-runtime-manifest-path ote_live/runtime_manifests/frvp_es_paper_signal_20260816/live_runtime_manifest_short.json `
  --frvp-registry-path models/frvp_es_paper_signal_registry_20260816.json
```

The dashboard grants the controlled label only when all three files are at the
frozen paths and match their immutable SHA-256 digests. Its FRVP confidence and
signal history is then restricted to those exact persisted runtime-manifest
hashes, so July shadow rows with the same model IDs cannot appear in the August
view. `Paper Markout` reads the dedicated FRVP event ledger and reports the full
trial's 120-completed-bar, friction-adjusted ES net ticks; it does not reuse the
legacy three-bar FX-pip calculation.

The collector detects any active FRVP manifest by content, rejects renamed or
copied active bundles, reruns the full default readiness audit, and requires
paper mode on `4002` or `7497` with delayed fallback disabled. The 28-day clock
starts only at the final process's first healthy `running` heartbeat. This trial
records independent 120-completed-bar close markouts; it never submits or tracks
broker orders. The standalone FRVP collector is not the controlled-trial entry
point.

Optional FRVP-only ES collector:

```powershell
python -m ote_live.scripts.run_frvp_live_collector --dashboard-url http://127.0.0.1:8050
```

Typical operator flow for all three tabs:

```powershell
python -m ote_live.scripts.run_live_collector --dashboard-url http://127.0.0.1:8050
python -m ote_live.scripts.run_es_live_collector --include-ict --allow-ict-clean-handoff --dashboard-url http://127.0.0.1:8050
python -m ote_live.scripts.run_live_dashboard
```

The shared ES collector evaluates both FRVP and ICT from one IBKR polling loop, so do not run a second independent ICT ES collector in parallel.

## FRVP Defaults

The FRVP tab and collector default to the current extended-shadow bundle in the repo:

- `models/frvp_es_shadow_live_registry_20260721.json`
- `ote_live/runtime_manifests/frvp_es_shadow_20260721/live_runtime_manifest_long.json`
- `ote_live/runtime_manifests/frvp_es_shadow_20260721/live_runtime_manifest_short.json`
- `ote_live/runtime_manifests/frvp_es_shadow_20260721/shadow_selection_summary.json`
- `ote_live/policy_artifacts/frvp_es_shadow_20260721/`
- `model_testing/reports/frvp_backtests/frvp_es_shadow_live_bundle_20260721/run_summary.json`

The FRVP dashboard ordering prioritizes:

1. `frvp_long_continuation_xgb_v1`
2. `frvp_long_reversal_xgb_v1`
3. `frvp_short_meta_xgb_v1`

The tab still renders every model loaded from the long/short FRVP manifests.

The July 21, 2026 FRVP bundle uses:

- `frvp_long_continuation_xgb_v1` v3 as the primary extended-shadow baseline
- `frvp_long_reversal_xgb_recent_regime_prune_v2` as the operational selective-deployment reversal contract

The immutable August 16 decision bundle is staged separately:

- `models/frvp_es_paper_signal_registry_20260816.json`
- `ote_live/runtime_manifests/frvp_es_paper_signal_20260816/`
- `ote_live/policy_artifacts/frvp_es_paper_signal_20260816/`
- `model_testing/reports/frvp_paper_signal_bundles/frvp_es_paper_signal_20260816/run_summary.json`

Only `frvp_long_reversal_xgb_v1` is active there, under the corrected global
`0.60` recent-regime contract with all 11 accepted filters. Continuation remains
candidate/shadow because its exact fixed global-`0.70` sensitivity missed the
Sharpe, profitable-quarter, and single-trade-concentration gates. Reversal's
fixed-policy sensitivity remained strong but missed concentration, so the
controlled paper-signal window is specifically a concentration-validation trial,
not full promotion. The deferred human TradingView signoff is waived only for
this no-order trial and remains required for full promotion.

## ICT Defaults

The ICT tab defaults to the August 13, 2026 controlled paper-signal bundle built
from the leakage-safe July 26 full retrain:

- `models/ict_es_paper_signal_registry_20260813.json`
- `ote_live/runtime_manifests/ict_es_paper_signal_20260813/live_runtime_manifest_long.json`
- `ote_live/runtime_manifests/ict_es_paper_signal_20260813/live_runtime_manifest_short.json`
- `ote_live/policy_artifacts/ict_es_paper_signal_20260813/`

The ICT dashboard ordering prioritizes:

1. `ict_long_meta_xgb_v1`
2. `ict_long_reversal_xgb_v1`
3. `ict_short_reversal_xgb_v1`
4. `ict_short_meta_xgb_v1`
5. `ict_short_continuation_xgb_v1`
6. `ict_long_continuation_xgb_v1`

`ict_long_meta_xgb_v1` is the sole active paper-signal model and uses the exact
accepted walk-forward contract: a global `0.40` threshold with abstention
disabled. `ict_long_reversal_xgb_v1` remains a candidate/shadow challenger
because its historical edge is substantially non-additive with long-meta; the
other four branches also remain candidate/shadow. Keep
`ES_LIVE_ALL_MODELS_ACTIVE=false` so the runtime cannot bypass these lifecycle
decisions.

The shared collector also fails closed while
`ICT_PAPER_SIGNAL_TRIAL_ENABLED=false`. Leave it false while running
`scripts/audit_ict_paper_signal_readiness.py --preflight --allow-clean-stopped-handoff`.
Set it true only after the preflight reports
`ready_except_enable_switch`, immediately before starting the controlled trial
with `--allow-ict-clean-handoff`. The collector then requires its own final
audit to report `ready_to_start` before ICT runtime construction.

In this repository, active paper-signal status permits persisted `emit`
decisions and configured email/SMS notifications. It does not place broker
orders or maintain a broker fill/position/P&L ledger. Start the four-week
confirmation window only after the readiness audit passes against an
authenticated paper TWS/IB Gateway session and a current healthy heartbeat.

## Dashboard Notes

The tabbed dashboard keeps the existing OTE card layout and adds:

- ES `5m` price bars on the FRVP tab.
- ES `5m` price bars on the ICT tab.
- Persisted FRVP overlays for `POC`, `VAH`, `VAL`, `IB high`, `IB low`, and nearest naked VPOCs when the runtime has those distances available.
- Persisted ICT overlays for key session/liquidity levels plus the nearest live FVG/order-block zones and setup-fire markers.
- FRVP setup markers sourced from persisted runtime state.
- ICT setup markers sourced from persisted runtime state via `detect_ict_setups`.
- Per-model probability / threshold / decision cards using the saved live policies from the manifests.

The FRVP runtime persists its chart/setup state into `runtime_state` under the `FRVP` dashboard key, and the ICT runtime does the same under `ICT`, so Dash callbacks do not need to recompute those detector surfaces on refresh.

## Existing OTE Launcher

The legacy convenience launcher still exists:

```powershell
python -m ote_live.scripts.run_live_stack
```

It remains OTE-oriented. The dedicated `run_frvp_live_collector` command above
is available for the legacy FRVP shadow bundle only; the August 16 controlled
trial must use the readiness-bound shared ES collector documented earlier.

## IBKR Requirements And Limits

- The production ES provider uses the official callback-based `ibapi` client. It
  does not use `ib_insync`, and `ibapi` is intentionally installed separately
  from the TWS API source package.
- TWS or IB Gateway must remain running and authenticated with socket clients
  enabled. Use paper trading and read-only API mode first.
- Real-time ES data requires the correct CME market data subscription.
- `reqContractDetails` discovers actual quarterly `FUT` contracts and preserves
  `conId`, local symbol, expiration, trading hours, and contract month.
- The default calendar policy rolls at 09:30 America/New_York on the Monday
  before the third Friday of March, June, September, or December. Manual
  `conId` or contract-month overrides are validated before use.
- One `reqHistoricalData(..., keepUpToDate=True)` subscription backfills two
  days of five-minute `TRADES` bars and updates the active candle. A separate
  `reqMktData` subscription supplies bid, ask, last, sizes, and session fields.
- The callback thread writes into a bounded thread-safe cache. Completed bars
  continue through the existing FRVP/ICT inference path; the partial candle and
  quote/status mirror are visualization-only and cannot leak into inference.
- IBKR codes 1100/1101/1102/1300 and local socket failures are handled by the
  managed service. Delayed data is used only when
  `IBKR_ALLOW_DELAYED_FALLBACK=true`.
- Full setup, ports, status interpretation, smoke testing, and troubleshooting:
  [`ote_live/README.md#ibkr_es_live_data_setup`](../ote_live/README.md#ibkr_es_live_data_setup).
- ES observation provenance and the 90-second completed-bar gate are documented
  in [the A4 evidence note](../docs/live_app_change_journal.md#imported-a4). Dated broker
  calendar coverage is required for bar eligibility; delayed/backfilled scores
  retain diagnostic metadata and receive no executable historical entry.
- ES FRVP/ICT shadow decisions separate raw scores, threshold crossings, matching
  setups and full-policy eligibility under the versioned [A6 contract](../docs/live_app_change_journal.md#imported-a6).
  Gate results and rejection reasons are persisted, with collection-scoped queries
  through `LiveAuditRepository.list_shadow_evaluations`. Qualified entries remain
  blocked pending finalized B1/B2/B3 and explicit scored-ledger integration;
  these diagnostics do not open paper positions.
- The [A7 research ledger](../docs/live_app_change_journal.md#imported-a7) supports
  explicit diagnostic execution replay with independent setup/threshold/policy
  books, costs, position limits and censoring. Its [synthetic example](examples/research_execution/README.md)
  runs in a separate database. Scored registration is disabled until the
  outcome/comparison and baseline prerequisites are finalized.

<a id="ibkr_es_live_data_setup"></a>

## IBKR ES Live Market-Data Setup

*Merged 2026-10-09 from `docs/IBKR_ES_live_data_setup.md`.*

The existing `ote_live` collector now owns one read-only official TWS API
connection per process. It discovers the live E-mini S&P 500 quarterly futures
contract, streams top-of-book quotes, backfills two days of five-minute bars,
keeps the active candle updated, and mirrors cached state into the same SQLite
store used by the OTE/FRVP/ICT dashboard. No order API is implemented.

### 1. Install and configure TWS or IB Gateway

1. Install Trader Workstation or IB Gateway and sign in to a **paper** session.
2. In TWS, open **Global Configuration → API → Settings**.
3. Enable **ActiveX and Socket Clients**.
4. Keep **Read-Only API** enabled because this integration retrieves data only.
5. Confirm the socket port:

   | Application | Live | Paper |
   |---|---:|---:|
   | TWS | `7496` | `7497` |
   | IB Gateway | `4001` | `4002` |

6. Keep TWS or IB Gateway running and authenticated whenever the feed runs.
7. Confirm the account has the required CME real-time market-data subscription.
   IBKR error `10089` is surfaced as an error unless delayed fallback was
   explicitly enabled.

### 2. Install the official Python API

Download and install the Interactive Brokers TWS API package. The repository
does not vendor it and does not install an unofficial substitute.

From the default Windows installation directory:

```powershell
cd 'C:\TWS API\source\pythonclient'
C:\Users\zebfr\Documents\All_Files\TRADING\trade_bot\ote_venv\Scripts\python.exe -m pip install .
```

Verify the same Python environment used by the collector:

```powershell
cd C:\Users\zebfr\Documents\All_Files\TRADING\trade_bot
ote_venv\Scripts\python.exe -c "import ibapi; print(ibapi.__file__)"
```

When `IBKR_ENABLED=false`, `ibapi` is optional and unrelated dashboard or CLI
imports continue to work. When enabled without the package, startup reports the
official installation path above.

### 3. Configure the feed

Copy the IBKR section from `ote_live/.env.example` into the repository `.env`.
The safe paper defaults are:

```text
IBKR_ENABLED=false
IBKR_HOST=127.0.0.1
IBKR_PORT=7497
IBKR_CLIENT_ID=21
IBKR_ACCOUNT_MODE=paper
IBKR_MARKET_DATA_TYPE=live
IBKR_ALLOW_DELAYED_FALLBACK=false

IBKR_ES_SYMBOL=ES
IBKR_ES_SECURITY_TYPE=FUT
IBKR_ES_EXCHANGE=CME
IBKR_ES_CURRENCY=USD
IBKR_ES_MULTIPLIER=50
IBKR_ES_TRADING_CLASS=ES

IBKR_ES_BAR_SIZE=5 mins
IBKR_ES_HISTORY_DURATION=2 D
IBKR_ES_USE_RTH=false

IBKR_ES_ROLL_POLICY=cme_calendar
IBKR_ES_ROLL_TIME_ET=09:30
IBKR_ES_MANUAL_CONID=
IBKR_ES_MANUAL_CONTRACT_MONTH=

IBKR_STALE_QUOTE_SECONDS=15
IBKR_STALE_BAR_SECONDS=420
IBKR_RECONNECT_INITIAL_SECONDS=2
IBKR_RECONNECT_MAX_SECONDS=60
```

Set `IBKR_ENABLED=true` only after the paper session is ready. Each running API
client must use a client ID not used by another process.

`IBKR_MARKET_DATA_TYPE` accepts `live`, `frozen`, `delayed`, or
`delayed_frozen`. The provider does not silently switch from live to delayed.
To explicitly permit that fallback, set
`IBKR_ALLOW_DELAYED_FALLBACK=true`.

### 4. Smoke test

With the paper TWS/Gateway session running:

```powershell
python -m ote_live.scripts.test_ibkr_es_feed
```

The script prints the qualified contract, next roll time, a limited number of
quote/bar changes, and then exits cleanly. Press Ctrl+C to stop sooner. It has
no order-placement methods.

### 5. Run the existing application

Run one shared ES collector for FRVP and ICT:

```powershell
python -m ote_live.scripts.run_es_live_collector
```

Run the existing dashboard in another terminal:

```powershell
python -m ote_live.scripts.run_live_dashboard
```

The FRVP and ICT tabs read the shared SQLite cache. Dash callbacks never contact
IBKR, block on the socket, rebuild the service, or start a second API thread.
The active partial candle is shown on the chart; model inference still receives
completed five-minute bars only.

### 6. Status and recovery

- `connected`: live socket and fresh data are flowing.
- `delayed`: IBKR confirmed delayed data, permitted by configuration.
- `stale`: the socket may still be connected, but quote or bar updates exceeded
  the configured threshold during an expected Globex session.
- `rolling`: the next contract is backfilling and being verified before the old
  subscriptions are cancelled.
- `reconnecting`: a socket/server reset is being retried with bounded
  exponential backoff.
- `disconnected` or `error`: inspect the visible IBKR card and collector log.

Normal weekends and the daily 17:00–18:00 America/New_York Globex maintenance
break do not trigger a stale emergency. IBKR code `1101` reconstructs lost
subscriptions; `1102` keeps existing subscriptions intact.

### 7. Automatic roll and overrides

The `cme_calendar` policy calculates the third Friday for each March, June,
September, and December contract. At 09:30 America/New_York on the immediately
preceding Monday, selection moves to the next quarterly contract. The calendar
is calculated dynamically and covers DST, leap years, and December-to-March
transitions.

During a switch, the new actual `FUT` contract is qualified and backfilled
before the active reference changes. Old and new bars retain `conId`, local
symbol, and contract month; prices are not back-adjusted and the dashboard marks
the raw roll gap.

For an operational exception, set exactly one validated override:

```text
IBKR_ES_MANUAL_CONID=123456789
IBKR_ES_MANUAL_CONTRACT_MONTH=
```

or:

```text
IBKR_ES_MANUAL_CONID=
IBKR_ES_MANUAL_CONTRACT_MONTH=202609
```

Remove the override to resume automatic selection. An expired, non-quarterly,
MES, CFD, index, wrong-multiplier, or otherwise ambiguous contract is rejected.
