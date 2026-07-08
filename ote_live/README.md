# OTE Live Stack

The live Dash app now serves both operator views from one process:

- `OTE`: the existing EURUSD / OTE live view.
- `FRVP`: an ES `5m` shadow view fed by IBKR and rendered from persisted runtime state.

The FRVP tab does not talk to IBKR directly. The collector writes completed ES bars, model outputs, and FRVP chart/setup state into the shared SQLite store, and the dashboard reads that store.

## Install

```powershell
pip install -r ote_requirements.txt
```

`ote_requirements.txt` now includes `ib_insync` for the IBKR-backed FRVP collector.

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
FRVP_LIVE_REGISTRY_PATH=models/frvp_es_shadow_live_registry_20260705.json
FRVP_LIVE_LONG_RUNTIME_MANIFEST_PATH=ote_live/runtime_manifests/frvp_es_shadow_20260705/live_runtime_manifest_long.json
FRVP_LIVE_SHORT_RUNTIME_MANIFEST_PATH=ote_live/runtime_manifests/frvp_es_shadow_20260705/live_runtime_manifest_short.json

IBKR_HOST=127.0.0.1
IBKR_PORT=7497
IBKR_CLIENT_ID=17
IBKR_MARKET_DATA_TYPE=1
IBKR_ES_SYMBOL=ES
IBKR_ES_EXCHANGE=CME
IBKR_ES_CURRENCY=USD
IBKR_ES_LOCAL_SYMBOL=ESU6
# Or:
# IBKR_ES_CONTRACT_MONTH=202609
IBKR_USE_RTH=false
IBKR_BAR_SIZE=5 mins
IBKR_WHAT_TO_SHOW=TRADES
IBKR_KEEP_UP_TO_DATE=true
IBKR_BACKFILL_DURATION=2 D
```

Keep `FRVP_LIVE_DB_PATH` aligned with `OTE_LIVE_DB_PATH` if you want the combined dashboard to show both tabs from the same store.

## Run Commands

Combined dashboard with tabs:

```powershell
python -m ote_live.scripts.run_live_dashboard
```

Existing OTE collector:

```powershell
python -m ote_live.scripts.run_live_collector --dashboard-url http://127.0.0.1:8050
```

FRVP IBKR collector:

```powershell
python -m ote_live.scripts.run_frvp_live_collector --dashboard-url http://127.0.0.1:8050
```

Typical operator flow for both tabs:

```powershell
python -m ote_live.scripts.run_live_collector --dashboard-url http://127.0.0.1:8050
python -m ote_live.scripts.run_frvp_live_collector --dashboard-url http://127.0.0.1:8050
python -m ote_live.scripts.run_live_dashboard
```

If you only launch the FRVP collector, the FRVP tab will populate and the OTE tab will still render, but it will only show data already present in the shared store.

## FRVP Defaults

The FRVP tab and collector default to the shadow bundle already in the repo:

- `models/frvp_es_shadow_live_registry_20260705.json`
- `ote_live/runtime_manifests/frvp_es_shadow_20260705/live_runtime_manifest_long.json`
- `ote_live/runtime_manifests/frvp_es_shadow_20260705/live_runtime_manifest_short.json`
- `ote_live/runtime_manifests/frvp_es_shadow_20260705/shadow_selection_summary.json`
- `ote_live/policy_artifacts/frvp_es_shadow_20260705/`
- `model_testing/reports/frvp_backtests/frvp_es_shadow_live_bundle_20260705/run_summary.json`

The FRVP dashboard ordering prioritizes:

1. `frvp_long_continuation_xgb_v1`
2. `frvp_long_reversal_xgb_v1`
3. `frvp_short_meta_xgb_v1`

The tab still renders every model loaded from the long/short FRVP manifests.

## Dashboard Notes

The tabbed dashboard keeps the existing OTE card layout and adds:

- ES `5m` price bars on the FRVP tab.
- Persisted FRVP overlays for `POC`, `VAH`, `VAL`, `IB high`, `IB low`, and nearest naked VPOCs when the runtime has those distances available.
- FRVP setup markers sourced from persisted runtime state.
- Per-model probability / threshold / decision cards using the saved live policies from the manifests.

The FRVP runtime persists its chart/setup state into `runtime_state` under the `FRVP` dashboard key so Dash callbacks do not need to recompute expensive FRVP context on refresh.

## Existing OTE Launcher

The legacy convenience launcher still exists:

```powershell
python -m ote_live.scripts.run_live_stack
```

It remains OTE-oriented. FRVP v1 is launched with the dedicated `run_frvp_live_collector` command above.

## IBKR Requirements And Limits

- TWS or IB Gateway must be running locally with API access enabled.
- Real-time ES data requires the correct CME market data subscription.
- The collector requests `TRADES` bars so ES volume remains real traded contract volume.
- v1 uses safe historical-bar polling for ongoing updates. `IBKR_KEEP_UP_TO_DATE` is accepted in config, but the current collector does not open a persistent `keepUpToDate` subscription.
- IBKR pacing still applies, so the runtime caches bars locally and backfills only missing windows.
- Completed `5m` bars only are used for FRVP inference and chart state.
- Safe ES auto-roll handling is not implemented in v1.
- You must set the active front-month contract manually with `IBKR_ES_LOCAL_SYMBOL`, `IBKR_ES_CONTRACT_MONTH`, or `conId`.
- Do not let FRVP absolute profile levels span silent contract-coordinate changes.
