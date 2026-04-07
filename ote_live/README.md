# OTE Live Stack

The live stack now supports repo-root `.env` loading for the Python entrypoints under `ote_live/scripts`.

## Install

```powershell
pip install -r ote_requirements.txt
```

## Minimum `.env`

```dotenv
TWELVEDATA_API_KEY=replace_me
```

Optional launcher settings:

```dotenv
OTE_LIVE_ASSET=EURUSD
OTE_LIVE_DB_PATH=ote_live/runtime_data/live_market_data.sqlite3
OTE_LIVE_COLLECTOR_POLL_INTERVAL_SECONDS=120
OTE_LIVE_STREAM_OUTPUTSIZE=2
OTE_LIVE_REGISTRY_PATH=models/ote_model_registry_v1_v2_candidates.json
OTE_LIVE_DASHBOARD_HOST=127.0.0.1
OTE_LIVE_DASHBOARD_PORT=8050
OTE_LIVE_DASHBOARD_TIMEFRAME=5m
OTE_LIVE_DASHBOARD_REFRESH_INTERVAL_MS=10000
OTE_LIVE_DASHBOARD_SIGNAL_LIMIT=120
OTE_LIVE_DASHBOARD_URL=http://127.0.0.1:8050
OTE_LIVE_LONG_RUNTIME_MANIFEST_PATH=ote_live/runtime_manifests/live_runtime_manifest_long.json
OTE_LIVE_SHORT_RUNTIME_MANIFEST_PATH=ote_live/runtime_manifests/live_runtime_manifest_short.json
OTE_LIVE_OPEN_BROWSER=1
OTE_LIVE_ALERT_EMAIL_RECIPIENTS=
OTE_LIVE_ALERT_SMS_RECIPIENTS=
```

## One-command startup

PowerShell:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/start_ote_live_stack.ps1
```

Batch:

```cmd
scripts\start_ote_live_stack.bat
```

Python module:

```powershell
python -m ote_live.scripts.run_live_stack
```

What the launcher does:

1. Packages live policy artifacts.
2. Exports fresh runtime manifests from the v1/v2 candidate registry so the live primaries default to the long and short TCN v2 models.
3. Starts the Dash dashboard.
4. Starts the live collector with signal generation and SQLite storage.

The dashboard now stays pinned to the configured long and short primary models from the runtime manifests. If a configured primary has not produced stored live predictions yet, its primary confidence and inspection panels stay empty instead of silently switching to another model. The main price chart still shows all stored model signals.

The live collector now defaults to one TwelveData poll every two minutes with a
small `outputsize`, which keeps steady-state polling near 720 requests per day
and leaves some room under an 800-credit plan for restarts or occasional gap
backfills.

## Useful options

```powershell
python -m ote_live.scripts.run_live_stack --dry-run
python -m ote_live.scripts.run_live_stack --no-open-browser
python -m ote_live.scripts.run_live_stack --skip-policy-package --skip-manifest-export
python -m ote_live.scripts.run_live_stack --collector-arg=--max-cycles --collector-arg=5
```

Direct entrypoints still work and now auto-load `.env`:

```powershell
python -m ote_live.scripts.run_live_collector
python -m ote_live.scripts.run_live_dashboard
python -m ote_live.scripts.package_candidate_policies
python -m ote_live.scripts.export_live_runtime_manifests
```
