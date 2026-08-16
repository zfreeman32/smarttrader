# IBKR ES Live Market-Data Setup

The existing `ote_live` collector now owns one read-only official TWS API
connection per process. It discovers the live E-mini S&P 500 quarterly futures
contract, streams top-of-book quotes, backfills two days of five-minute bars,
keeps the active candle updated, and mirrors cached state into the same SQLite
store used by the OTE/FRVP/ICT dashboard. No order API is implemented.

## 1. Install and configure TWS or IB Gateway

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

## 2. Install the official Python API

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

## 3. Configure the feed

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

## 4. Smoke test

With the paper TWS/Gateway session running:

```powershell
python -m ote_live.scripts.test_ibkr_es_feed
```

The script prints the qualified contract, next roll time, a limited number of
quote/bar changes, and then exits cleanly. Press Ctrl+C to stop sooner. It has
no order-placement methods.

## 5. Run the existing application

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

## 6. Status and recovery

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

## 7. Automatic roll and overrides

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
