from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import logging
from pathlib import Path
from queue import Empty, Queue
from threading import Event, RLock, Thread, current_thread
import time
from typing import Any, Callable

from ote_live.storage import SQLiteLiveDataStore

from .client import IBKRAPIClient
from .config import IBKRConfig, market_data_type_name
from .contracts import (
    ContractSelection,
    QualifiedESContract,
    filter_es_contracts,
    is_contract_trading_time,
    select_es_contract,
)
from .errors import IBKRConnectionError, IBKRContractError, IBKRError, IBKRSubscriptionError
from .roll import cme_calendar_roll_at, ensure_utc, is_es_globex_open
from .store import IBKRBar, IBKRMarketDataStore

LOGGER = logging.getLogger(__name__)
IBKR_RUNTIME_STATE_SCOPE = "ibkr.market_data"
_INFORMATIONAL_CODES = frozenset({2104, 2106, 2158})
_MARKET_DATA_SUBSCRIPTION_CODES = frozenset({354, 10089, 10168})


@dataclass(frozen=True)
class RequestRecord:
    request_id: int
    request_type: str
    contract: QualifiedESContract | None
    started_at_utc: datetime
    subscription_state: str = "active"


class RequestIdAllocator:
    def __init__(self, *, start: int = 1) -> None:
        self._lock = RLock()
        self._next = max(1, int(start))
        self._records: dict[int, RequestRecord] = {}

    def set_floor(self, value: int) -> None:
        with self._lock:
            self._next = max(self._next, int(value))

    def allocate(
        self,
        request_type: str,
        contract: QualifiedESContract | None = None,
    ) -> int:
        with self._lock:
            request_id = self._next
            self._next += 1
            self._records[request_id] = RequestRecord(
                request_id=request_id,
                request_type=str(request_type),
                contract=contract,
                started_at_utc=datetime.now(UTC),
            )
            return request_id

    def get(self, request_id: int) -> RequestRecord | None:
        with self._lock:
            return self._records.get(int(request_id))

    def retire(self, request_id: int) -> None:
        with self._lock:
            self._records.pop(int(request_id), None)

    def snapshot(self) -> tuple[RequestRecord, ...]:
        with self._lock:
            return tuple(self._records.values())


@dataclass(frozen=True)
class ActiveSubscriptions:
    contract: QualifiedESContract
    historical_request_id: int
    quote_request_id: int
    generation: int


class SQLiteIBKRSnapshotSink:
    """Mirrors callback snapshots for the existing separate Dash process."""

    def __init__(
        self,
        db_path: str | Path,
        *,
        asset: str = "ES",
        minimum_write_interval_seconds: float = 1.0,
    ) -> None:
        self._store = SQLiteLiveDataStore(db_path)
        self._asset = str(asset).upper()
        self._minimum_interval = max(0.0, float(minimum_write_interval_seconds))
        self._lock = RLock()
        self._last_write_monotonic = 0.0
        self._last_state_signature: tuple[Any, ...] | None = None

    def __call__(self, snapshot: dict[str, Any]) -> None:
        status = dict(snapshot.get("status") or {})
        signature = (
            status.get("connection_state"),
            status.get("active_conId"),
            status.get("subscription_generation"),
            status.get("last_error_code"),
        )
        now = time.monotonic()
        with self._lock:
            if (
                signature == self._last_state_signature
                and now - self._last_write_monotonic < self._minimum_interval
            ):
                return
            self._store.upsert_runtime_state(
                scope=IBKR_RUNTIME_STATE_SCOPE,
                state_key=self._asset,
                payload=snapshot,
            )
            self._last_write_monotonic = now
            self._last_state_signature = signature

    def should_publish(self, status: dict[str, Any]) -> bool:
        signature = (
            status.get("connection_state"),
            status.get("active_conId"),
            status.get("subscription_generation"),
            status.get("last_error_code"),
        )
        with self._lock:
            return (
                signature != self._last_state_signature
                or time.monotonic() - self._last_write_monotonic
                >= self._minimum_interval
            )

    def close(self) -> None:
        with self._lock:
            self._store.close()


class IBKRMarketDataService:
    """Owns one official TWS API connection and one API network-loop thread."""

    _process_guard = RLock()
    _active_process_service: "IBKRMarketDataService | None" = None

    def __init__(
        self,
        config: IBKRConfig,
        *,
        store: IBKRMarketDataStore | None = None,
        api_factory: Callable[[Any], Any] = IBKRAPIClient,
        snapshot_sink: Callable[[dict[str, Any]], None] | None = None,
        now_provider: Callable[[], datetime] | None = None,
    ) -> None:
        self.config = config
        self.store = store or IBKRMarketDataStore(max_bars=config.store_max_bars)
        if snapshot_sink is not None:
            self.store.add_listener(snapshot_sink)
        self._snapshot_sink = snapshot_sink
        self._api_factory = api_factory
        self._now = now_provider or (lambda: datetime.now(UTC))
        self._request_ids = RequestIdAllocator()
        self._lifecycle_lock = RLock()
        self._callback_lock = RLock()
        self._stop_event = Event()
        self._wake_event = Event()
        self._handshake_event = Event()
        self._ready_event = Event()
        self._connection_failed_event = Event()
        self._startup_failure_event = Event()
        self._commands: Queue[str] = Queue()
        self._worker_thread: Thread | None = None
        self._api: Any | None = None
        self._contract_waiters: dict[int, tuple[Event, list[Any]]] = {}
        self._history_waiters: dict[int, Event] = {}
        self._history_received: set[int] = set()
        self._contracts: tuple[QualifiedESContract, ...] = ()
        self._active_selection: ContractSelection | None = None
        self._subscriptions: ActiveSubscriptions | None = None
        self._last_contract_refresh_at: datetime | None = None
        self._reconnect_count = 0
        self._subscription_generation = 0
        self._requested_reconnect = False
        self._requested_resubscribe = False
        self._fatal_market_data_error = False
        self._delayed_fallback_requested = False
        self._last_stale = False
        self._rolling = False
        self._tearing_down = False
        self.store.update_status(**self._base_status("disabled" if not config.enabled else "stopped"))

    def start(self) -> bool:
        if not self.config.enabled:
            self.store.update_status(**self._base_status("disabled"))
            return False
        with self._lifecycle_lock:
            if self._worker_thread is not None and self._worker_thread.is_alive():
                return False
            # Construct synchronously so a missing official API is immediately
            # actionable instead of being hidden in a daemon-thread traceback.
            self._api = self._api_factory(self)
            with self._process_guard:
                active = self._active_process_service
                if active is not None and active is not self and active.is_running():
                    raise IBKRError(
                        "Another IBKR market-data service is already running in this process."
                    )
                type(self)._active_process_service = self
            self._stop_event.clear()
            self._wake_event.clear()
            self._ready_event.clear()
            self._connection_failed_event.clear()
            self._startup_failure_event.clear()
            self._worker_thread = Thread(
                target=self._supervise,
                name="ibkr-market-data-service",
                daemon=True,
            )
            self.store.update_status(**self._base_status("starting"))
            self._worker_thread.start()
            return True

    def stop(self, *, timeout: float = 10.0) -> None:
        with self._lifecycle_lock:
            thread = self._worker_thread
            if thread is None:
                if not self.config.enabled:
                    self.store.update_status(**self._base_status("disabled"))
                self._teardown_api()
                self._release_process_guard()
                self._close_snapshot_sink()
                return
            self._stop_event.set()
            self._wake_event.set()
        if thread is not current_thread():
            thread.join(timeout=max(0.0, float(timeout)))
        with self._lifecycle_lock:
            self._worker_thread = None
        self._release_process_guard()
        self._close_snapshot_sink()

    def is_running(self) -> bool:
        with self._lifecycle_lock:
            return self._worker_thread is not None and self._worker_thread.is_alive()

    def wait_until_ready(self, timeout: float | None = None) -> bool:
        resolved_timeout = (
            self.config.history_timeout_seconds
            if timeout is None
            else max(0.0, float(timeout))
        )
        deadline = time.monotonic() + resolved_timeout
        while time.monotonic() < deadline:
            if self._ready_event.is_set():
                return True
            if self._startup_failure_event.is_set():
                return False
            self._ready_event.wait(min(0.05, max(0.0, deadline - time.monotonic())))
        return self._ready_event.is_set()

    def get_status(self) -> dict[str, Any]:
        return self.store.get_status()

    def get_active_contract(self) -> QualifiedESContract | None:
        selection = self._active_selection
        return None if selection is None else selection.contract

    def get_quote_snapshot(self):
        return self.store.get_quote()

    def get_bars(self, limit: int | None = None):
        return self.store.get_bars(limit=limit)

    def force_contract_refresh(self) -> None:
        self._commands.put("refresh_contracts")
        self._wake_event.set()

    def evaluate_roll(self, now: datetime | None = None) -> ContractSelection | None:
        with self._callback_lock:
            if not self._contracts:
                self.force_contract_refresh()
                return None
            selection = select_es_contract(
                self._contracts,
                config=self.config,
                now=ensure_utc(now or self._now()),
            )
            active = self.get_active_contract()
            if active is not None and active.conid != selection.contract.conid:
                self._commands.put("evaluate_roll")
                self._wake_event.set()
            return selection

    # Official API callbacks -------------------------------------------------

    def on_next_valid_id(self, order_id: int) -> None:
        self._request_ids.set_floor(int(order_id))
        self._handshake_event.set()
        LOGGER.info("IBKR API handshake completed; next valid ID is %s.", order_id)

    def on_contract_details(self, request_id: int, details: Any) -> None:
        with self._callback_lock:
            waiter = self._contract_waiters.get(int(request_id))
            if waiter is not None:
                waiter[1].append(details)

    def on_contract_details_end(self, request_id: int) -> None:
        with self._callback_lock:
            waiter = self._contract_waiters.get(int(request_id))
            if waiter is not None:
                waiter[0].set()

    def on_historical_data(self, request_id: int, payload: Any, is_update: bool) -> None:
        record = self._request_ids.get(request_id)
        if record is None or record.contract is None:
            return
        try:
            bar = _normalize_historical_bar(
                payload,
                contract=record.contract,
                config=self.config,
                market_data_type=self.get_status().get("market_data_type_received"),
                received_at=self._now(),
            )
        except (TypeError, ValueError) as exc:
            LOGGER.warning("Ignored malformed IBKR historical bar: %s", exc)
            return
        self.store.upsert_bar(bar)
        self._history_received.add(int(request_id))
        self.store.update_status(
            last_bar_at=bar.received_at_utc.isoformat(),
            connection_state=self._healthy_connection_state(),
        )
        self._clear_stale_if_recovered()

    def on_historical_data_end(self, request_id: int, start: str, end: str) -> None:
        del start, end
        record = self._request_ids.get(request_id)
        if record is not None and record.contract is not None:
            self.store.mark_request_history_complete(record.contract.conid)
        with self._callback_lock:
            waiter = self._history_waiters.get(int(request_id))
            if waiter is not None:
                waiter.set()

    def on_quote_field(
        self,
        request_id: int,
        field: str,
        value: float,
        received_at: datetime,
    ) -> None:
        record = self._request_ids.get(request_id)
        if record is None or record.contract is None:
            return
        self.store.update_quote(
            record.contract,
            field=field,
            value=value,
            received_at_utc=received_at,
            market_data_type=self.get_status().get("market_data_type_received"),
        )
        quote = self.store.get_quote()
        if quote is not None and quote.conid == record.contract.conid:
            self.store.update_status(
                last_quote_at=ensure_utc(received_at).isoformat(),
                connection_state=self._healthy_connection_state(),
            )
            self._clear_stale_if_recovered()

    def on_market_data_type(self, request_id: int, market_data_type: int) -> None:
        received = market_data_type_name(market_data_type)
        record = self._request_ids.get(request_id)
        if record is not None and record.contract is not None and received is not None:
            self.store.set_market_data_type(record.contract.conid, received)
        state = (
            "rolling"
            if self._rolling
            else ("delayed" if received in {"delayed", "delayed_frozen"} else "connected")
        )
        self.store.update_status(
            market_data_type_received=received,
            live_or_delayed="delayed" if state == "delayed" else "live",
            connection_state=state,
        )
        LOGGER.info("IBKR market-data type received: %s.", received)

    def on_error(
        self,
        request_id: int,
        error_code: int,
        message: str,
        advanced_reject: str = "",
        error_time: int | None = None,
    ) -> None:
        code = int(error_code)
        if code in _INFORMATIONAL_CODES:
            LOGGER.info("IBKR farm status %s: %s", code, message)
            return
        if code == 10167 and self.config.allow_delayed_fallback:
            LOGGER.info(
                "IBKR confirmed delayed market data for request %s: %s",
                request_id,
                " ".join(str(message).split()),
            )
            record = self._request_ids.get(request_id)
            if record is not None and record.contract is not None:
                self.store.set_market_data_type(record.contract.conid, "delayed")
            self.store.update_status(
                market_data_type_received="delayed",
                live_or_delayed="delayed",
                connection_state="rolling" if self._rolling else "delayed",
            )
            return
        if code == 162 and "query cancelled" in str(message).lower():
            LOGGER.info(
                "IBKR historical request %s cancellation acknowledged.",
                request_id,
            )
            return
        if code == 300 and self._request_ids.get(request_id) is None:
            LOGGER.info(
                "IBKR market-data request %s was already cancelled.",
                request_id,
            )
            return
        self.store.update_status(
            last_error_code=code,
            last_error_message=str(message),
            last_error_at=_format_ibkr_error_time(error_time, fallback=self._now()),
            last_error_details=str(advanced_reject or "") or None,
        )
        if code == 1100:
            LOGGER.warning("IBKR server connectivity lost: %s", message)
            self.store.update_status(connection_state="reconnecting")
            return
        if code == 1101:
            LOGGER.warning("IBKR connectivity restored; subscriptions were lost.")
            self._requested_resubscribe = True
            self._commands.put("resubscribe")
            self._wake_event.set()
            return
        if code == 1102:
            LOGGER.info("IBKR connectivity restored with subscriptions maintained.")
            self.store.update_status(connection_state=self._healthy_connection_state())
            return
        if code in {502, 1300}:
            LOGGER.error(
                "IBKR socket connection error %s: %s",
                code,
                " ".join(str(message).split()),
            )
            self._connection_failed_event.set()
            self._handshake_event.set()
            self._request_reconnect()
            return
        if code in _MARKET_DATA_SUBSCRIPTION_CODES:
            if self.config.allow_delayed_fallback:
                if not self._delayed_fallback_requested:
                    LOGGER.warning(
                        "Live IBKR market data unavailable (code %s); explicitly "
                        "configured delayed fallback is being requested.",
                        code,
                    )
                    self._delayed_fallback_requested = True
                    api = self._api
                    if api is not None:
                        api.request_market_data_type(3)
                    self._requested_resubscribe = True
                    self._commands.put("resubscribe")
                    self._wake_event.set()
                else:
                    LOGGER.info(
                        "Ignored repeated live-market-data error %s while "
                        "delayed fallback is active.",
                        code,
                    )
            else:
                LOGGER.error(
                    "Live IBKR market data unavailable (code %s), and delayed "
                    "fallback is disabled: %s",
                    code,
                    " ".join(str(message).split()),
                )
                self._fatal_market_data_error = True
                self.store.update_status(
                    connection_state="error",
                    live_or_delayed="unavailable",
                )
                self._wake_event.set()
            return
        LOGGER.warning("IBKR error request=%s code=%s: %s", request_id, code, message)

    def on_connection_closed(self) -> None:
        if (
            self._stop_event.is_set()
            or self._tearing_down
            or self._connection_failed_event.is_set()
        ):
            return
        LOGGER.warning("IBKR API connection closed.")
        self.store.update_status(connection_state="disconnected")
        self._request_reconnect()

    # Supervisor -------------------------------------------------------------

    def _supervise(self) -> None:
        backoff = self.config.reconnect_initial_seconds
        try:
            while not self._stop_event.is_set():
                try:
                    if self._api is None:
                        self._api = self._api_factory(self)
                    self._connect_and_subscribe()
                    backoff = self.config.reconnect_initial_seconds
                    self._run_connected_loop()
                    if self._fatal_market_data_error:
                        break
                except IBKRError as exc:
                    LOGGER.error("IBKR service cycle failed: %s", exc)
                    self.store.update_status(
                        connection_state="error",
                        last_error_message=str(exc),
                    )
                    self._startup_failure_event.set()
                except Exception as exc:
                    LOGGER.exception("IBKR service cycle failed: %s", exc)
                    self.store.update_status(
                        connection_state="error",
                        last_error_message=str(exc),
                    )
                    self._startup_failure_event.set()
                if self._stop_event.is_set() or self._fatal_market_data_error:
                    break
                self._cancel_subscriptions()
                self._teardown_api()
                self._reconnect_count += 1
                self.store.update_status(
                    connection_state="reconnecting",
                    reconnect_count=self._reconnect_count,
                )
                LOGGER.info(
                    "IBKR reconnect attempt %s in %.1f seconds.",
                    self._reconnect_count,
                    backoff,
                )
                if self._stop_event.wait(backoff):
                    break
                backoff = min(backoff * 2.0, self.config.reconnect_max_seconds)
                self._api = self._api_factory(self)
        finally:
            self._cancel_subscriptions()
            self._teardown_api()
            self._ready_event.clear()
            final_state = "error" if self._fatal_market_data_error else "stopped"
            self.store.update_status(connection_state=final_state)
            self._release_process_guard()
            LOGGER.info("IBKR market-data service stopped.")

    def _connect_and_subscribe(self) -> None:
        api = self._require_api()
        self._requested_reconnect = False
        self._delayed_fallback_requested = False
        self._handshake_event.clear()
        self._connection_failed_event.clear()
        self.store.update_status(connection_state="connecting")
        LOGGER.info(
            "Connecting to IBKR at %s:%s clientId=%s (%s).",
            self.config.host,
            self.config.port,
            self.config.client_id,
            self.config.account_mode,
        )
        api.connect(
            self.config.host,
            self.config.port,
            self.config.client_id,
        )
        if self._connection_failed_event.is_set():
            raise self._connection_failure_error()
        api.start_network_loop()
        if not self._handshake_event.wait(self.config.handshake_timeout_seconds):
            raise IBKRConnectionError(
                "Timed out waiting for the IBKR API handshake. Confirm TWS/Gateway API access and socket port."
            )
        if self._connection_failed_event.is_set():
            raise self._connection_failure_error()
        api.request_market_data_type(self.config.market_data_type_code)
        self._refresh_contracts()
        selection = select_es_contract(
            self._contracts,
            config=self.config,
            now=self._now(),
        )
        self._switch_contract(selection, reason="startup")
        self._startup_failure_event.clear()
        self._ready_event.set()
        LOGGER.info(
            "IBKR ES feed ready on %s month=%s conId=%s.",
            selection.contract.local_symbol,
            selection.contract.contract_month,
            selection.contract.conid,
        )

    def _run_connected_loop(self) -> None:
        while not self._stop_event.is_set():
            self._drain_commands()
            if self._requested_reconnect:
                self._requested_reconnect = False
                return
            if self._fatal_market_data_error:
                return
            now = ensure_utc(self._now())
            if (
                self._last_contract_refresh_at is None
                or (now - self._last_contract_refresh_at).total_seconds()
                >= self.config.contract_refresh_seconds
            ):
                self._refresh_contracts()
            self._evaluate_and_roll(now)
            self._update_stale_state(now)
            self._wake_event.wait(1.0)
            self._wake_event.clear()

    def _drain_commands(self) -> None:
        while True:
            try:
                command = self._commands.get_nowait()
            except Empty:
                return
            if command == "refresh_contracts":
                self._refresh_contracts()
                self._evaluate_and_roll(self._now())
            elif command == "evaluate_roll":
                self._evaluate_and_roll(self._now())
            elif command == "resubscribe" and self._requested_resubscribe:
                self._reconstruct_subscriptions()

    def _refresh_contracts(self) -> None:
        api = self._require_api()
        request_contract_values: dict[str, Any] = {
            "symbol": self.config.symbol,
            "secType": self.config.security_type,
            "exchange": self.config.exchange,
            "currency": self.config.currency,
            "multiplier": self.config.multiplier,
            "tradingClass": self.config.trading_class,
        }
        if self.config.manual_conid is not None:
            request_contract_values = {
                "conId": int(self.config.manual_conid),
                "secType": "FUT",
                "exchange": self.config.exchange,
                "currency": self.config.currency,
            }
        elif self.config.manual_contract_month is not None:
            request_contract_values["lastTradeDateOrContractMonth"] = (
                self.config.manual_contract_month
            )
        elif self.config.local_symbol:
            request_contract_values["localSymbol"] = self.config.local_symbol

        request_id = self._request_ids.allocate("contract_details")
        event = Event()
        payloads: list[Any] = []
        with self._callback_lock:
            self._contract_waiters[request_id] = (event, payloads)
        try:
            api.request_contract_details(
                request_id,
                api.build_contract(**request_contract_values),
            )
            if not event.wait(self.config.contract_timeout_seconds):
                raise IBKRContractError("Timed out waiting for IBKR ES contract details.")
            contracts = filter_es_contracts(
                payloads,
                config=self.config,
                now=self._now(),
            )
            if not contracts:
                raise IBKRContractError(
                    "IBKR contract discovery returned zero qualifying ES FUT contracts."
                )
            self._contracts = contracts
            self._last_contract_refresh_at = ensure_utc(self._now())
            LOGGER.info(
                "Discovered %s valid ES contracts: %s",
                len(contracts),
                ", ".join(
                    f"{item.local_symbol}/{item.contract_month}/conId={item.conid}"
                    for item in contracts
                ),
            )
        finally:
            with self._callback_lock:
                self._contract_waiters.pop(request_id, None)
            self._request_ids.retire(request_id)

    def _evaluate_and_roll(self, now: datetime) -> None:
        if not self._contracts:
            return
        selection = select_es_contract(
            self._contracts,
            config=self.config,
            now=now,
        )
        active = self.get_active_contract()
        if active is None:
            self._switch_contract(selection, reason="contract_evaluation")
            return
        if active.conid != selection.contract.conid:
            self._switch_contract(selection, reason="cme_calendar_roll")
            return
        self._update_contract_status(selection)

    def _switch_contract(self, selection: ContractSelection, *, reason: str) -> None:
        old_subscriptions = self._subscriptions
        old_contract = (
            None if old_subscriptions is None else old_subscriptions.contract
        )
        if old_contract is not None and old_contract.conid == selection.contract.conid:
            self._active_selection = selection
            self._update_contract_status(selection)
            return
        self._rolling = True
        self.store.update_status(connection_state="rolling")
        LOGGER.info(
            "IBKR contract roll beginning old=%s new=%s reason=%s.",
            None if old_contract is None else old_contract.identity,
            selection.contract.identity,
            reason,
        )
        try:
            new_subscriptions = self._start_subscriptions(selection.contract)
        except Exception:
            self._rolling = False
            raise
        # Only after the new contract has supplied valid history do readers see
        # it as active and the old requests get cancelled.
        self._subscriptions = new_subscriptions
        self._active_selection = selection
        self.store.set_active_contract(selection.contract)
        quote = self.store.get_quote()
        if quote is not None and quote.last_update_timestamp_utc is not None:
            self.store.update_status(
                last_quote_at=quote.last_update_timestamp_utc.isoformat()
            )
        self._subscription_generation = new_subscriptions.generation
        if old_subscriptions is not None:
            self._cancel_subscription_set(old_subscriptions)
            if reason == "cme_calendar_roll":
                effective_at = cme_calendar_roll_at(
                    old_subscriptions.contract.contract_month,
                    roll_time_et=self.config.roll_time_et,
                ).astimezone(UTC)
            else:
                observed = ensure_utc(self._now())
                effective_at = observed.replace(
                    minute=(observed.minute // 5) * 5,
                    second=0,
                    microsecond=0,
                )
            roll_payload = {
                "from_local_symbol": old_subscriptions.contract.local_symbol,
                "from_contract_month": old_subscriptions.contract.contract_month,
                "from_conId": old_subscriptions.contract.conid,
                "to_local_symbol": selection.contract.local_symbol,
                "to_contract_month": selection.contract.contract_month,
                "to_conId": selection.contract.conid,
                "effective_at_utc": effective_at.isoformat(),
                "reason": reason,
            }
            self.store.add_roll_event(roll_payload)
            LOGGER.info("IBKR contract roll completed: %s", roll_payload)
        self._update_contract_status(selection)
        self._rolling = False
        self.store.update_status(
            connection_state=self._healthy_connection_state(),
            subscription_generation=self._subscription_generation,
        )

    def _start_subscriptions(
        self,
        contract: QualifiedESContract,
    ) -> ActiveSubscriptions:
        api = self._require_api()
        generation = self._subscription_generation + 1
        historical_request_id = self._request_ids.allocate(
            "historical_keep_up_to_date",
            contract,
        )
        quote_request_id = self._request_ids.allocate("market_data", contract)
        history_event = Event()
        with self._callback_lock:
            self._history_waiters[historical_request_id] = history_event
        ib_contract = api.build_contract(
            conId=contract.conid,
            secType="FUT",
            exchange=contract.exchange,
            symbol=contract.symbol,
            currency=contract.currency,
            multiplier=contract.multiplier,
            tradingClass=contract.trading_class,
            localSymbol=contract.local_symbol,
        )
        try:
            api.request_historical_data(
                historical_request_id,
                ib_contract,
                duration=self.config.history_duration,
                bar_size=self.config.bar_size,
                what_to_show=self.config.what_to_show,
                use_rth=self.config.use_rth,
                keep_up_to_date=True,
            )
            api.request_market_data(quote_request_id, ib_contract)
            LOGGER.info(
                "IBKR subscriptions started for %s historyReqId=%s quoteReqId=%s generation=%s.",
                contract.local_symbol,
                historical_request_id,
                quote_request_id,
                generation,
            )
            if not history_event.wait(self.config.history_timeout_seconds):
                raise IBKRSubscriptionError(
                    f"Timed out waiting for ES history on {contract.local_symbol}."
                )
            if historical_request_id not in self._history_received:
                raise IBKRSubscriptionError(
                    f"IBKR returned no valid ES bars for {contract.local_symbol}."
                )
            return ActiveSubscriptions(
                contract=contract,
                historical_request_id=historical_request_id,
                quote_request_id=quote_request_id,
                generation=generation,
            )
        except Exception:
            self._request_ids.retire(historical_request_id)
            self._request_ids.retire(quote_request_id)
            try:
                api.cancel_historical_data(historical_request_id)
                api.cancel_market_data(quote_request_id)
            finally:
                self._history_received.discard(historical_request_id)
            raise
        finally:
            with self._callback_lock:
                self._history_waiters.pop(historical_request_id, None)

    def _reconstruct_subscriptions(self) -> None:
        self._requested_resubscribe = False
        selection = self._active_selection
        old = self._subscriptions
        if selection is None:
            self._refresh_contracts()
            selection = select_es_contract(
                self._contracts,
                config=self.config,
                now=self._now(),
            )
        new = self._start_subscriptions(selection.contract)
        self._subscriptions = new
        self._subscription_generation = new.generation
        if old is not None:
            self._cancel_subscription_set(old)
        self.store.set_active_contract(selection.contract)
        self.store.update_status(
            connection_state=self._healthy_connection_state(),
            subscription_generation=self._subscription_generation,
        )
        self._evaluate_and_roll(self._now())
        LOGGER.info(
            "Reconstructed IBKR subscriptions for %s generation=%s.",
            selection.contract.local_symbol,
            self._subscription_generation,
        )

    def _cancel_subscriptions(self) -> None:
        subscriptions = self._subscriptions
        self._subscriptions = None
        if subscriptions is not None:
            self._cancel_subscription_set(subscriptions)

    def _cancel_subscription_set(self, subscriptions: ActiveSubscriptions) -> None:
        self._request_ids.retire(subscriptions.historical_request_id)
        self._request_ids.retire(subscriptions.quote_request_id)
        self._history_received.discard(subscriptions.historical_request_id)
        api = self._api
        if api is not None:
            try:
                api.cancel_historical_data(subscriptions.historical_request_id)
            except Exception:
                LOGGER.debug("IBKR historical cancellation failed.", exc_info=True)
            try:
                api.cancel_market_data(subscriptions.quote_request_id)
            except Exception:
                LOGGER.debug("IBKR quote cancellation failed.", exc_info=True)

    def _update_contract_status(self, selection: ContractSelection) -> None:
        contract = selection.contract
        self.store.update_status(
            active_conId=contract.conid,
            active_local_symbol=contract.local_symbol,
            active_contract_month=contract.contract_month,
            active_expiration=contract.expiration.isoformat(),
            active_exchange=contract.exchange,
            manual_override=selection.manual_override,
            contract_selection_fallback=selection.fallback_reason,
            next_roll_at=cme_calendar_roll_at(
                contract.contract_month,
                roll_time_et=self.config.roll_time_et,
            ).astimezone(UTC).isoformat(),
        )

    def _update_stale_state(self, now: datetime) -> None:
        active_contract = self.get_active_contract()
        reported_open = (
            None
            if active_contract is None
            else is_contract_trading_time(active_contract, now)
        )
        market_open = (
            is_es_globex_open(now) if reported_open is None else reported_open
        )
        if not market_open:
            if self._last_stale:
                self._last_stale = False
                self.store.update_status(
                    connection_state=self._healthy_connection_state(),
                    stale_quote=False,
                    stale_bar=False,
                    market_session_open=False,
                )
            else:
                self.store.update_status(market_session_open=False)
            return
        status = self.store.get_status()
        previously_bar_stale = bool(status.get("stale_bar", False))
        quote_at = _parse_optional_datetime(status.get("last_quote_at"))
        bar_at = _parse_optional_datetime(status.get("last_bar_at"))
        quote_stale = (
            quote_at is None
            or (now - quote_at).total_seconds() > self.config.stale_quote_seconds
        )
        bar_age_seconds = (
            float("inf")
            if bar_at is None
            else (now - bar_at).total_seconds()
        )
        bar_stale = (
            bar_at is None
            or bar_age_seconds > self.config.stale_bar_seconds
        )
        stale = quote_stale or bar_stale
        bar_became_stale = bar_stale and not previously_bar_stale
        if stale != self._last_stale:
            LOGGER.warning(
                "IBKR data %s (quote_stale=%s bar_stale=%s).",
                "became stale" if stale else "recovered",
                quote_stale,
                bar_stale,
            )
        received_market_data_type = status.get("market_data_type_received")
        delayed_refresh_due = (
            received_market_data_type in {"delayed", "delayed_frozen"}
            and bar_age_seconds > self.config.delayed_history_refresh_seconds
        )
        if (
            (delayed_refresh_due or bar_became_stale)
            and received_market_data_type in {"delayed", "delayed_frozen"}
            and self._subscriptions is not None
            and not self._requested_resubscribe
        ):
            LOGGER.warning(
                "IBKR delayed completed-bar refresh interval elapsed; rebuilding subscriptions."
            )
            self._requested_resubscribe = True
            self._commands.put("resubscribe")
            self._wake_event.set()
        self._last_stale = stale
        self.store.update_status(
            connection_state="stale" if stale else self._healthy_connection_state(),
            stale_quote=quote_stale,
            stale_bar=bar_stale,
            market_session_open=True,
        )

    def _clear_stale_if_recovered(self) -> None:
        if not self._last_stale:
            return
        self._update_stale_state(ensure_utc(self._now()))

    def _healthy_connection_state(self) -> str:
        if self._rolling:
            return "rolling"
        received = self.store.get_status().get("market_data_type_received")
        return "delayed" if received in {"delayed", "delayed_frozen"} else "connected"

    def _request_reconnect(self) -> None:
        self._requested_reconnect = True
        self._wake_event.set()

    def _teardown_api(self) -> None:
        api = self._api
        self._api = None
        if api is not None:
            self._tearing_down = True
            try:
                api.stop()
            except Exception:
                LOGGER.debug("IBKR API teardown failed.", exc_info=True)
            finally:
                self._tearing_down = False
        self._handshake_event.clear()

    def _require_api(self):
        if self._api is None:
            raise IBKRConnectionError("IBKR API client is not initialized.")
        return self._api

    def _base_status(self, state: str) -> dict[str, Any]:
        return {
            "enabled": bool(self.config.enabled),
            "connection_state": state,
            "host": self.config.host,
            "port": self.config.port,
            "client_id": self.config.client_id,
            "account_mode": self.config.account_mode,
            "market_data_type_requested": self.config.market_data_type,
            "market_data_type_received": None,
            "live_or_delayed": None,
            "active_conId": None,
            "active_local_symbol": None,
            "active_contract_month": None,
            "active_expiration": None,
            "roll_policy": self.config.roll_policy,
            "next_roll_at": None,
            "last_quote_at": None,
            "last_bar_at": None,
            "last_error_code": None,
            "last_error_message": None,
            "last_error_at": None,
            "last_error_details": None,
            "reconnect_count": self._reconnect_count,
            "subscription_generation": self._subscription_generation,
            "stale_quote": False,
            "stale_bar": False,
        }

    def _close_snapshot_sink(self) -> None:
        close = getattr(self._snapshot_sink, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                LOGGER.debug("IBKR snapshot sink close failed.", exc_info=True)

    def _connection_failure_error(self) -> IBKRConnectionError:
        status = self.store.get_status()
        code = status.get("last_error_code")
        if code == 502:
            return IBKRConnectionError(
                f"IBKR API socket connection failed (code 502): no listener is "
                f"reachable at {self.config.host}:{self.config.port}. In TWS, "
                'enable "ActiveX and Socket Clients" and set the Socket Port '
                f"to {self.config.port}."
            )
        message = " ".join(str(status.get("last_error_message") or "").split())
        return IBKRConnectionError(
            f"IBKR API socket connection failed (code {code}): {message} "
            f"Confirm that TWS/Gateway is listening at "
            f"{self.config.host}:{self.config.port}."
        )

    def _release_process_guard(self) -> None:
        with self._process_guard:
            if self._active_process_service is self:
                type(self)._active_process_service = None


def _format_ibkr_error_time(
    value: int | None,
    *,
    fallback: datetime,
) -> str:
    """Convert IB API error times, which may be seconds or milliseconds."""

    if value is None or value <= 0:
        return ensure_utc(fallback).isoformat()
    timestamp = float(value)
    if timestamp >= 100_000_000_000_000:
        timestamp /= 1_000_000.0
    elif timestamp >= 100_000_000_000:
        timestamp /= 1_000.0
    try:
        return datetime.fromtimestamp(timestamp, tz=UTC).isoformat()
    except (OSError, OverflowError, ValueError):
        return ensure_utc(fallback).isoformat()


def _normalize_historical_bar(
    payload: Any,
    *,
    contract: QualifiedESContract,
    config: IBKRConfig,
    market_data_type: str | None,
    received_at: datetime,
) -> IBKRBar:
    return IBKRBar(
        timestamp_utc=_coerce_ibkr_timestamp(getattr(payload, "date")),
        open=float(getattr(payload, "open")),
        high=float(getattr(payload, "high")),
        low=float(getattr(payload, "low")),
        close=float(getattr(payload, "close")),
        volume=float(getattr(payload, "volume", 0.0) or 0.0),
        wap=_optional_float(getattr(payload, "wap", None)),
        trade_count=_optional_int(
            getattr(payload, "barCount", getattr(payload, "count", None))
        ),
        conid=contract.conid,
        local_symbol=contract.local_symbol,
        contract_month=contract.contract_month,
        bar_size=config.bar_size,
        data_type=config.what_to_show,
        is_complete=False,
        source="ibkr.historical.update",
        market_data_type=market_data_type,
        received_at_utc=ensure_utc(received_at),
    )


def _coerce_ibkr_timestamp(value: Any) -> datetime:
    if isinstance(value, datetime):
        return ensure_utc(value)
    text = str(value).strip()
    if not text:
        raise ValueError("IBKR bar timestamp was empty.")
    if text.isdigit():
        if len(text) == 8:
            return datetime.strptime(text, "%Y%m%d").replace(tzinfo=UTC)
        return datetime.fromtimestamp(int(text), tz=UTC)
    normalized = text.replace(" UTC", "+00:00")
    parsed = datetime.fromisoformat(normalized)
    return ensure_utc(parsed)


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _parse_optional_datetime(value: Any) -> datetime | None:
    if value in {None, ""}:
        return None
    if isinstance(value, datetime):
        return ensure_utc(value)
    return ensure_utc(datetime.fromisoformat(str(value)))
