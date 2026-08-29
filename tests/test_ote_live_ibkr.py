from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
import sys
import threading
import time

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ote_live.ingestion.ibkr import (  # noqa: E402
    IBAPIUnavailableError,
    IBKRBar,
    IBKRConfig,
    IBKRError,
    IBKRMarketDataService,
    IBKRMarketDataStore,
    QualifiedESContract,
    RequestIdAllocator,
    cme_calendar_roll_at,
    cme_calendar_roll_date,
    expected_quarterly_contract_month,
    filter_es_contracts,
    is_contract_trading_time,
    select_es_contract,
    third_friday,
)
from ote_live.ingestion.ibkr.client import (  # noqa: E402
    _TICK_PRICE_FIELDS,
    _TICK_SIZE_FIELDS,
    _parse_error_callback_args,
    load_official_ibapi,
)
from ote_live.dashboard.app import (  # noqa: E402
    _format_ibkr_feed_status,
    _merge_ibkr_live_bars,
)
from ote_live.dashboard.charts import build_price_signal_figure  # noqa: E402
from ote_live.contracts.market_data import MarketBar  # noqa: E402
from ote_live.ingestion.connector_ibkr import (  # noqa: E402
    _filter_bars_to_effective_roll_segments,
)


@pytest.mark.parametrize(
    ("month", "expected_friday", "expected_monday"),
    [
        (3, date(2026, 3, 20), date(2026, 3, 16)),
        (6, date(2026, 6, 19), date(2026, 6, 15)),
        (9, date(2026, 9, 18), date(2026, 9, 14)),
        (12, date(2026, 12, 18), date(2026, 12, 14)),
    ],
)
def test_cme_quarterly_roll_calendar(
    month: int,
    expected_friday: date,
    expected_monday: date,
) -> None:
    assert third_friday(2026, month) == expected_friday
    assert cme_calendar_roll_date(2026, month) == expected_monday


def test_contract_month_switches_at_exact_roll_timestamp_and_across_year() -> None:
    march_roll = cme_calendar_roll_at("202603").astimezone(UTC)
    assert expected_quarterly_contract_month(
        march_roll - timedelta(microseconds=1)
    ) == "202603"
    assert expected_quarterly_contract_month(march_roll) == "202606"
    assert expected_quarterly_contract_month(
        march_roll + timedelta(seconds=1)
    ) == "202606"

    december_roll = cme_calendar_roll_at("202612").astimezone(UTC)
    assert expected_quarterly_contract_month(
        december_roll - timedelta(seconds=1)
    ) == "202612"
    assert expected_quarterly_contract_month(december_roll) == "202703"


def test_roll_time_preserves_new_york_wall_clock_across_dst() -> None:
    march = cme_calendar_roll_at("202603").astimezone(UTC)
    december = cme_calendar_roll_at("202612").astimezone(UTC)
    assert (march.hour, march.minute) == (13, 30)
    assert (december.hour, december.minute) == (14, 30)


def test_ibkr_config_parses_documented_environment_contract() -> None:
    config = IBKRConfig.from_env(
        {
            "IBKR_ENABLED": "true",
            "IBKR_HOST": "localhost",
            "IBKR_PORT": "4002",
            "IBKR_CLIENT_ID": "21",
            "IBKR_ACCOUNT_MODE": "paper",
            "IBKR_MARKET_DATA_TYPE": "live",
            "IBKR_ALLOW_DELAYED_FALLBACK": "false",
            "IBKR_ES_BAR_SIZE": "5 mins",
            "IBKR_ES_HISTORY_DURATION": "2 D",
            "IBKR_ES_USE_RTH": "false",
            "IBKR_ES_ROLL_TIME_ET": "09:30",
            "IBKR_ES_MANUAL_CONID": "",
            "IBKR_ES_MANUAL_CONTRACT_MONTH": "",
            "IBKR_STALE_QUOTE_SECONDS": "15",
            "IBKR_STALE_BAR_SECONDS": "420",
            "IBKR_RECONNECT_INITIAL_SECONDS": "2",
            "IBKR_RECONNECT_MAX_SECONDS": "60",
        }
    )
    assert config.enabled is True
    assert config.port == 4002
    assert config.market_data_type_code == 1
    assert config.manual_conid is None
    assert config.manual_contract_month is None


def test_ibapi_error_callback_parser_supports_old_and_timestamped_signatures() -> None:
    assert _parse_error_callback_args((502, "Couldn't connect", "")) == (
        None,
        502,
        "Couldn't connect",
        "",
    )
    assert _parse_error_callback_args(
        (1_785_372_083_261, 502, "Couldn't connect", "")
    ) == (
        1_785_372_083_261,
        502,
        "Couldn't connect",
        "",
    )


def test_ibapi_delayed_ticks_map_to_the_same_quote_fields_as_live_ticks() -> None:
    assert {
        tick_type: _TICK_PRICE_FIELDS[tick_type]
        for tick_type in (66, 67, 68, 72, 73, 75)
    } == {
        66: "bid",
        67: "ask",
        68: "last",
        72: "session_high",
        73: "session_low",
        75: "previous_close",
    }
    assert {
        tick_type: _TICK_SIZE_FIELDS[tick_type]
        for tick_type in (69, 70, 71, 74)
    } == {
        69: "bid_size",
        70: "ask_size",
        71: "last_size",
        74: "reported_volume",
    }


def test_contract_selection_manual_override_precedes_calendar() -> None:
    config = IBKRConfig(
        enabled=True,
        manual_contract_month="202606",
    )
    selection = select_es_contract(
        (_contract("202603", 101), _contract("202606", 102)),
        config=config,
        now=datetime(2026, 2, 2, tzinfo=UTC),
    )
    assert selection.contract.conid == 102
    assert selection.manual_override == "contract_month:202606"


def test_contract_selection_rejects_expired_manual_contract() -> None:
    config = IBKRConfig(enabled=True, manual_conid=100)
    with pytest.raises(Exception, match="did not resolve"):
        select_es_contract(
            (
                QualifiedESContract(
                    conid=100,
                    local_symbol="ESZ5",
                    contract_month="202512",
                    expiration=date(2025, 12, 19),
                ),
                _contract("202603", 101),
            ),
            config=config,
            now=datetime(2026, 2, 2, tzinfo=UTC),
        )


def test_contract_selection_falls_back_to_nearest_valid_quarter() -> None:
    selection = select_es_contract(
        (_contract("202606", 102), _contract("202609", 103)),
        config=IBKRConfig(enabled=True),
        now=datetime(2026, 2, 2, tzinfo=UTC),
    )
    assert selection.contract.contract_month == "202606"
    assert "Expected ES contract month 202603" in str(selection.fallback_reason)


def test_contract_filter_rejects_mes_non_quarterly_expired_and_malformed() -> None:
    config = IBKRConfig(enabled=True)
    now = datetime(2026, 2, 2, tzinfo=UTC)
    valid = _details("ES", "FUT", "20260320", 101, "ESH6")
    duplicate = _details("ES", "FUT", "20260320", 101, "ESH6")
    payloads = [
        valid,
        duplicate,
        _details("MES", "FUT", "20260320", 201, "MESH6", multiplier="5"),
        _details("ES", "FUT", "20260417", 202, "ESJ6"),
        _details("ES", "FUT", "20251219", 203, "ESZ5"),
        _details("ES", "IND", "20260320", 204, "SPX"),
        _details("ES", "FUT", "malformed", 205, "BAD"),
    ]
    contracts = filter_es_contracts(payloads, config=config, now=now)
    assert [(item.conid, item.local_symbol) for item in contracts] == [(101, "ESH6")]


def test_contract_filter_and_selector_handle_no_qualifying_contracts() -> None:
    config = IBKRConfig(enabled=True)
    assert (
        filter_es_contracts(
            [_details("MES", "FUT", "20260320", 1, "MESH6", multiplier="5")],
            config=config,
            now=datetime(2026, 1, 1, tzinfo=UTC),
        )
        == ()
    )
    with pytest.raises(Exception, match="no valid"):
        select_es_contract(
            (),
            config=config,
            now=datetime(2026, 1, 1, tzinfo=UTC),
        )


def test_reported_contract_hours_are_used_when_available() -> None:
    contract = QualifiedESContract(
        conid=101,
        local_symbol="ESH6",
        contract_month="202603",
        expiration=date(2026, 3, 20),
        timezone="America/Chicago",
        trading_hours=(
            "20260201:1700-20260202:1600;"
            "20260202:1700-20260203:1600"
        ),
    )
    assert (
        is_contract_trading_time(
            contract,
            datetime(2026, 2, 2, 15, 0, tzinfo=UTC),
        )
        is True
    )
    assert (
        is_contract_trading_time(
            contract,
            datetime(2026, 2, 2, 22, 30, tzinfo=UTC),
        )
        is False
    )


def test_bar_store_upserts_active_bar_and_completes_prior_bucket() -> None:
    store = IBKRMarketDataStore(max_bars=10)
    first = _bar(datetime(2026, 2, 2, 14, 30, tzinfo=UTC), close=6000.0)
    store.upsert_bar(first)
    store.upsert_bar(_bar(first.timestamp_utc, close=6001.0))
    assert len(store.get_bars()) == 1
    assert store.get_bars()[0].close == 6001.0
    assert store.get_bars()[0].is_complete is False

    store.upsert_bar(
        _bar(datetime(2026, 2, 2, 14, 35, tzinfo=UTC), close=6002.0)
    )
    bars = store.get_bars()
    assert len(bars) == 2
    assert bars[0].is_complete is True
    assert bars[1].is_complete is False


def test_bar_store_preserves_same_timestamp_for_different_contracts_and_is_bounded() -> None:
    store = IBKRMarketDataStore(max_bars=3)
    timestamp = datetime(2026, 3, 16, 13, 30, tzinfo=UTC)
    store.upsert_bar(_bar(timestamp, close=6000.0, conid=101, symbol="ESH6"))
    store.upsert_bar(_bar(timestamp, close=6008.0, conid=102, symbol="ESM6"))
    store.upsert_bar(_bar(timestamp + timedelta(minutes=5), close=6009.0, conid=102, symbol="ESM6"))
    store.upsert_bar(_bar(timestamp + timedelta(minutes=10), close=6010.0, conid=102, symbol="ESM6"))
    bars = store.get_bars()
    assert len(bars) == 3
    assert len({bar.key for bar in bars}) == 3
    assert any(bar.conid == 101 for bar in bars) is False


def test_bar_store_snapshot_reads_are_thread_safe() -> None:
    store = IBKRMarketDataStore(max_bars=200)
    errors: list[BaseException] = []

    def _write() -> None:
        try:
            start = datetime(2026, 2, 2, 14, 30, tzinfo=UTC)
            for index in range(100):
                store.upsert_bar(
                    _bar(start + timedelta(minutes=5 * index), close=6000 + index)
                )
        except BaseException as exc:  # pragma: no cover - assertion aid
            errors.append(exc)

    def _read() -> None:
        try:
            for _ in range(100):
                store.snapshot()
        except BaseException as exc:  # pragma: no cover - assertion aid
            errors.append(exc)

    threads = [threading.Thread(target=_write), threading.Thread(target=_read)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)
    assert errors == []


def test_request_ids_are_unique_and_retired_cleanly() -> None:
    allocator = RequestIdAllocator(start=50)
    ids = {allocator.allocate("market_data") for _ in range(500)}
    assert len(ids) == 500
    allocator.retire(50)
    assert allocator.get(50) is None
    allocator.set_floor(10_000)
    assert allocator.allocate("history") == 10_000


def test_missing_official_api_has_actionable_message(monkeypatch) -> None:
    import builtins

    original_import = builtins.__import__

    def _blocked_import(name, *args, **kwargs):
        if name.startswith("ibapi"):
            raise ImportError("blocked for test")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _blocked_import)
    with pytest.raises(IBAPIUnavailableError, match="TWS API"):
        load_official_ibapi()


def test_dashboard_merges_partial_bar_without_erasing_other_roll_contract() -> None:
    timestamp = datetime(2026, 3, 16, 13, 30, tzinfo=UTC)
    persisted = pd.DataFrame(
        [
            {
                "asset": "ES",
                "timeframe": "5m",
                "timestamp": timestamp,
                "open": 6000,
                "high": 6002,
                "low": 5999,
                "close": 6001,
                "volume": 100,
                "contract_symbol": "ESH6",
                "instrument_id": 101,
            }
        ]
    )
    state = {
        "bars": [
            {
                "timestamp_utc": timestamp.isoformat(),
                "open": 6008,
                "high": 6010,
                "low": 6007,
                "close": 6009,
                "volume": 80,
                "conid": 102,
                "local_symbol": "ESM6",
                "is_complete": False,
            }
        ]
    }
    merged = _merge_ibkr_live_bars(persisted, state)
    assert len(merged) == 2
    assert set(merged["instrument_id"]) == {101, 102}

    figure = build_price_signal_figure(merged, pd.DataFrame())
    assert figure.layout.uirevision == "ote-live-price"
    assert len(figure.layout.shapes) == 1


def test_inference_roll_filter_excludes_overlapping_backfill_contract_bars() -> None:
    effective = datetime(2026, 3, 16, 13, 30, tzinfo=UTC)
    old_before = _market_bar(effective - timedelta(minutes=5), conid=101, symbol="ESH6")
    old_after = _market_bar(effective, conid=101, symbol="ESH6")
    new_before = _market_bar(effective - timedelta(minutes=5), conid=102, symbol="ESM6")
    new_after = _market_bar(effective, conid=102, symbol="ESM6")
    filtered = _filter_bars_to_effective_roll_segments(
        [old_before, old_after, new_before, new_after],
        [
            {
                "from_conId": 101,
                "to_conId": 102,
                "effective_at_utc": effective.isoformat(),
            }
        ],
    )
    assert [(bar.instrument_id, bar.timestamp) for bar in filtered] == [
        (101, old_before.timestamp),
        (102, new_after.timestamp),
    ]


def test_dashboard_feed_status_includes_quote_roll_and_stale_state() -> None:
    text = _format_ibkr_feed_status(
        {
            "status": {
                "connection_state": "stale",
                "active_local_symbol": "ESM6",
                "active_expiration": "2026-06-19",
                "live_or_delayed": "live",
                "next_roll_at": "2026-06-15T13:30:00+00:00",
                "stale_quote": True,
            },
            "quote": {
                "bid": 6000.0,
                "ask": 6000.25,
                "last": 6000.0,
                "last_update_timestamp_utc": "2026-03-16T13:30:01+00:00",
            },
        },
        expected=True,
    )
    assert "ESM6" in text
    assert "spread 0.25" in text
    assert "STALE" in text
    assert "EDT" in text


def test_service_reconstructs_1101_but_not_1102_and_start_is_idempotent() -> None:
    now = datetime(2026, 2, 2, 14, 0, tzinfo=UTC)
    created: list[_FakeAPI] = []

    def _factory(target):
        api = _FakeAPI(target)
        created.append(api)
        return api

    service = IBKRMarketDataService(
        IBKRConfig(
            enabled=True,
            handshake_timeout_seconds=1,
            contract_timeout_seconds=1,
            history_timeout_seconds=1,
        ),
        api_factory=_factory,
        now_provider=lambda: now,
    )
    try:
        assert service.start() is True
        assert service.start() is False
        assert service.wait_until_ready(3)
        worker = service._worker_thread
        initial_generation = service.get_status()["subscription_generation"]

        service.on_error(-1, 1101, "subscriptions lost")
        _wait_for(
            lambda: service.get_status()["subscription_generation"]
            > initial_generation
        )
        rebuilt_generation = service.get_status()["subscription_generation"]
        assert created[0].history_request_count == 2

        service.on_error(-1, 1102, "data maintained")
        time.sleep(0.05)
        assert service.get_status()["subscription_generation"] == rebuilt_generation
        assert service._worker_thread is worker
        assert worker is not None and worker.is_alive()
    finally:
        service.stop()


def test_delayed_stale_bar_stream_queues_one_subscription_rebuild() -> None:
    now = datetime(2026, 7, 30, 14, 0, tzinfo=UTC)
    service = IBKRMarketDataService(
        IBKRConfig(
            enabled=True,
            stale_quote_seconds=15,
            stale_bar_seconds=420,
        ),
        now_provider=lambda: now,
    )
    service._subscriptions = SimpleNamespace()
    service.store.update_status(
        market_data_type_received="delayed",
        last_quote_at=now.isoformat(),
        last_bar_at=(now - timedelta(minutes=8)).isoformat(),
    )

    service._update_stale_state(now)
    service._update_stale_state(now)

    assert service._requested_resubscribe is True
    assert service._commands.qsize() == 1
    assert service._commands.get_nowait() == "resubscribe"


def test_delayed_bar_stale_transition_rebuilds_when_quote_was_already_stale() -> None:
    now = datetime(2026, 7, 30, 14, 0, tzinfo=UTC)
    service = IBKRMarketDataService(
        IBKRConfig(
            enabled=True,
            stale_quote_seconds=15,
            stale_bar_seconds=420,
        ),
        now_provider=lambda: now,
    )
    service._subscriptions = SimpleNamespace()
    service._last_stale = True
    service.store.update_status(
        market_data_type_received="delayed",
        stale_quote=True,
        stale_bar=False,
        last_quote_at=now.isoformat(),
        last_bar_at=(now - timedelta(minutes=8)).isoformat(),
    )

    service._update_stale_state(now)
    service._update_stale_state(now)

    assert service._requested_resubscribe is True
    assert service._commands.qsize() == 1
    assert service._commands.get_nowait() == "resubscribe"


def test_delayed_history_refreshes_before_bar_is_declared_stale() -> None:
    now = datetime(2026, 7, 30, 14, 0, tzinfo=UTC)
    service = IBKRMarketDataService(
        IBKRConfig(
            enabled=True,
            stale_quote_seconds=15,
            stale_bar_seconds=420,
            delayed_history_refresh_seconds=240,
        ),
        now_provider=lambda: now,
    )
    service._subscriptions = SimpleNamespace()
    service.store.update_status(
        market_data_type_received="delayed",
        stale_bar=False,
        last_quote_at=now.isoformat(),
        last_bar_at=(now - timedelta(minutes=5)).isoformat(),
    )

    service._update_stale_state(now)
    service._update_stale_state(now)

    assert service.get_status()["stale_bar"] is False
    assert service._requested_resubscribe is True
    assert service._commands.qsize() == 1
    assert service._commands.get_nowait() == "resubscribe"


def test_socket_reconnect_rebuilds_subscriptions_without_duplicate_worker() -> None:
    now = datetime(2026, 2, 2, 14, 0, tzinfo=UTC)
    created: list[_FakeAPI] = []

    def _factory(target):
        api = _FakeAPI(target)
        created.append(api)
        return api

    service = IBKRMarketDataService(
        IBKRConfig(
            enabled=True,
            reconnect_initial_seconds=0.01,
            reconnect_max_seconds=0.02,
            handshake_timeout_seconds=1,
            contract_timeout_seconds=1,
            history_timeout_seconds=1,
        ),
        api_factory=_factory,
        now_provider=lambda: now,
    )
    try:
        service.start()
        assert service.wait_until_ready(3)
        worker = service._worker_thread
        initial_generation = service.get_status()["subscription_generation"]

        service.on_connection_closed()
        service.on_connection_closed()
        _wait_for(lambda: len(created) >= 2)
        _wait_for(
            lambda: service.get_status()["subscription_generation"]
            > initial_generation
        )
        assert service._worker_thread is worker
        assert worker is not None and worker.is_alive()
        assert created[1].history_request_count == 1
    finally:
        service.stop()


def test_socket_502_fails_startup_immediately_with_actionable_status() -> None:
    error_time_ms = 1_785_372_083_261

    class _FailingAPI:
        def __init__(self, target) -> None:
            self.target = target

        def connect(self, host, port, client_id) -> bool:
            del host, port, client_id
            self.target.on_error(
                -1,
                502,
                "Couldn't connect to TWS",
                "",
                error_time_ms,
            )
            return False

        def stop(self) -> None:
            return None

    service = IBKRMarketDataService(
        IBKRConfig(
            enabled=True,
            reconnect_initial_seconds=10,
            reconnect_max_seconds=10,
            handshake_timeout_seconds=10,
            contract_timeout_seconds=10,
            history_timeout_seconds=10,
        ),
        api_factory=lambda target: _FailingAPI(target),
    )
    started_at = time.monotonic()
    try:
        service.start()
        assert service.wait_until_ready(1) is False
        assert time.monotonic() - started_at < 0.5
        status = service.get_status()
        assert status["connection_state"] == "reconnecting"
        assert status["last_error_code"] == 502
        assert status["last_error_at"] == "2026-07-30T00:41:23.261000+00:00"
        assert "127.0.0.1:4001" in status["last_error_message"]
    finally:
        service.stop()


@pytest.mark.parametrize("error_code", [354, 10089, 10168])
def test_live_market_data_subscription_error_is_fatal_without_delayed_fallback(
    error_code: int,
) -> None:
    service = IBKRMarketDataService(
        IBKRConfig(
            enabled=True,
            market_data_type="live",
            allow_delayed_fallback=False,
        )
    )

    service.on_error(
        3,
        error_code,
        "Requested market data is not subscribed. Delayed market data is available.",
    )

    status = service.get_status()
    assert status["connection_state"] == "error"
    assert status["live_or_delayed"] == "unavailable"
    assert status["last_error_code"] == error_code


@pytest.mark.parametrize("initial_error_code", [354, 10089, 10168])
def test_delayed_fallback_is_requested_once_and_10167_confirms_it(
    initial_error_code: int,
) -> None:
    requested_types: list[int] = []
    service = IBKRMarketDataService(
        IBKRConfig(
            enabled=True,
            market_data_type="live",
            allow_delayed_fallback=True,
        )
    )
    service._api = SimpleNamespace(
        request_market_data_type=lambda value: requested_types.append(value)
    )

    service.on_error(3, initial_error_code, "Delayed market data is available.")
    queued_after_first_request = service._commands.qsize()
    service.on_error(3, initial_error_code, "Delayed market data is available.")

    assert requested_types == [3]
    assert service._commands.qsize() == queued_after_first_request

    service.on_error(4, 10167, "Displaying delayed market data.")
    status = service.get_status()
    assert status["connection_state"] == "delayed"
    assert status["market_data_type_received"] == "delayed"
    assert status["live_or_delayed"] == "delayed"
    assert service._commands.qsize() == queued_after_first_request


@pytest.mark.parametrize(
    ("code", "message"),
    [
        (162, "Historical data query cancelled: 2"),
        (300, "Can't find EId with tickerId:3"),
    ],
)
def test_expected_subscription_cancellation_messages_do_not_pollute_status(
    code: int,
    message: str,
) -> None:
    service = IBKRMarketDataService(IBKRConfig(enabled=True))
    service.on_error(3, code, message)
    assert service.get_status()["last_error_code"] is None


def test_process_guard_rejects_second_live_service() -> None:
    config = IBKRConfig(
        enabled=True,
        handshake_timeout_seconds=1,
        contract_timeout_seconds=1,
        history_timeout_seconds=1,
    )
    first = IBKRMarketDataService(
        config,
        api_factory=lambda target: _FakeAPI(target),
        now_provider=lambda: datetime(2026, 2, 2, tzinfo=UTC),
    )
    second = IBKRMarketDataService(
        config,
        api_factory=lambda target: _FakeAPI(target),
        now_provider=lambda: datetime(2026, 2, 2, tzinfo=UTC),
    )
    try:
        first.start()
        assert first.wait_until_ready(3)
        with pytest.raises(IBKRError, match="already running"):
            second.start()
    finally:
        second.stop()
        first.stop()


def test_service_rolls_after_boundary_and_preserves_roll_metadata() -> None:
    now_box = [datetime(2026, 3, 16, 13, 29, 59, tzinfo=UTC)]

    service = IBKRMarketDataService(
        IBKRConfig(
            enabled=True,
            handshake_timeout_seconds=1,
            contract_timeout_seconds=1,
            history_timeout_seconds=1,
        ),
        api_factory=lambda target: _FakeAPI(target),
        now_provider=lambda: now_box[0],
    )
    try:
        service.start()
        assert service.wait_until_ready(3)
        assert service.get_active_contract().contract_month == "202603"
        now_box[0] = datetime(2026, 3, 16, 13, 30, tzinfo=UTC)
        service.evaluate_roll(now_box[0])
        _wait_for(
            lambda: service.get_active_contract() is not None
            and service.get_active_contract().contract_month == "202606"
        )
        snapshot = service.store.snapshot()
        assert snapshot["roll_events"][-1]["from_conId"] == 101
        assert snapshot["roll_events"][-1]["to_conId"] == 102
        assert len({bar["conid"] for bar in snapshot["bars"]}) == 2
    finally:
        service.stop()


def _contract(month: str, conid: int) -> QualifiedESContract:
    expiration_day = {
        "202603": date(2026, 3, 20),
        "202606": date(2026, 6, 19),
        "202609": date(2026, 9, 18),
        "202612": date(2026, 12, 18),
    }.get(month, date(int(month[:4]), int(month[4:6]), 19))
    code = {3: "H", 6: "M", 9: "U", 12: "Z"}[int(month[4:6])]
    return QualifiedESContract(
        conid=conid,
        local_symbol=f"ES{code}{month[3]}",
        contract_month=month,
        expiration=expiration_day,
    )


def _details(
    symbol: str,
    security_type: str,
    expiration: str,
    conid: int,
    local_symbol: str,
    *,
    multiplier: str = "50",
):
    return SimpleNamespace(
        contract=SimpleNamespace(
            symbol=symbol,
            secType=security_type,
            exchange="CME",
            currency="USD",
            multiplier=multiplier,
            tradingClass=symbol,
            conId=conid,
            localSymbol=local_symbol,
            lastTradeDateOrContractMonth=expiration,
        ),
        realExpirationDate=expiration,
        timeZoneId="US/Central",
        tradingHours="20260202:1700-20260203:1600",
        liquidHours="20260203:0830-20260203:1500",
    )


def _bar(
    timestamp: datetime,
    *,
    close: float,
    conid: int = 101,
    symbol: str = "ESH6",
) -> IBKRBar:
    return IBKRBar(
        timestamp_utc=timestamp,
        open=close - 1,
        high=close + 1,
        low=close - 2,
        close=close,
        volume=100,
        wap=close - 0.25,
        trade_count=25,
        conid=conid,
        local_symbol=symbol,
        contract_month="202603" if conid == 101 else "202606",
        bar_size="5 mins",
        received_at_utc=timestamp + timedelta(seconds=1),
    )


def _market_bar(timestamp: datetime, *, conid: int, symbol: str) -> MarketBar:
    return MarketBar(
        asset="ES",
        timeframe="5m",
        timestamp=timestamp,
        open=6000,
        high=6002,
        low=5999,
        close=6001,
        volume=100,
        source="ibkr.historical.update",
        symbol="ES",
        contract_symbol=symbol,
        instrument_id=conid,
    )


class _FakeAPI:
    def __init__(self, target) -> None:
        self.target = target
        self.connected = False
        self.history_request_count = 0
        self.cancelled_history: list[int] = []
        self.cancelled_quotes: list[int] = []

    def connect(self, host, port, client_id) -> None:
        del host, port, client_id
        self.connected = True

    def start_network_loop(self):
        self.target.on_next_valid_id(1_000)
        return threading.current_thread()

    def stop(self) -> None:
        self.connected = False
        self.target.on_connection_closed()

    def is_connected(self) -> bool:
        return self.connected

    def build_contract(self, **values):
        return SimpleNamespace(**values)

    def request_market_data_type(self, market_data_type: int) -> None:
        del market_data_type

    def request_contract_details(self, request_id: int, contract) -> None:
        del contract
        for details in (
            _details("ES", "FUT", "20260320", 101, "ESH6"),
            _details("ES", "FUT", "20260619", 102, "ESM6"),
        ):
            self.target.on_contract_details(request_id, details)
        self.target.on_contract_details_end(request_id)

    def request_historical_data(self, request_id: int, contract, **kwargs) -> None:
        del kwargs
        self.history_request_count += 1
        base = int(datetime(2026, 3, 16, 13, 25, tzinfo=UTC).timestamp())
        payload = SimpleNamespace(
            date=base,
            open=6000 + int(contract.conId == 102) * 8,
            high=6002 + int(contract.conId == 102) * 8,
            low=5999 + int(contract.conId == 102) * 8,
            close=6001 + int(contract.conId == 102) * 8,
            volume=100,
            wap=6000.5,
            barCount=20,
        )
        self.target.on_historical_data(request_id, payload, False)
        self.target.on_historical_data_end(request_id, "", "")

    def cancel_historical_data(self, request_id: int) -> None:
        self.cancelled_history.append(request_id)

    def request_market_data(self, request_id: int, contract) -> None:
        del contract
        self.target.on_market_data_type(request_id, 1)
        self.target.on_quote_field(
            request_id,
            "bid",
            6000.0,
            datetime(2026, 3, 16, 13, 25, 1, tzinfo=UTC),
        )

    def cancel_market_data(self, request_id: int) -> None:
        self.cancelled_quotes.append(request_id)


def _wait_for(predicate, *, timeout: float = 3.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("Condition was not met before timeout.")
