from __future__ import annotations

from datetime import UTC, datetime
import logging
from threading import Event, RLock, Thread, current_thread
from typing import Any

from .errors import IBAPIUnavailableError

LOGGER = logging.getLogger(__name__)

_TICK_PRICE_FIELDS = {
    1: "bid",
    2: "ask",
    4: "last",
    6: "session_high",
    7: "session_low",
    9: "previous_close",
    66: "bid",
    67: "ask",
    68: "last",
    72: "session_high",
    73: "session_low",
    75: "previous_close",
}
_TICK_SIZE_FIELDS = {
    0: "bid_size",
    3: "ask_size",
    5: "last_size",
    8: "reported_volume",
    69: "bid_size",
    70: "ask_size",
    71: "last_size",
    74: "reported_volume",
}


def load_official_ibapi():
    try:
        from ibapi.client import EClient
        from ibapi.contract import Contract
        from ibapi.wrapper import EWrapper
    except ImportError as exc:  # pragma: no cover - depends on workstation install
        raise IBAPIUnavailableError(
            "IBKR market data is enabled, but the official 'ibapi' Python client is not "
            "installed. Install it from the Interactive Brokers TWS API source package, "
            r"typically: cd 'C:\TWS API\source\pythonclient'; python -m pip install ."
        ) from exc
    return EWrapper, EClient, Contract


class IBKRAPIClient:
    """Small official-API adapter that forwards callbacks to the managed service."""

    def __init__(self, callback_target: Any) -> None:
        EWrapper, EClient, Contract = load_official_ibapi()
        self._contract_class = Contract
        self._callback_target = callback_target
        self._loop_lock = RLock()
        self._loop_thread: Thread | None = None
        self._loop_stopped = Event()

        owner = self

        class _OfficialClient(EWrapper, EClient):
            def __init__(self) -> None:
                EWrapper.__init__(self)
                EClient.__init__(self, self)

            def nextValidId(self, orderId: int) -> None:
                owner._forward("on_next_valid_id", int(orderId))

            def error(self, reqId: int, *args: Any) -> None:
                (
                    error_time,
                    error_code,
                    error_string,
                    advanced_reject,
                ) = _parse_error_callback_args(args)
                owner._forward(
                    "on_error",
                    int(reqId),
                    error_code,
                    error_string,
                    advanced_reject,
                    error_time,
                )

            def contractDetails(self, reqId: int, contractDetails: Any) -> None:
                owner._forward("on_contract_details", int(reqId), contractDetails)

            def contractDetailsEnd(self, reqId: int) -> None:
                owner._forward("on_contract_details_end", int(reqId))

            def historicalData(self, reqId: int, bar: Any) -> None:
                owner._forward("on_historical_data", int(reqId), bar, False)

            def historicalDataUpdate(self, reqId: int, bar: Any) -> None:
                owner._forward("on_historical_data", int(reqId), bar, True)

            def historicalDataEnd(self, reqId: int, start: str, end: str) -> None:
                owner._forward("on_historical_data_end", int(reqId), str(start), str(end))

            def tickPrice(self, reqId: int, tickType: int, price: float, attrib: Any) -> None:
                field = _TICK_PRICE_FIELDS.get(int(tickType))
                if field is not None:
                    owner._forward(
                        "on_quote_field",
                        int(reqId),
                        field,
                        float(price),
                        datetime.now(UTC),
                    )

            def tickSize(self, reqId: int, tickType: int, size: Any) -> None:
                field = _TICK_SIZE_FIELDS.get(int(tickType))
                if field is not None:
                    owner._forward(
                        "on_quote_field",
                        int(reqId),
                        field,
                        _number(size),
                        datetime.now(UTC),
                    )

            def marketDataType(self, reqId: int, marketDataType: int) -> None:
                owner._forward("on_market_data_type", int(reqId), int(marketDataType))

            def connectionClosed(self) -> None:
                owner._forward("on_connection_closed")

        self._client = _OfficialClient()

    def connect(self, host: str, port: int, client_id: int) -> None:
        # The official EClient.connect() reports failures through EWrapper.error
        # and returns None even after a successful synchronous socket handshake.
        self._client.connect(str(host), int(port), int(client_id))

    def start_network_loop(self) -> Thread:
        with self._loop_lock:
            if self._loop_thread is not None and self._loop_thread.is_alive():
                return self._loop_thread
            self._loop_stopped.clear()
            self._loop_thread = Thread(
                target=self._run_loop,
                name="ibkr-api-network-loop",
                daemon=True,
            )
            self._loop_thread.start()
            return self._loop_thread

    def stop(self) -> None:
        try:
            if self._client.isConnected():
                self._client.disconnect()
        finally:
            with self._loop_lock:
                thread = self._loop_thread
            if thread is not None and thread is not current_thread():
                thread.join(timeout=5.0)

    def is_connected(self) -> bool:
        return bool(self._client.isConnected())

    def build_contract(self, **values: Any):
        contract = self._contract_class()
        for field, value in values.items():
            if value not in {None, ""}:
                setattr(contract, field, value)
        return contract

    def request_market_data_type(self, market_data_type: int) -> None:
        self._client.reqMarketDataType(int(market_data_type))

    def request_contract_details(self, request_id: int, contract: Any) -> None:
        self._client.reqContractDetails(int(request_id), contract)

    def request_historical_data(
        self,
        request_id: int,
        contract: Any,
        *,
        duration: str,
        bar_size: str,
        what_to_show: str,
        use_rth: bool,
        keep_up_to_date: bool,
    ) -> None:
        self._client.reqHistoricalData(
            int(request_id),
            contract,
            "",
            str(duration),
            str(bar_size),
            str(what_to_show),
            int(bool(use_rth)),
            2,
            bool(keep_up_to_date),
            [],
        )

    def cancel_historical_data(self, request_id: int) -> None:
        self._client.cancelHistoricalData(int(request_id))

    def request_market_data(self, request_id: int, contract: Any) -> None:
        self._client.reqMktData(int(request_id), contract, "", False, False, [])

    def cancel_market_data(self, request_id: int) -> None:
        self._client.cancelMktData(int(request_id))

    def _run_loop(self) -> None:
        try:
            self._client.run()
        except Exception:
            LOGGER.exception("IBKR API network loop stopped unexpectedly.")
            self._forward("on_connection_closed")
        finally:
            self._loop_stopped.set()

    def _forward(self, name: str, *args: Any) -> None:
        callback = getattr(self._callback_target, name, None)
        if callback is None:
            return
        try:
            callback(*args)
        except Exception:
            LOGGER.exception("IBKR callback %s failed.", name)


def _number(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _parse_error_callback_args(
    args: tuple[Any, ...],
) -> tuple[int | None, int, str, str]:
    """Support both pre-10.33 and timestamped 10.33+ EWrapper.error calls."""

    if (
        len(args) >= 3
        and _is_int_like(args[0])
        and _is_int_like(args[1])
    ):
        error_time = int(args[0])
        error_code = int(args[1])
        error_string = str(args[2])
        advanced_reject = str(args[3] or "") if len(args) >= 4 else ""
        return error_time, error_code, error_string, advanced_reject
    if len(args) >= 2 and _is_int_like(args[0]):
        error_code = int(args[0])
        error_string = str(args[1])
        advanced_reject = str(args[2] or "") if len(args) >= 3 else ""
        return None, error_code, error_string, advanced_reject
    raise TypeError(f"Unsupported IBKR error callback payload: {args!r}")


def _is_int_like(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)
