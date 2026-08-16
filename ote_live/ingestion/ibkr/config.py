from __future__ import annotations

from dataclasses import dataclass
from datetime import time
import os
from typing import Any


_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})
_FALSE_VALUES = frozenset({"0", "false", "no", "off"})
_MARKET_DATA_TYPES = {
    "live": 1,
    "frozen": 2,
    "delayed": 3,
    "delayed_frozen": 4,
    "delayed-frozen": 4,
    "1": 1,
    "2": 2,
    "3": 3,
    "4": 4,
}
_MARKET_DATA_NAMES = {1: "live", 2: "frozen", 3: "delayed", 4: "delayed_frozen"}


def market_data_type_code(value: str | int) -> int:
    key = str(value).strip().lower()
    try:
        return _MARKET_DATA_TYPES[key]
    except KeyError as exc:
        raise ValueError(
            "IBKR market data type must be live, frozen, delayed, delayed_frozen, or 1-4."
        ) from exc


def market_data_type_name(value: str | int | None) -> str | None:
    if value is None:
        return None
    return _MARKET_DATA_NAMES.get(market_data_type_code(value), f"unknown:{value}")


def parse_roll_time(value: str | time) -> time:
    if isinstance(value, time):
        if value.tzinfo is not None:
            return value.replace(tzinfo=None)
        return value
    text = str(value).strip()
    parts = text.split(":")
    if len(parts) not in {2, 3}:
        raise ValueError("IBKR_ES_ROLL_TIME_ET must use HH:MM or HH:MM:SS.")
    return time(*[int(part) for part in parts])


@dataclass(frozen=True)
class IBKRConfig:
    enabled: bool = False
    host: str = "127.0.0.1"
    port: int = 4001
    client_id: int = 21
    account_mode: str = "paper"
    market_data_type: str | int = "live"
    allow_delayed_fallback: bool = False

    symbol: str = "ES"
    security_type: str = "FUT"
    exchange: str = "CME"
    currency: str = "USD"
    multiplier: str = "50"
    trading_class: str = "ES"

    bar_size: str = "5 mins"
    history_duration: str = "2 D"
    what_to_show: str = "TRADES"
    use_rth: bool = False
    keep_up_to_date: bool = True

    roll_policy: str = "cme_calendar"
    roll_time_et: str | time = "09:30"
    manual_conid: int | None = None
    manual_contract_month: str | None = None

    stale_quote_seconds: float = 15.0
    stale_bar_seconds: float = 420.0
    delayed_history_refresh_seconds: float = 240.0
    reconnect_initial_seconds: float = 2.0
    reconnect_max_seconds: float = 60.0
    contract_refresh_seconds: float = 86_400.0
    contract_timeout_seconds: float = 15.0
    history_timeout_seconds: float = 30.0
    handshake_timeout_seconds: float = 15.0
    store_max_bars: int = 2_000

    readonly: bool = True

    # Compatibility aliases retained for the existing collector CLI.
    local_symbol: str | None = None
    contract_month: str | None = None
    con_id: int | None = None
    backfill_duration: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "host", str(self.host).strip() or "127.0.0.1")
        object.__setattr__(self, "port", int(self.port))
        object.__setattr__(self, "client_id", int(self.client_id))
        object.__setattr__(self, "account_mode", str(self.account_mode).strip().lower())
        object.__setattr__(self, "symbol", str(self.symbol).strip().upper())
        object.__setattr__(self, "security_type", str(self.security_type).strip().upper())
        object.__setattr__(self, "exchange", str(self.exchange).strip().upper())
        object.__setattr__(self, "currency", str(self.currency).strip().upper())
        object.__setattr__(self, "multiplier", str(self.multiplier).strip())
        object.__setattr__(self, "trading_class", str(self.trading_class).strip().upper())
        object.__setattr__(self, "what_to_show", str(self.what_to_show).strip().upper())
        object.__setattr__(self, "roll_policy", str(self.roll_policy).strip().lower())
        object.__setattr__(self, "market_data_type", market_data_type_name(self.market_data_type) or "live")
        object.__setattr__(self, "roll_time_et", parse_roll_time(self.roll_time_et))
        object.__setattr__(self, "manual_conid", self.manual_conid or self.con_id)
        object.__setattr__(
            self,
            "manual_contract_month",
            _normalize_contract_month(self.manual_contract_month or self.contract_month),
        )
        if self.backfill_duration:
            object.__setattr__(self, "history_duration", str(self.backfill_duration))

        if not (1 <= self.port <= 65_535):
            raise ValueError("IBKR_PORT must be between 1 and 65535.")
        if self.client_id < 0:
            raise ValueError("IBKR_CLIENT_ID must be non-negative.")
        if self.account_mode not in {"paper", "live"}:
            raise ValueError("IBKR_ACCOUNT_MODE must be paper or live.")
        if self.security_type != "FUT":
            raise ValueError("The ES live provider only supports IBKR_ES_SECURITY_TYPE=FUT.")
        if self.roll_policy != "cme_calendar":
            raise ValueError("The supported IBKR_ES_ROLL_POLICY is cme_calendar.")
        if int(float(self.multiplier)) != 50:
            raise ValueError("The E-mini S&P 500 multiplier must be 50.")
        for name in (
            "stale_quote_seconds",
            "stale_bar_seconds",
            "delayed_history_refresh_seconds",
            "reconnect_initial_seconds",
            "reconnect_max_seconds",
            "contract_refresh_seconds",
            "contract_timeout_seconds",
            "history_timeout_seconds",
            "handshake_timeout_seconds",
        ):
            if float(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive.")
        if self.reconnect_initial_seconds > self.reconnect_max_seconds:
            raise ValueError("Reconnect initial delay cannot exceed reconnect maximum delay.")
        if self.store_max_bars < 2:
            raise ValueError("IBKR bar store must retain at least two bars.")

    @property
    def market_data_type_code(self) -> int:
        return market_data_type_code(self.market_data_type)

    @classmethod
    def from_env(cls, environ: dict[str, str] | None = None) -> "IBKRConfig":
        values = os.environ if environ is None else environ
        get = values.get
        return cls(
            enabled=_env_bool(get("IBKR_ENABLED"), False),
            host=get("IBKR_HOST", "127.0.0.1"),
            port=int(get("IBKR_PORT", "4001")),
            client_id=int(get("IBKR_CLIENT_ID", "21")),
            account_mode=get("IBKR_ACCOUNT_MODE", "paper"),
            market_data_type=get("IBKR_MARKET_DATA_TYPE", "live"),
            allow_delayed_fallback=_env_bool(get("IBKR_ALLOW_DELAYED_FALLBACK"), False),
            symbol=get("IBKR_ES_SYMBOL", "ES"),
            security_type=get("IBKR_ES_SECURITY_TYPE", "FUT"),
            exchange=get("IBKR_ES_EXCHANGE", "CME"),
            currency=get("IBKR_ES_CURRENCY", "USD"),
            multiplier=get("IBKR_ES_MULTIPLIER", "50"),
            trading_class=get("IBKR_ES_TRADING_CLASS", "ES"),
            bar_size=get("IBKR_ES_BAR_SIZE", "5 mins"),
            history_duration=get("IBKR_ES_HISTORY_DURATION", "2 D"),
            use_rth=_env_bool(get("IBKR_ES_USE_RTH"), False),
            roll_policy=get("IBKR_ES_ROLL_POLICY", "cme_calendar"),
            roll_time_et=get("IBKR_ES_ROLL_TIME_ET", "09:30"),
            manual_conid=_optional_int(get("IBKR_ES_MANUAL_CONID")),
            manual_contract_month=get("IBKR_ES_MANUAL_CONTRACT_MONTH"),
            stale_quote_seconds=float(get("IBKR_STALE_QUOTE_SECONDS", "15")),
            stale_bar_seconds=float(get("IBKR_STALE_BAR_SECONDS", "420")),
            delayed_history_refresh_seconds=float(
                get("IBKR_DELAYED_HISTORY_REFRESH_SECONDS", "240")
            ),
            reconnect_initial_seconds=float(get("IBKR_RECONNECT_INITIAL_SECONDS", "2")),
            reconnect_max_seconds=float(get("IBKR_RECONNECT_MAX_SECONDS", "60")),
        )


def _env_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    lowered = str(value).strip().lower()
    if lowered in _TRUE_VALUES:
        return True
    if lowered in _FALSE_VALUES:
        return False
    raise ValueError(f"Boolean environment value must be one of {_TRUE_VALUES | _FALSE_VALUES}.")


def _optional_int(value: Any) -> int | None:
    if value in {None, ""}:
        return None
    return int(value)


def _normalize_contract_month(value: str | None) -> str | None:
    if value is None:
        return None
    digits = "".join(character for character in str(value) if character.isdigit())
    if len(digits) < 6:
        raise ValueError("Manual ES contract month must be YYYYMM or YYYYMMDD.")
    month = digits[:6]
    if int(month[-2:]) not in {3, 6, 9, 12}:
        raise ValueError("Manual ES contract month must be a quarterly H/M/U/Z month.")
    return month
