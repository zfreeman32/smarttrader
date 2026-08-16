from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime, time
import logging
from typing import Any, Iterable
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from .config import IBKRConfig
from .errors import IBKRContractError
from .roll import (
    cme_calendar_roll_at,
    expected_quarterly_contract_month,
    normalize_contract_month,
)

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class QualifiedESContract:
    conid: int
    local_symbol: str
    contract_month: str
    expiration: date
    symbol: str = "ES"
    security_type: str = "FUT"
    exchange: str = "CME"
    currency: str = "USD"
    multiplier: str = "50"
    trading_class: str = "ES"
    timezone: str | None = None
    trading_hours: str | None = None
    liquid_hours: str | None = None

    @property
    def identity(self) -> dict[str, Any]:
        return {
            "conId": self.conid,
            "local_symbol": self.local_symbol,
            "contract_month": self.contract_month,
            "expiration": self.expiration.isoformat(),
            "exchange": self.exchange,
            "multiplier": self.multiplier,
            "trading_class": self.trading_class,
        }


@dataclass(frozen=True)
class ContractSelection:
    contract: QualifiedESContract
    expected_contract_month: str
    manual_override: str | None = None
    fallback_reason: str | None = None
    roll_time_et: time = time(9, 30)

    @property
    def next_roll_at(self) -> datetime:
        return cme_calendar_roll_at(
            self.contract.contract_month,
            roll_time_et=self.roll_time_et,
        )


def contract_from_details(details: Any) -> QualifiedESContract:
    contract = getattr(details, "contract", details)
    raw_expiration = (
        getattr(details, "realExpirationDate", None)
        or getattr(contract, "lastTradeDateOrContractMonth", None)
    )
    expiration, contract_month = parse_expiration(raw_expiration)
    conid = int(getattr(contract, "conId", 0) or 0)
    local_symbol = str(getattr(contract, "localSymbol", "") or "").strip()
    if conid <= 0 or not local_symbol:
        raise ValueError("Qualified contract was missing conId or localSymbol.")
    return QualifiedESContract(
        conid=conid,
        local_symbol=local_symbol,
        contract_month=contract_month,
        expiration=expiration,
        symbol=str(getattr(contract, "symbol", "") or "").upper(),
        security_type=str(getattr(contract, "secType", "") or "").upper(),
        exchange=str(
            getattr(contract, "exchange", "")
            or getattr(contract, "primaryExchange", "")
            or ""
        ).upper(),
        currency=str(getattr(contract, "currency", "") or "").upper(),
        multiplier=str(getattr(contract, "multiplier", "") or ""),
        trading_class=str(getattr(contract, "tradingClass", "") or "").upper(),
        timezone=_optional_text(getattr(details, "timeZoneId", None)),
        trading_hours=_optional_text(getattr(details, "tradingHours", None)),
        liquid_hours=_optional_text(getattr(details, "liquidHours", None)),
    )


def filter_es_contracts(
    details: Iterable[Any],
    *,
    config: IBKRConfig,
    now: datetime,
) -> tuple[QualifiedESContract, ...]:
    today = _ensure_utc(now).date()
    by_conid: dict[int, QualifiedESContract] = {}
    for payload in details:
        try:
            contract = (
                payload
                if isinstance(payload, QualifiedESContract)
                else contract_from_details(payload)
            )
        except (TypeError, ValueError):
            continue
        if contract.symbol != config.symbol:
            continue
        if contract.security_type != "FUT":
            continue
        if contract.exchange != config.exchange:
            continue
        if contract.currency != config.currency:
            continue
        try:
            if int(float(contract.multiplier)) != int(float(config.multiplier)):
                continue
        except (TypeError, ValueError):
            continue
        if contract.trading_class and contract.trading_class != config.trading_class:
            continue
        try:
            if int(contract.contract_month[4:6]) not in {3, 6, 9, 12}:
                continue
        except (TypeError, ValueError):
            continue
        if contract.expiration < today:
            continue
        by_conid[contract.conid] = contract
    return tuple(
        sorted(
            by_conid.values(),
            key=lambda item: (item.expiration, item.contract_month, item.conid),
        )
    )


def select_es_contract(
    contracts: Iterable[QualifiedESContract],
    *,
    config: IBKRConfig,
    now: datetime,
) -> ContractSelection:
    candidates = tuple(contracts)
    if not candidates:
        raise IBKRContractError("IBKR returned no valid, unexpired quarterly ES FUT contracts.")
    today = _ensure_utc(now).date()
    valid = tuple(contract for contract in candidates if contract.expiration >= today)
    if not valid:
        raise IBKRContractError("IBKR returned only expired ES futures contracts.")

    expected_month = expected_quarterly_contract_month(
        now,
        roll_time_et=config.roll_time_et,
    )
    if config.manual_conid is not None:
        matches = [item for item in valid if item.conid == int(config.manual_conid)]
        if len(matches) != 1:
            raise IBKRContractError(
                f"IBKR_ES_MANUAL_CONID={config.manual_conid} did not resolve to one valid ES contract."
            )
        return ContractSelection(
            matches[0],
            expected_contract_month=expected_month,
            manual_override=f"conId:{config.manual_conid}",
            roll_time_et=config.roll_time_et,
        )

    if config.manual_contract_month is not None:
        manual_month = normalize_contract_month(config.manual_contract_month)
        matches = [item for item in valid if item.contract_month == manual_month]
        if len(matches) != 1:
            raise IBKRContractError(
                f"IBKR_ES_MANUAL_CONTRACT_MONTH={manual_month} did not resolve to one valid ES contract."
            )
        return ContractSelection(
            matches[0],
            expected_contract_month=expected_month,
            manual_override=f"contract_month:{manual_month}",
            roll_time_et=config.roll_time_et,
        )

    expected_matches = [item for item in valid if item.contract_month == expected_month]
    if len(expected_matches) == 1:
        return ContractSelection(
            expected_matches[0],
            expected_contract_month=expected_month,
            roll_time_et=config.roll_time_et,
        )
    if len(expected_matches) > 1:
        raise IBKRContractError(
            f"IBKR returned ambiguous qualified contracts for expected ES month {expected_month}."
        )

    fallback = min(valid, key=lambda item: (item.expiration, item.conid))
    reason = (
        f"Expected ES contract month {expected_month} was unavailable; "
        f"selected nearest unexpired {fallback.contract_month} ({fallback.local_symbol})."
    )
    LOGGER.warning(reason)
    return ContractSelection(
        fallback,
        expected_contract_month=expected_month,
        fallback_reason=reason,
        roll_time_et=config.roll_time_et,
    )


def parse_expiration(value: Any) -> tuple[date, str]:
    digits = "".join(character for character in str(value or "") if character.isdigit())
    if len(digits) >= 8:
        parsed = datetime.strptime(digits[:8], "%Y%m%d").date()
        return parsed, normalize_contract_month(digits[:6])
    if len(digits) == 6:
        month = normalize_contract_month(digits)
        year = int(month[:4])
        month_number = int(month[4:6])
        # A month-only callback is accepted for discovery, but uses the standard
        # third-Friday last-trading date until realExpirationDate is supplied.
        from .roll import third_friday

        return third_friday(year, month_number), month
    raise ValueError(f"Malformed IBKR contract expiration {value!r}.")


def is_contract_trading_time(
    contract: QualifiedESContract,
    now: datetime,
) -> bool | None:
    """Use IBKR-reported trading hours when the current date is in the cache."""

    if not contract.trading_hours or not contract.timezone:
        return None
    try:
        timezone = ZoneInfo(contract.timezone)
    except ZoneInfoNotFoundError:
        return None
    current = _ensure_utc(now).astimezone(timezone)
    current_naive = current.replace(tzinfo=None)
    saw_current_date = False
    for segment in contract.trading_hours.split(";"):
        text = segment.strip()
        if not text or ":" not in text:
            continue
        session_date_text, hours = text.split(":", 1)
        if session_date_text == current.strftime("%Y%m%d"):
            saw_current_date = True
        if hours.upper() == "CLOSED" or "-" not in hours:
            continue
        start_text, end_text = hours.split("-", 1)
        try:
            start = datetime.strptime(
                f"{session_date_text}:{start_text}",
                "%Y%m%d:%H%M",
            )
            if ":" in end_text:
                end_date_text, end_clock_text = end_text.split(":", 1)
            else:
                end_date_text, end_clock_text = session_date_text, end_text
            end = datetime.strptime(
                f"{end_date_text}:{end_clock_text}",
                "%Y%m%d:%H%M",
            )
        except ValueError:
            continue
        if start <= current_naive < end:
            return True
    if saw_current_date:
        return False
    return None


def _optional_text(value: Any) -> str | None:
    text = str(value or "").strip()
    return text or None


def _ensure_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)
