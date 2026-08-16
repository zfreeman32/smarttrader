from __future__ import annotations

from calendar import monthcalendar
from datetime import UTC, date, datetime, time, timedelta
from zoneinfo import ZoneInfo


NEW_YORK = ZoneInfo("America/New_York")
QUARTERLY_MONTHS = (3, 6, 9, 12)


def third_friday(year: int, month: int) -> date:
    if month not in QUARTERLY_MONTHS:
        raise ValueError("ES expiration month must be March, June, September, or December.")
    fridays = [
        week[4]
        for week in monthcalendar(int(year), int(month))
        if week[4] != 0
    ]
    return date(int(year), int(month), fridays[2])


def cme_calendar_roll_date(year: int, month: int) -> date:
    return third_friday(year, month) - timedelta(days=4)


def cme_calendar_roll_at(
    contract_month: str,
    *,
    roll_time_et: time = time(9, 30),
) -> datetime:
    normalized = normalize_contract_month(contract_month)
    year = int(normalized[:4])
    month = int(normalized[4:6])
    return datetime.combine(
        cme_calendar_roll_date(year, month),
        roll_time_et,
        tzinfo=NEW_YORK,
    )


def expected_quarterly_contract_month(
    now: datetime,
    *,
    roll_time_et: time = time(9, 30),
) -> str:
    current = ensure_utc(now)
    for year in range(current.year - 1, current.year + 3):
        for month in QUARTERLY_MONTHS:
            contract_month = f"{year:04d}{month:02d}"
            if current < cme_calendar_roll_at(
                contract_month,
                roll_time_et=roll_time_et,
            ).astimezone(UTC):
                return contract_month
    raise RuntimeError("Could not calculate the next quarterly ES contract.")


def next_quarterly_contract_month(contract_month: str) -> str:
    normalized = normalize_contract_month(contract_month)
    year = int(normalized[:4])
    month = int(normalized[4:6])
    index = QUARTERLY_MONTHS.index(month)
    if index == len(QUARTERLY_MONTHS) - 1:
        return f"{year + 1:04d}03"
    return f"{year:04d}{QUARTERLY_MONTHS[index + 1]:02d}"


def normalize_contract_month(value: str) -> str:
    digits = "".join(character for character in str(value) if character.isdigit())
    if len(digits) < 6:
        raise ValueError(f"Malformed ES contract month {value!r}.")
    normalized = digits[:6]
    if int(normalized[4:6]) not in QUARTERLY_MONTHS:
        raise ValueError(f"ES contract month {value!r} is not quarterly.")
    return normalized


def is_es_globex_open(now: datetime) -> bool:
    local = ensure_utc(now).astimezone(NEW_YORK)
    weekday = local.weekday()
    clock = local.timetz().replace(tzinfo=None)
    if weekday == 5:
        return False
    if weekday == 6:
        return clock >= time(18, 0)
    if weekday == 4 and clock >= time(17, 0):
        return False
    return not (time(17, 0) <= clock < time(18, 0))


def ensure_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)
