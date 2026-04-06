from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Iterable

from ote_live.ingestion.base import ensure_utc, utc_now
from ote_live.storage.archive import has_archive_for_month
from ote_live.storage.db import SQLiteLiveDataStore
from ote_live.storage.repositories import AUDIT_TABLE_SPECS, AUDIT_TABLE_SPECS_BY_NAME


@dataclass(frozen=True)
class RetentionPolicy:
    table_name: str
    max_age_days: int
    require_archive: bool = True


@dataclass(frozen=True)
class RetentionCandidate:
    table_name: str
    cutoff_utc: datetime
    max_age_days: int
    eligible_rows: int
    oldest_timestamp_utc: datetime | None
    newest_timestamp_utc: datetime | None
    require_archive: bool
    archive_confirmed: bool


@dataclass(frozen=True)
class RetentionPlan:
    generated_at_utc: datetime
    candidates: tuple[RetentionCandidate, ...]

    @property
    def total_eligible_rows(self) -> int:
        return sum(candidate.eligible_rows for candidate in self.candidates)


@dataclass(frozen=True)
class RetentionExecutionResult:
    deleted_rows_by_table: dict[str, int]

    @property
    def total_deleted_rows(self) -> int:
        return sum(self.deleted_rows_by_table.values())


def build_retention_plan(
    store: SQLiteLiveDataStore,
    *,
    policies: Iterable[RetentionPolicy],
    reference_time: datetime | None = None,
    archive_root: str | Path | None = None,
) -> RetentionPlan:
    generated_at = ensure_utc(reference_time or utc_now())
    candidates: list[RetentionCandidate] = []

    for policy in policies:
        spec = AUDIT_TABLE_SPECS_BY_NAME.get(policy.table_name)
        if spec is None:
            raise KeyError(f"Unknown audit table {policy.table_name!r}.")

        cutoff = generated_at - timedelta(days=int(policy.max_age_days))
        row = store.connection.execute(
            f"""
            SELECT
                COUNT(*) AS eligible_rows,
                MIN({spec.timestamp_expression}) AS oldest_timestamp_utc,
                MAX({spec.timestamp_expression}) AS newest_timestamp_utc
            FROM {spec.table_name}
            WHERE {spec.timestamp_expression} < ?
            """,
            (cutoff.isoformat(),),
        ).fetchone()
        oldest = _parse_optional_datetime(row["oldest_timestamp_utc"])
        newest = _parse_optional_datetime(row["newest_timestamp_utc"])
        archive_confirmed = (
            _archive_coverage_confirmed(
                archive_root=archive_root,
                table_name=policy.table_name,
                oldest_timestamp_utc=oldest,
                newest_timestamp_utc=newest,
            )
            if policy.require_archive
            else True
        )
        candidates.append(
            RetentionCandidate(
                table_name=policy.table_name,
                cutoff_utc=cutoff,
                max_age_days=int(policy.max_age_days),
                eligible_rows=int(row["eligible_rows"]),
                oldest_timestamp_utc=oldest,
                newest_timestamp_utc=newest,
                require_archive=bool(policy.require_archive),
                archive_confirmed=archive_confirmed,
            )
        )

    return RetentionPlan(
        generated_at_utc=generated_at,
        candidates=tuple(candidates),
    )


def purge_retained_data(
    store: SQLiteLiveDataStore,
    *,
    plan: RetentionPlan,
    dry_run: bool = True,
) -> RetentionExecutionResult:
    deleted_rows_by_table: dict[str, int] = {}
    spec_order = {
        spec.table_name: spec.restore_order
        for spec in AUDIT_TABLE_SPECS
    }

    for candidate in sorted(
        plan.candidates,
        key=lambda item: spec_order.get(item.table_name, 0),
        reverse=True,
    ):
        if candidate.eligible_rows <= 0:
            deleted_rows_by_table[candidate.table_name] = 0
            continue
        if candidate.require_archive and not candidate.archive_confirmed:
            deleted_rows_by_table[candidate.table_name] = 0
            continue

        spec = AUDIT_TABLE_SPECS_BY_NAME[candidate.table_name]
        if dry_run:
            deleted_rows_by_table[candidate.table_name] = candidate.eligible_rows
            continue

        cursor = store.connection.execute(
            f"""
            DELETE FROM {candidate.table_name}
            WHERE {spec.timestamp_expression} < ?
            """,
            (candidate.cutoff_utc.isoformat(),),
        )
        deleted_rows_by_table[candidate.table_name] = int(cursor.rowcount)

    if not dry_run:
        store.connection.commit()

    return RetentionExecutionResult(deleted_rows_by_table=deleted_rows_by_table)


def _archive_coverage_confirmed(
    *,
    archive_root: str | Path | None,
    table_name: str,
    oldest_timestamp_utc: datetime | None,
    newest_timestamp_utc: datetime | None,
) -> bool:
    if oldest_timestamp_utc is None or newest_timestamp_utc is None:
        return True
    if archive_root is None:
        return False

    current = datetime(oldest_timestamp_utc.year, oldest_timestamp_utc.month, 1, tzinfo=UTC)
    last = datetime(newest_timestamp_utc.year, newest_timestamp_utc.month, 1, tzinfo=UTC)
    while current <= last:
        if not has_archive_for_month(
            archive_root,
            table_name=table_name,
            year=current.year,
            month=current.month,
        ):
            return False
        if current.month == 12:
            current = datetime(current.year + 1, 1, 1, tzinfo=UTC)
        else:
            current = datetime(current.year, current.month + 1, 1, tzinfo=UTC)
    return True


def _parse_optional_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    return ensure_utc(datetime.fromisoformat(value))
