from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, Literal

from ote_live.storage.db import SQLiteLiveDataStore
from ote_live.storage.repositories import AUDIT_TABLE_SPECS, AUDIT_TABLE_SPECS_BY_NAME, AuditTableSpec

if TYPE_CHECKING:
    import pandas as pd


@dataclass(frozen=True)
class ArchivedTableResult:
    table_name: str
    row_count: int
    archive_path: Path | None


@dataclass(frozen=True)
class ArchiveBatchResult:
    archive_month: str
    table_results: tuple[ArchivedTableResult, ...]

    @property
    def total_rows(self) -> int:
        return sum(result.row_count for result in self.table_results)


@dataclass(frozen=True)
class RestoreTableResult:
    table_name: str
    row_count: int
    archive_path: Path | None


@dataclass(frozen=True)
class RestoreBatchResult:
    archive_month: str
    table_results: tuple[RestoreTableResult, ...]

    @property
    def total_rows(self) -> int:
        return sum(result.row_count for result in self.table_results)


def archive_calendar_month(
    store: SQLiteLiveDataStore,
    archive_root: str | Path,
    *,
    year: int,
    month: int,
    table_names: Iterable[str] | None = None,
    compression: str = "snappy",
) -> ArchiveBatchResult:
    pd = _require_pandas()
    _require_pyarrow()
    start, end = _calendar_month_bounds(year=year, month=month)
    selected_specs = _select_specs(table_names)
    archive_root = Path(archive_root)

    table_results: list[ArchivedTableResult] = []
    for spec in selected_specs:
        frame = pd.read_sql_query(
            f"""
            SELECT *
            FROM {spec.table_name}
            WHERE {spec.timestamp_expression} >= ? AND {spec.timestamp_expression} < ?
            ORDER BY {spec.timestamp_expression} ASC
            """,
            store.connection,
            params=(start.isoformat(), end.isoformat()),
        )
        archive_path: Path | None = None
        if not frame.empty:
            archive_path = archive_path_for_table(
                archive_root,
                table_name=spec.table_name,
                year=year,
                month=month,
            )
            archive_path.parent.mkdir(parents=True, exist_ok=True)
            frame.to_parquet(archive_path, index=False, compression=compression)
        table_results.append(
            ArchivedTableResult(
                table_name=spec.table_name,
                row_count=int(len(frame)),
                archive_path=archive_path,
            )
        )

    return ArchiveBatchResult(
        archive_month=f"{year:04d}-{month:02d}",
        table_results=tuple(table_results),
    )


def restore_calendar_month(
    store: SQLiteLiveDataStore,
    archive_root: str | Path,
    *,
    year: int,
    month: int,
    table_names: Iterable[str] | None = None,
    conflict_strategy: Literal["abort", "ignore", "replace"] = "ignore",
) -> RestoreBatchResult:
    pd = _require_pandas()
    _require_pyarrow()
    selected_specs = _select_specs(table_names)
    archive_root = Path(archive_root)
    sqlite_conflict_clause = {
        "abort": "",
        "ignore": " OR IGNORE",
        "replace": " OR REPLACE",
    }[conflict_strategy]

    table_results: list[RestoreTableResult] = []
    for spec in sorted(selected_specs, key=lambda item: item.restore_order):
        archive_path = archive_path_for_table(
            archive_root,
            table_name=spec.table_name,
            year=year,
            month=month,
        )
        if not archive_path.exists():
            table_results.append(
                RestoreTableResult(
                    table_name=spec.table_name,
                    row_count=0,
                    archive_path=None,
                )
            )
            continue

        frame = pd.read_parquet(archive_path)
        if frame.empty:
            table_results.append(
                RestoreTableResult(
                    table_name=spec.table_name,
                    row_count=0,
                    archive_path=archive_path,
                )
            )
            continue

        records = frame.where(frame.notna(), None).to_dict(orient="records")
        columns = list(frame.columns)
        placeholders = ", ".join("?" for _ in columns)
        column_clause = ", ".join(columns)
        store.connection.executemany(
            f"""
            INSERT{sqlite_conflict_clause} INTO {spec.table_name} ({column_clause})
            VALUES ({placeholders})
            """,
            ([record.get(column) for column in columns] for record in records),
        )
        table_results.append(
            RestoreTableResult(
                table_name=spec.table_name,
                row_count=int(len(records)),
                archive_path=archive_path,
            )
        )

    store.connection.commit()
    return RestoreBatchResult(
        archive_month=f"{year:04d}-{month:02d}",
        table_results=tuple(table_results),
    )


def archive_path_for_table(
    archive_root: str | Path,
    *,
    table_name: str,
    year: int,
    month: int,
) -> Path:
    return Path(archive_root) / table_name / f"year={year:04d}" / f"month={month:02d}" / "data.parquet"


def has_archive_for_month(
    archive_root: str | Path,
    *,
    table_name: str,
    year: int,
    month: int,
) -> bool:
    return archive_path_for_table(
        archive_root,
        table_name=table_name,
        year=year,
        month=month,
    ).exists()


def _calendar_month_bounds(*, year: int, month: int) -> tuple[datetime, datetime]:
    if month < 1 or month > 12:
        raise ValueError("month must be between 1 and 12.")
    start = datetime(year, month, 1, tzinfo=UTC)
    if month == 12:
        end = datetime(year + 1, 1, 1, tzinfo=UTC)
    else:
        end = datetime(year, month + 1, 1, tzinfo=UTC)
    return start, end


def _select_specs(table_names: Iterable[str] | None) -> tuple[AuditTableSpec, ...]:
    if table_names is None:
        return tuple(sorted(AUDIT_TABLE_SPECS, key=lambda item: item.restore_order))

    resolved: list[AuditTableSpec] = []
    for table_name in table_names:
        spec = AUDIT_TABLE_SPECS_BY_NAME.get(table_name)
        if spec is None:
            raise KeyError(f"Unknown audit table {table_name!r}.")
        resolved.append(spec)
    return tuple(sorted(resolved, key=lambda item: item.restore_order))


def _require_pandas() -> "pd":
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError("Phase 4 archive helpers require the 'pandas' package.") from exc
    return pd


def _require_pyarrow() -> None:
    try:
        import pyarrow  # noqa: F401
    except ImportError as exc:
        raise ImportError("Phase 4 archive helpers require the 'pyarrow' package.") from exc
