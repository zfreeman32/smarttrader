from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory

from ote_live.env import env_path, load_repo_env
from ote_live.ingestion.runtime import DEFAULT_DB_PATH
from ote_live.storage import (
    AUDIT_TABLE_SPECS,
    SQLiteLiveDataStore,
    archive_calendar_month,
    build_retention_plan,
    purge_retained_data,
    restore_calendar_month,
)
from ote_live.storage.archive import ArchiveBatchResult, RestoreBatchResult
from ote_live.storage.retention import RetentionExecutionResult, RetentionPlan, RetentionPolicy

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARCHIVE_ROOT = REPO_ROOT / "ote_live" / "runtime_data" / "archive"


@dataclass(frozen=True)
class MonthlyArchiveOpsReport:
    archive_result: ArchiveBatchResult
    restore_result: RestoreBatchResult
    verification_errors: tuple[str, ...]
    retention_plan: RetentionPlan | None = None
    purge_result: RetentionExecutionResult | None = None
    purge_executed: bool = False

    @property
    def verification_passed(self) -> bool:
        return not self.verification_errors


def build_parser() -> argparse.ArgumentParser:
    load_repo_env()
    parser = argparse.ArgumentParser(
        description="Archive one audit-data month to Parquet, verify it can be restored, and optionally purge retained rows.",
    )
    parser.add_argument("--db-path", default=str(env_path("OTE_LIVE_DB_PATH", DEFAULT_DB_PATH)))
    parser.add_argument("--archive-root", default=str(env_path("OTE_LIVE_ARCHIVE_ROOT", DEFAULT_ARCHIVE_ROOT)))
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--month", type=int, required=True)
    parser.add_argument(
        "--table",
        action="append",
        dest="table_names",
        help="Limit the operation to one audit table. Repeatable. Defaults to all audit tables.",
    )
    parser.add_argument("--compression", default="snappy")
    parser.add_argument(
        "--retention-days",
        type=int,
        default=None,
        help="If provided, build a retention plan for the selected tables using a uniform max-age threshold.",
    )
    parser.add_argument(
        "--reference-time",
        default=None,
        help="Optional ISO-8601 timestamp used when building the retention plan.",
    )
    parser.add_argument(
        "--purge",
        action="store_true",
        help="Actually delete retained rows after archive verification passes. Without this flag, retention stays dry-run.",
    )
    return parser


def execute_monthly_archive_ops(
    *,
    db_path: str | Path,
    archive_root: str | Path,
    year: int,
    month: int,
    table_names: tuple[str, ...] | None = None,
    compression: str = "snappy",
    retention_days: int | None = None,
    reference_time: datetime | None = None,
    purge: bool = False,
) -> MonthlyArchiveOpsReport:
    selected_table_names = table_names or tuple(spec.table_name for spec in AUDIT_TABLE_SPECS)

    with SQLiteLiveDataStore(db_path) as store:
        archive_result = archive_calendar_month(
            store,
            archive_root,
            year=year,
            month=month,
            table_names=selected_table_names,
            compression=compression,
        )
        restore_result = _verify_archive_restore(
            archive_root=archive_root,
            year=year,
            month=month,
            table_names=selected_table_names,
        )
        verification_errors = _collect_verification_errors(
            archive_result=archive_result,
            restore_result=restore_result,
        )

        retention_plan: RetentionPlan | None = None
        purge_result: RetentionExecutionResult | None = None
        purge_executed = False
        if retention_days is not None:
            policies = tuple(
                RetentionPolicy(
                    table_name=table_name,
                    max_age_days=int(retention_days),
                    require_archive=True,
                )
                for table_name in selected_table_names
            )
            retention_plan = build_retention_plan(
                store,
                policies=policies,
                reference_time=reference_time,
                archive_root=archive_root,
            )
            if purge and verification_errors:
                report = MonthlyArchiveOpsReport(
                    archive_result=archive_result,
                    restore_result=restore_result,
                    verification_errors=verification_errors,
                    retention_plan=retention_plan,
                    purge_result=None,
                    purge_executed=False,
                )
                raise RuntimeError(
                    "Archive verification failed; refusing to purge retained rows.\n"
                    + "\n".join(report.verification_errors)
                )
            purge_result = purge_retained_data(
                store,
                plan=retention_plan,
                dry_run=not purge,
            )
            purge_executed = bool(purge)

        report = MonthlyArchiveOpsReport(
            archive_result=archive_result,
            restore_result=restore_result,
            verification_errors=verification_errors,
            retention_plan=retention_plan,
            purge_result=purge_result,
            purge_executed=purge_executed,
        )
        return report


def main() -> int:
    load_repo_env()
    parser = build_parser()
    args = parser.parse_args()
    if args.purge and args.retention_days is None:
        parser.error("--purge requires --retention-days.")

    reference_time = _parse_optional_datetime(args.reference_time) if args.reference_time else None
    report = execute_monthly_archive_ops(
        db_path=Path(args.db_path),
        archive_root=Path(args.archive_root),
        year=args.year,
        month=args.month,
        table_names=tuple(args.table_names) if args.table_names else None,
        compression=args.compression,
        retention_days=args.retention_days,
        reference_time=reference_time,
        purge=args.purge,
    )
    _print_report(report)
    return 0 if report.verification_passed else 2


def _verify_archive_restore(
    *,
    archive_root: str | Path,
    year: int,
    month: int,
    table_names: tuple[str, ...],
) -> RestoreBatchResult:
    archive_root = Path(archive_root)
    archive_root.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="ote_live_archive_verify_", dir=archive_root) as temp_dir:
        verify_db_path = Path(temp_dir) / "verify.sqlite"
        with SQLiteLiveDataStore(verify_db_path) as verify_store:
            return restore_calendar_month(
                verify_store,
                archive_root,
                year=year,
                month=month,
                table_names=table_names,
                conflict_strategy="abort",
            )


def _collect_verification_errors(
    *,
    archive_result: ArchiveBatchResult,
    restore_result: RestoreBatchResult,
) -> tuple[str, ...]:
    restore_by_table = {
        result.table_name: result
        for result in restore_result.table_results
    }
    errors: list[str] = []
    for archived in archive_result.table_results:
        restored = restore_by_table.get(archived.table_name)
        if archived.row_count > 0:
            if archived.archive_path is None or not archived.archive_path.exists():
                errors.append(f"{archived.table_name}: archive file missing after archive step.")
        if restored is None:
            errors.append(f"{archived.table_name}: restore step did not return a result.")
            continue
        if restored.row_count != archived.row_count:
            errors.append(
                f"{archived.table_name}: archived_rows={archived.row_count} restored_rows={restored.row_count}."
            )
    return tuple(errors)


def _print_report(report: MonthlyArchiveOpsReport) -> None:
    print(f"archive_month={report.archive_result.archive_month}")
    print(
        f"verification={'passed' if report.verification_passed else 'failed'} "
        f"errors={len(report.verification_errors)}"
    )
    print("archive_results:")
    for archived in report.archive_result.table_results:
        restored_rows = next(
            (item.row_count for item in report.restore_result.table_results if item.table_name == archived.table_name),
            0,
        )
        archive_path = str(archived.archive_path) if archived.archive_path is not None else "-"
        print(
            f"  {archived.table_name}: archived_rows={archived.row_count} "
            f"restored_rows={restored_rows} archive_path={archive_path}"
        )

    if report.verification_errors:
        print("verification_errors:")
        for error in report.verification_errors:
            print(f"  {error}")

    if report.retention_plan is not None:
        print(
            f"retention_plan: generated_at_utc={report.retention_plan.generated_at_utc.isoformat()} "
            f"eligible_rows={report.retention_plan.total_eligible_rows}"
        )
        for candidate in report.retention_plan.candidates:
            print(
                f"  {candidate.table_name}: eligible_rows={candidate.eligible_rows} "
                f"cutoff_utc={candidate.cutoff_utc.isoformat()} "
                f"archive_confirmed={candidate.archive_confirmed}"
            )

    if report.purge_result is not None:
        mode = "executed" if report.purge_executed else "dry_run"
        print(f"purge_{mode}: deleted_rows={report.purge_result.total_deleted_rows}")
        for table_name, deleted_rows in sorted(report.purge_result.deleted_rows_by_table.items()):
            print(f"  {table_name}: {deleted_rows}")


def _parse_optional_datetime(value: str) -> datetime:
    try:
        return datetime.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Expected an ISO-8601 timestamp for --reference-time, got {value!r}."
        ) from exc


if __name__ == "__main__":
    raise SystemExit(main())
