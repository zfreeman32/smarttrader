from __future__ import annotations

import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ote_live.scripts import run_monthly_archive_retention as script
from ote_live.storage.archive import (
    ArchiveBatchResult,
    ArchivedTableResult,
    RestoreBatchResult,
    RestoreTableResult,
)
from ote_live.storage.retention import (
    RetentionCandidate,
    RetentionExecutionResult,
    RetentionPlan,
)


def test_execute_monthly_archive_ops_runs_archive_verify_and_dry_run_purge(monkeypatch) -> None:
    calls: list[tuple[str, object]] = []

    class _FakeStore:
        def __init__(self, path) -> None:
            self.path = Path(path)

        def __enter__(self):
            calls.append(("open_store", self.path))
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            calls.append(("close_store", self.path))

    archive_file = ROOT / "tmp" / "archive_ops_script_test" / "signal_decisions" / "year=2026" / "month=03" / "data.parquet"
    archive_file.parent.mkdir(parents=True, exist_ok=True)
    archive_file.write_bytes(b"placeholder")

    archive_result = ArchiveBatchResult(
        archive_month="2026-03",
        table_results=(
            ArchivedTableResult(
                table_name="signal_decisions",
                row_count=12,
                archive_path=archive_file,
            ),
        ),
    )
    restore_result = RestoreBatchResult(
        archive_month="2026-03",
        table_results=(
            RestoreTableResult(
                table_name="signal_decisions",
                row_count=12,
                archive_path=archive_file,
            ),
        ),
    )
    retention_plan = RetentionPlan(
        generated_at_utc=datetime(2026, 4, 5, tzinfo=UTC),
        candidates=(
            RetentionCandidate(
                table_name="signal_decisions",
                cutoff_utc=datetime(2026, 1, 5, tzinfo=UTC),
                max_age_days=90,
                eligible_rows=7,
                oldest_timestamp_utc=datetime(2025, 12, 1, tzinfo=UTC),
                newest_timestamp_utc=datetime(2026, 1, 4, tzinfo=UTC),
                require_archive=True,
                archive_confirmed=True,
            ),
        ),
    )
    purge_result = RetentionExecutionResult(deleted_rows_by_table={"signal_decisions": 7})

    monkeypatch.setattr(script, "SQLiteLiveDataStore", _FakeStore)

    def _archive_calendar_month(store, archive_root, **kwargs):
        calls.append(("archive", kwargs["table_names"]))
        return archive_result

    def _verify_archive_restore(**kwargs):
        calls.append(("verify", kwargs["table_names"]))
        return restore_result

    def _build_retention_plan(store, **kwargs):
        calls.append(("retention_plan", tuple(policy.table_name for policy in kwargs["policies"])))
        return retention_plan

    def _purge_retained_data(store, **kwargs):
        calls.append(("purge", kwargs["dry_run"]))
        return purge_result

    monkeypatch.setattr(script, "archive_calendar_month", _archive_calendar_month)
    monkeypatch.setattr(script, "_verify_archive_restore", _verify_archive_restore)
    monkeypatch.setattr(script, "build_retention_plan", _build_retention_plan)
    monkeypatch.setattr(script, "purge_retained_data", _purge_retained_data)

    report = script.execute_monthly_archive_ops(
        db_path=ROOT / "tmp" / "fake.sqlite",
        archive_root=ROOT / "tmp" / "archive",
        year=2026,
        month=3,
        table_names=("signal_decisions",),
        retention_days=90,
        purge=False,
    )

    assert report.verification_passed
    assert report.purge_executed is False
    assert report.purge_result == purge_result
    assert calls == [
        ("open_store", ROOT / "tmp" / "fake.sqlite"),
        ("archive", ("signal_decisions",)),
        ("verify", ("signal_decisions",)),
        ("retention_plan", ("signal_decisions",)),
        ("purge", True),
        ("close_store", ROOT / "tmp" / "fake.sqlite"),
    ]


def test_execute_monthly_archive_ops_refuses_purge_when_verification_fails(monkeypatch) -> None:
    class _FakeStore:
        def __init__(self, path) -> None:
            self.path = Path(path)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            return None

    archive_result = ArchiveBatchResult(
        archive_month="2026-03",
        table_results=(
            ArchivedTableResult(
                table_name="signal_decisions",
                row_count=5,
                archive_path=Path("archive/signal_decisions/year=2026/month=03/data.parquet"),
            ),
        ),
    )
    restore_result = RestoreBatchResult(
        archive_month="2026-03",
        table_results=(
            RestoreTableResult(
                table_name="signal_decisions",
                row_count=4,
                archive_path=Path("archive/signal_decisions/year=2026/month=03/data.parquet"),
            ),
        ),
    )
    retention_plan = RetentionPlan(
        generated_at_utc=datetime(2026, 4, 5, tzinfo=UTC),
        candidates=(),
    )

    monkeypatch.setattr(script, "SQLiteLiveDataStore", _FakeStore)
    monkeypatch.setattr(script, "archive_calendar_month", lambda *args, **kwargs: archive_result)
    monkeypatch.setattr(script, "_verify_archive_restore", lambda **kwargs: restore_result)
    monkeypatch.setattr(script, "build_retention_plan", lambda *args, **kwargs: retention_plan)

    def _unexpected_purge(*args, **kwargs):
        raise AssertionError("purge_retained_data should not be called when verification fails.")

    monkeypatch.setattr(script, "purge_retained_data", _unexpected_purge)

    try:
        script.execute_monthly_archive_ops(
            db_path=ROOT / "tmp" / "fake.sqlite",
            archive_root=ROOT / "tmp" / "archive",
            year=2026,
            month=3,
            table_names=("signal_decisions",),
            retention_days=90,
            purge=True,
        )
    except RuntimeError as exc:
        message = str(exc)
    else:
        raise AssertionError("Expected purge to be blocked when archive verification fails.")

    assert "refusing to purge" in message
    assert "archived_rows=5 restored_rows=4" in message
