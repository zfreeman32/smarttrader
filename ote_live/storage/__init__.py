from importlib import import_module

from .archive import (
    archive_calendar_month,
    archive_path_for_table,
    has_archive_for_month,
    restore_calendar_month,
)
from .db import SQLiteLiveDataStore, StoredIngestionGap
from .repositories import (
    AUDIT_TABLE_SPECS,
    HealthEventRecord,
    LiveAuditRepository,
    MediaArtifactRecord,
    NotificationRecord,
    PersistedRuntimeManifest,
    SignalAuditTrail,
)
from .retention import (
    RetentionCandidate,
    RetentionExecutionResult,
    RetentionPlan,
    RetentionPolicy,
    build_retention_plan,
    purge_retained_data,
)

__all__ = [
    "AUDIT_TABLE_SPECS",
    "HealthEventRecord",
    "LiveAuditRepository",
    "MediaArtifactRecord",
    "NotificationRecord",
    "PersistedRuntimeManifest",
    "AuditReplayReport",
    "FeatureReplayMismatch",
    "RetentionCandidate",
    "RetentionExecutionResult",
    "RetentionPlan",
    "RetentionPolicy",
    "SQLiteLiveDataStore",
    "StoredIngestionGap",
    "SignalAuditTrail",
    "archive_calendar_month",
    "archive_path_for_table",
    "build_retention_plan",
    "has_archive_for_month",
    "purge_retained_data",
    "replay_audited_signal",
    "restore_calendar_month",
]


def __getattr__(name: str):
    if name in {"AuditReplayReport", "FeatureReplayMismatch", "replay_audited_signal"}:
        module = import_module(".replay", __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
