"""Reobserve rule setups after late source corrections without rerunning trades."""
from __future__ import annotations

from contextlib import nullcontext
from datetime import datetime
from typing import Any

from ote_live.features.incremental_engine import IncrementalFeatureEngine
from ote_live.ingestion.base import ensure_utc, utc_now
from ote_live.storage.setup_events import _bar_version


_SCOPE = "setup_source_revision_cursor"
_EVALUATED_SCOPE = "setup_evaluated_bar_versions"


def mark_setup_evaluated(processor: Any, bar: Any) -> None:
    """Record successful zero/nonzero setup evaluation for one source version."""
    processor.audit_repository.store.upsert_runtime_state(
        scope=_EVALUATED_SCOPE, state_key=_state_key(processor) + ":" + ensure_utc(bar.timestamp).isoformat(),
        payload={"bar_version": str(_bar_version(bar)), "evaluated_at_utc": utc_now().isoformat()},
    )


def initialize_setup_revision_cursor(processor: Any) -> dict[str, Any]:
    """Watermark precollection history while retaining a restart-safe cursor."""
    store = processor.audit_repository.store
    key = _state_key(processor)
    existing = store.get_runtime_state(scope=_SCOPE, state_key=key)
    if existing is not None:
        return existing
    retained = store.connection.execute(
        "SELECT 1 FROM setup_events WHERE asset=? AND timeframe=? AND collection_version=? LIMIT 1",
        (processor.asset, processor.timeframe, processor.collection_version),
    ).fetchone()
    # A runtime upgraded after collecting setups may not yet have a cursor.
    # Replay its source revisions rather than silently skipping that interval.
    cursor = 0 if retained else _latest_history_id(processor)
    state = {"history_id": cursor, "initialized_at_utc": utc_now().isoformat(), "pending": None}
    store.upsert_runtime_state(scope=_SCOPE, state_key=key, payload=state)
    return state


def reconcile_setup_revisions(processor: Any, *, max_evaluations: int = 128) -> int:
    """Append corrected setups from causal prefixes, including prior no-fire bars.

    Call serially before normal signal processing, outside parallel feature jobs.
    The persistent pending range advances only after each successful append, so
    a failure or process restart cannot lose invalidations or backfilled setups.
    Each prefix stops at its target source timestamp. Its new observation time
    is the actual reconciliation time; no model, decision, notification or paper
    ledger is invoked. Long correction ranges continue on subsequent calls.
    """
    if max_evaluations <= 0:
        raise ValueError("max_evaluations must be positive")
    state = initialize_setup_revision_cursor(processor)
    last_processed = processor.last_processed_timestamp
    strategies = _strategies(processor)
    if last_processed is None or not strategies:
        return 0
    store = processor.audit_repository.store
    repository = processor.setup_event_repository
    last_timestamp = ensure_utc(last_processed).isoformat()
    pending = state.get("pending")
    if not pending:
        history = store.connection.execute(
            "SELECT id,timestamp_utc,bar_version,event_type FROM source_bar_history "
            "WHERE asset=? AND timeframe=? AND id>? ORDER BY id",
            (processor.asset, processor.timeframe, int(state["history_id"])),
        ).fetchall()
        deferred = state.get("deferred_observations") or []
        if not history and not deferred:
            return 0
        history_upper_id = int(history[-1]["id"]) if history else int(state["history_id"])
        newest = {row["timestamp_utc"]: row for row in (*deferred, *history)}
        affected: list[str] = []
        still_deferred: list[dict[str, Any]] = []
        for timestamp, row in newest.items():
            if timestamp > last_timestamp:
                continue
            evaluated = store.get_runtime_state(scope=_EVALUATED_SCOPE, state_key=_state_key(processor) + ":" + timestamp)
            if evaluated is not None and evaluated.get("bar_version") == row["bar_version"]:
                continue
            canonical = store.connection.execute(
                "SELECT bar_version FROM canonical_bars WHERE asset=? AND timeframe=? AND timestamp_utc=?",
                (processor.asset, processor.timeframe, timestamp),
            ).fetchone()
            # A provisional callback can be observed before canonical publication.
            # The complete canonical update has its own source-history record.
            if canonical is None or canonical["bar_version"] != row["bar_version"]:
                still_deferred.append(dict(row))
                continue
            affected.append(timestamp)
        state["deferred_observations"] = still_deferred
        if not affected:
            state["history_id"] = history_upper_id
            _save(processor, state)
            return 0
        pending = {"start_timestamp_utc": min(affected), "after_timestamp_utc": None,
                   "end_timestamp_utc": last_timestamp, "history_id": history_upper_id}
        state["pending"] = pending
        _save(processor, state)
        _refresh_live_state(processor, last_processed)

    query = (
        "SELECT timestamp_utc FROM canonical_bars WHERE asset=? AND timeframe=? "
        "AND timestamp_utc>=? AND timestamp_utc<=?"
    )
    parameters: list[Any] = [processor.asset, processor.timeframe, pending["start_timestamp_utc"],
                             pending["end_timestamp_utc"]]
    if pending.get("after_timestamp_utc"):
        query += " AND timestamp_utc>?"
        parameters.append(pending["after_timestamp_utc"])
    targets = store.connection.execute(query + " ORDER BY timestamp_utc LIMIT ?",
                                       [*parameters, max_evaluations + 1]).fetchall()
    evaluated = 0
    for target in targets[:max_evaluations]:
        timestamp = datetime.fromisoformat(target["timestamp_utc"])
        prefix_bars = _prefix_bars(processor, timestamp)
        if not prefix_bars:
            raise RuntimeError(f"Missing canonical source prefix for setup revision at {timestamp.isoformat()}")
        market_frame = _market_frame(prefix_bars)
        features = processor.feature_engine.build_feature_frame(market_frame, include_policy_features=True)
        if features.empty:
            raise RuntimeError(f"Empty feature producer for setup revision at {timestamp.isoformat()}")
        policy = processor._build_policy_frame_from_market_frame(features, market_frame=market_frame)
        observed_at = utc_now()
        for strategy in strategies:
            repository.collect_from_frame(
                policy, bar=prefix_bars[-1], collection_version=processor.collection_version,
                observed_at=observed_at, source_bar_version=prefix_bars[-1].bar_version,
                strategies=(strategy,),
            )
        repository.record_followup_bar(prefix_bars[-1], collection_version=processor.collection_version,
                                       observed_at=observed_at, source_bar_version=prefix_bars[-1].bar_version)
        mark_setup_evaluated(processor, prefix_bars[-1])
        pending["after_timestamp_utc"] = target["timestamp_utc"]
        _save(processor, state)
        evaluated += 1
    if len(targets) <= max_evaluations:
        state["history_id"] = int(pending["history_id"])
        state["pending"] = None
        _save(processor, state)
    return evaluated


def _prefix_bars(processor, end: datetime):
    store = processor.audit_repository.store
    history_bars = int(processor.feature_engine.state.max_history_bars)
    rows = store.connection.execute(
        "SELECT timestamp_utc FROM canonical_bars WHERE asset=? AND timeframe=? AND timestamp_utc<=? "
        "ORDER BY timestamp_utc DESC LIMIT ?",
        (processor.asset, processor.timeframe, ensure_utc(end).isoformat(), history_bars),
    ).fetchall()
    if not rows:
        return []
    return store.fetch_bars(asset=processor.asset, timeframe=processor.timeframe,
                            start=datetime.fromisoformat(rows[-1]["timestamp_utc"]), end=end)


def _market_frame(bars):
    frame = IncrementalFeatureEngine.market_frame_from_bars(bars)
    # The ordinary runtime state includes externally supplied feature context.
    # Preserve it here too; omitting it would manufacture producer regressions.
    keys = {key for bar in bars for key in bar.feature_context if key not in frame.columns}
    for key in keys:
        frame[key] = [bar.feature_context.get(key) for bar in bars]
    return frame


def _refresh_live_state(processor, last_processed: datetime) -> None:
    engine = processor.feature_engine
    bars = _prefix_bars(processor, last_processed)
    # State ingestion forbids older timestamps. Replace its retained window
    # directly, preserving seed offsets and the last processed signal timestamp.
    with getattr(engine, "_cache_lock", nullcontext()):
        engine.state.clear()
        engine.state.extend(bars)
        invalidate = getattr(engine, "_invalidate_cache", None)
        if callable(invalidate):
            invalidate()


def _strategies(processor) -> tuple[str, ...]:
    return tuple(strategy for strategy, flag in (
        ("FRVP", "_frvp_dashboard_state_enabled"), ("ICT", "_ict_dashboard_state_enabled")
    ) if getattr(processor, flag, str(getattr(processor, "group_name", "")).upper() == strategy))


def _latest_history_id(processor) -> int:
    row = processor.audit_repository.store.connection.execute(
        "SELECT COALESCE(MAX(id),0) AS id FROM source_bar_history WHERE asset=? AND timeframe=?",
        (processor.asset, processor.timeframe),
    ).fetchone()
    return int(row["id"])


def _state_key(processor) -> str:
    return ":".join((processor.asset, processor.timeframe, str(getattr(processor, "group_name", "")),
                     processor.collection_version))


def _save(processor, state) -> None:
    processor.audit_repository.store.upsert_runtime_state(scope=_SCOPE, state_key=_state_key(processor), payload=state)
