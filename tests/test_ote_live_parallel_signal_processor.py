from __future__ import annotations

import time
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pandas as pd

from ote_live.contracts.market_data import MarketBar
from ote_live.ingestion.signals import (
    MarketDataSnapshot,
    MultiGroupLiveSignalProcessor,
    PreparedSignalSnapshot,
    RuntimeSignalResult,
    build_market_snapshot_id,
)


def test_parallel_multigroup_matches_sequential_outputs_and_snapshot_identity() -> None:
    bars = [_bar(0), _bar(5)]
    legacy_processors = _processors(("OTE", "FRVP", "ICT"), bars=bars)
    parallel_processors = _processors(("OTE", "FRVP", "ICT"), bars=bars)
    target_bar = bars[-1]
    source_row_idx = 1
    snapshot_id = build_market_snapshot_id(
        asset=target_bar.asset,
        timeframe=target_bar.timeframe,
        timestamp=target_bar.timestamp,
        source_row_idx=source_row_idx,
    )

    legacy_results = _run_legacy_sequential(
        legacy_processors,
        target_bar,
        snapshot_id=snapshot_id,
        generation_id=1,
        source_row_idx=source_row_idx,
    )
    parallel = MultiGroupLiveSignalProcessor(parallel_processors, max_workers=3)
    try:
        parallel_results = parallel.process_bars([target_bar], emit_operator_artifacts=False)
    finally:
        parallel.close()

    assert _result_contract(legacy_results) == _result_contract(parallel_results)
    assert {result.snapshot_id for result in parallel_results} == {snapshot_id}
    assert {result.generation_id for result in parallel_results} == {1}


def test_parallel_multigroup_fans_in_out_of_order_completions_before_publish() -> None:
    bars = [_bar(0), _bar(5)]
    completed: set[str] = set()
    publish_completion_counts: list[int] = []
    processors = _processors(
        ("OTE", "FRVP", "ICT"),
        bars=bars,
        delays={"OTE": 0.03, "FRVP": 0.01, "ICT": 0.02},
        completed=completed,
        publish_completion_counts=publish_completion_counts,
    )
    parallel = MultiGroupLiveSignalProcessor(processors, max_workers=3)

    try:
        results = parallel.process_bars([bars[-1]], emit_operator_artifacts=False)
    finally:
        parallel.close()

    assert [result.model_id for result in results] == ["OTE_model", "FRVP_model", "ICT_model"]
    assert publish_completion_counts == [3, 3, 3]


def test_parallel_multigroup_rejects_stale_branch_generation_without_publishing() -> None:
    bars = [_bar(0), _bar(5)]
    processors = _processors(
        ("OTE", "FRVP", "ICT"),
        bars=bars,
        stale_generation_groups={"FRVP"},
    )
    parallel = MultiGroupLiveSignalProcessor(processors, max_workers=3)

    try:
        results = parallel.process_bars([bars[-1]], emit_operator_artifacts=False)
    finally:
        parallel.close()

    assert results == ()
    assert all(processor.published_snapshot_ids == [] for processor in processors)
    assert processors[0].health_events[-1]["event_type"] == "snapshot_consistency_failed"


def test_parallel_multigroup_branch_failure_keeps_previous_complete_state() -> None:
    bars = [_bar(0), _bar(5)]
    processors = _processors(
        ("OTE", "FRVP", "ICT"),
        bars=bars,
        failing_groups={"ICT"},
    )
    parallel = MultiGroupLiveSignalProcessor(processors, max_workers=3)

    try:
        results = parallel.process_bars([bars[-1]], emit_operator_artifacts=False)
    finally:
        parallel.close()

    assert results == ()
    assert all(processor.published_snapshot_ids == [] for processor in processors)
    assert any(event["event_type"] == "feature_branch_failed" for event in processors[2].health_events)
    assert any(event["event_type"] == "snapshot_fan_in_incomplete" for event in processors[0].health_events)
    assert [processor.last_processed_timestamp for processor in processors] == [None, None, None]


def test_parallel_multigroup_retries_failed_branch_from_previous_bar() -> None:
    bars = [_bar(0), _bar(5)]
    processors = _processors(
        ("OTE", "FRVP", "ICT"),
        bars=bars,
        failing_groups={"ICT"},
    )
    for processor in processors:
        processor.feature_engine.state.extend([bars[0]])
        processor.last_processed_timestamp = bars[0].timestamp
    parallel = MultiGroupLiveSignalProcessor(processors, max_workers=3)

    try:
        first = parallel.process_new_bars_from_store(
            emit_operator_artifacts=False,
            max_timestamp=bars[-1].timestamp,
        )
        processors[2].should_fail = False
        second = parallel.process_new_bars_from_store(
            emit_operator_artifacts=False,
            max_timestamp=bars[-1].timestamp,
        )
    finally:
        parallel.close()

    assert first == ()
    assert len(second) == 3
    assert [processor.prepare_count for processor in processors] == [2, 2, 2]
    assert [processor.last_processed_timestamp for processor in processors] == [
        bars[-1].timestamp,
        bars[-1].timestamp,
        bars[-1].timestamp,
    ]


def test_parallel_multigroup_fetches_new_bars_once_and_does_not_reprepare_without_new_bar() -> None:
    bars = [_bar(0), _bar(5)]
    processors = _processors(("OTE", "FRVP", "ICT"), bars=bars)
    for processor in processors:
        processor.feature_engine.state.extend([bars[0]])
        processor.last_processed_timestamp = bars[0].timestamp
    parallel = MultiGroupLiveSignalProcessor(processors, max_workers=3)

    try:
        first = parallel.process_new_bars_from_store(
            emit_operator_artifacts=False,
            max_timestamp=bars[-1].timestamp,
        )
        second = parallel.process_new_bars_from_store(
            emit_operator_artifacts=False,
            max_timestamp=bars[-1].timestamp,
        )
    finally:
        parallel.close()

    assert len(first) == 3
    assert second == ()
    assert processors[0].fetch_after_calls == 2
    assert processors[1].fetch_after_calls == 0
    assert processors[2].fetch_after_calls == 0
    assert [processor.prepare_count for processor in processors] == [1, 1, 1]


def test_parallel_multigroup_feature_timing_beats_legacy_sequential_baseline() -> None:
    bars = [_bar(0), _bar(5)]
    delays = {"OTE": 0.03, "FRVP": 0.03, "ICT": 0.03}
    legacy_processors = _processors(("OTE", "FRVP", "ICT"), bars=bars, delays=delays)
    parallel_processors = _processors(("OTE", "FRVP", "ICT"), bars=bars, delays=delays)
    target_bar = bars[-1]
    source_row_idx = 1
    snapshot_id = build_market_snapshot_id(
        asset=target_bar.asset,
        timeframe=target_bar.timeframe,
        timestamp=target_bar.timestamp,
        source_row_idx=source_row_idx,
    )

    sequential_started = time.perf_counter()
    _run_legacy_sequential(
        legacy_processors,
        target_bar,
        snapshot_id=snapshot_id,
        generation_id=1,
        source_row_idx=source_row_idx,
    )
    sequential_seconds = time.perf_counter() - sequential_started

    parallel = MultiGroupLiveSignalProcessor(parallel_processors, max_workers=3)
    try:
        parallel_started = time.perf_counter()
        parallel.process_bars([target_bar], emit_operator_artifacts=False)
        parallel_seconds = time.perf_counter() - parallel_started
    finally:
        parallel.close()

    assert parallel_seconds < sequential_seconds * 0.75


def test_parallel_multigroup_close_shuts_down_workers_and_children_once() -> None:
    processors = _processors(("OTE", "FRVP"), bars=[_bar(0)])
    parallel = MultiGroupLiveSignalProcessor(processors, max_workers=2)

    parallel.close()
    parallel.close()

    assert all(processor.close_count == 1 for processor in processors)


def _run_legacy_sequential(
    processors: tuple["_FakeSignalProcessor", ...],
    bar: MarketBar,
    *,
    snapshot_id: str,
    generation_id: int,
    source_row_idx: int,
) -> tuple[RuntimeSignalResult, ...]:
    results: list[RuntimeSignalResult] = []
    for processor in processors:
        processor.ingest_bar_for_evaluation(bar)
        prepared = processor.prepare_ingested_bar_snapshot(
            bar,
            snapshot_id=snapshot_id,
            generation_id=generation_id,
            source_row_idx=source_row_idx,
        )
        assert prepared is not None
        results.extend(
            processor.publish_prepared_snapshot(
                prepared,
                emit_operator_artifacts=False,
                emit_notifications=True,
                record_paper_signal_events=True,
            )
        )
    return tuple(results)


def _processors(
    names: tuple[str, ...],
    *,
    bars: list[MarketBar],
    delays: dict[str, float] | None = None,
    completed: set[str] | None = None,
    publish_completion_counts: list[int] | None = None,
    failing_groups: set[str] | None = None,
    stale_generation_groups: set[str] | None = None,
) -> tuple["_FakeSignalProcessor", ...]:
    audit_repository = object()
    return tuple(
        _FakeSignalProcessor(
            name,
            audit_repository=audit_repository,
            bars=bars,
            delay_seconds=(delays or {}).get(name, 0.0),
            completed=completed,
            publish_completion_counts=publish_completion_counts,
            should_fail=name in (failing_groups or set()),
            stale_generation=name in (stale_generation_groups or set()),
        )
        for name in names
    )


class _FakeSignalProcessor:
    def __init__(
        self,
        group_name: str,
        *,
        audit_repository: object,
        bars: list[MarketBar],
        delay_seconds: float = 0.0,
        completed: set[str] | None = None,
        publish_completion_counts: list[int] | None = None,
        should_fail: bool = False,
        stale_generation: bool = False,
    ) -> None:
        self.group_name = group_name
        self.asset = "ES"
        self.timeframe = "5m"
        self.audit_repository = audit_repository
        self.bindings = (SimpleNamespace(loaded_model=SimpleNamespace(model_id=f"{group_name}_model")),)
        self.feature_engine = SimpleNamespace(
            state=_FakeFeatureState(),
            plan=SimpleNamespace(runtime_history_bars=2),
        )
        self.last_processed_timestamp = None
        self.bars = list(bars)
        self.delay_seconds = float(delay_seconds)
        self.completed = completed
        self.publish_completion_counts = publish_completion_counts
        self.should_fail = should_fail
        self.stale_generation = stale_generation
        self.health_events: list[dict[str, object]] = []
        self.published_snapshot_ids: list[str] = []
        self.prepare_count = 0
        self.fetch_after_calls = 0
        self.close_count = 0

    def warm_from_store(self) -> int:
        self.feature_engine.state.extend(self.bars)
        self.last_processed_timestamp = self.bars[-1].timestamp
        return len(self.bars)

    def seed_latest_predictions_from_store(self):
        return ()

    def _reconcile_frvp_paper_signal_events(self) -> None:
        return None

    def _fetch_recent_signal_bars(self, *, limit: int) -> list[MarketBar]:
        return self.bars[-int(limit) :]

    def _fetch_signal_bars_after(self, timestamp, *, max_timestamp=None) -> list[MarketBar]:
        self.fetch_after_calls += 1
        return [
            bar
            for bar in self.bars
            if bar.timestamp > timestamp
            and (max_timestamp is None or bar.timestamp <= max_timestamp)
        ]

    def _resolve_source_row_idx(self, bar: MarketBar) -> int:
        return self.bars.index(bar)

    def _existing_prediction_model_ids(self, timestamp) -> set[str]:
        return set()

    def ingest_bar_for_evaluation(self, bar: MarketBar) -> bool:
        self.feature_engine.state.extend([bar])
        self.last_processed_timestamp = bar.timestamp
        return True

    def prepare_ingested_bar_snapshot(
        self,
        bar: MarketBar,
        *,
        snapshot_id: str | None = None,
        generation_id: int | None = None,
        source_row_idx: int | None = None,
    ) -> PreparedSignalSnapshot | None:
        self.prepare_count += 1
        if self.delay_seconds:
            time.sleep(self.delay_seconds)
        if self.should_fail:
            raise RuntimeError(f"{self.group_name} branch failed")
        if self.completed is not None:
            self.completed.add(self.group_name)
        resolved_source_row_idx = int(source_row_idx or 0)
        resolved_generation_id = int(generation_id or 0)
        if self.stale_generation:
            resolved_generation_id -= 1
        resolved_snapshot_id = snapshot_id or build_market_snapshot_id(
            asset=bar.asset,
            timeframe=bar.timeframe,
            timestamp=bar.timestamp,
            source_row_idx=resolved_source_row_idx,
        )
        market_frame = pd.DataFrame(
            {
                "asset": [bar.asset],
                "timeframe": [bar.timeframe],
                "timestamp": [bar.timestamp],
                "open": [bar.open],
                "high": [bar.high],
                "low": [bar.low],
                "close": [bar.close],
                "volume": [bar.volume],
            }
        )
        feature_frame = pd.DataFrame({"feature": [float(self.prepare_count)]})
        return PreparedSignalSnapshot(
            group_name=self.group_name,
            snapshot=MarketDataSnapshot(
                snapshot_id=resolved_snapshot_id,
                generation_id=resolved_generation_id,
                asset=bar.asset,
                timeframe=bar.timeframe,
                bar_timestamp=bar.timestamp,
                source_row_idx=resolved_source_row_idx,
                bar=bar,
                market_frame=market_frame,
            ),
            feature_frame=feature_frame,
            policy_frame=pd.concat([market_frame, feature_frame], axis=1),
            feature_metadata={"fake": True},
            timings_seconds={"feature_generation": self.delay_seconds},
            feature_cache_hit=False,
        )

    def publish_prepared_snapshot(
        self,
        prepared: PreparedSignalSnapshot,
        *,
        emit_operator_artifacts: bool,
        emit_notifications: bool,
        record_paper_signal_events: bool = False,
        bindings=None,
    ) -> tuple[RuntimeSignalResult, ...]:
        del emit_operator_artifacts, emit_notifications, record_paper_signal_events, bindings
        if self.publish_completion_counts is not None and self.completed is not None:
            self.publish_completion_counts.append(len(self.completed))
        self.published_snapshot_ids.append(prepared.snapshot.snapshot_id)
        return (
            RuntimeSignalResult(
                model_id=f"{self.group_name}_model",
                direction="long",
                decision="hold",
                shadow_mode=False,
                timestamp=prepared.snapshot.bar_timestamp,
                snapshot_id=prepared.snapshot.snapshot_id,
                generation_id=prepared.snapshot.generation_id,
                timings_seconds=dict(prepared.timings_seconds),
            ),
        )

    def _record_health_event(
        self,
        *,
        component: str,
        event_type: str,
        severity: str,
        message: str,
        payload: dict,
    ) -> None:
        self.health_events.append(
            {
                "component": component,
                "event_type": event_type,
                "severity": severity,
                "message": message,
                "payload": payload,
            }
        )

    def close(self) -> None:
        self.close_count += 1


class _FakeFeatureState:
    def __init__(self) -> None:
        self._bars: list[MarketBar] = []

    @property
    def latest_timestamp(self):
        return self._bars[-1].timestamp if self._bars else None

    def extend(self, bars: list[MarketBar]) -> None:
        self._bars.extend(bars)


def _result_contract(results: tuple[RuntimeSignalResult, ...]) -> list[tuple[object, ...]]:
    return [
        (
            result.model_id,
            result.direction,
            result.decision,
            result.shadow_mode,
            result.timestamp,
            result.snapshot_id,
            result.generation_id,
        )
        for result in results
    ]


def _bar(offset_minutes: int) -> MarketBar:
    timestamp = datetime(2026, 1, 2, 14, 30, tzinfo=UTC) + timedelta(minutes=offset_minutes)
    price = 4800.0 + (offset_minutes * 0.25)
    return MarketBar(
        asset="ES",
        timeframe="5m",
        timestamp=timestamp,
        open=price,
        high=price + 2.0,
        low=price - 2.0,
        close=price + 0.5,
        volume=1000.0 + offset_minutes,
        source="test",
    )
