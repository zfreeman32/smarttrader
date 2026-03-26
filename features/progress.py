from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from threading import RLock
from typing import Callable, Dict, Iterator


@dataclass(frozen=True)
class FeatureBuildProgressEvent:
    stage: str
    action: str
    name: str = ""
    index: int | None = None
    total: int | None = None
    elapsed_seconds: float | None = None
    message: str | None = None
    metrics: Dict[str, object] = field(default_factory=dict)


ProgressCallback = Callable[[FeatureBuildProgressEvent], None]


_callback_stack: list[ProgressCallback | None] = []
_callback_lock = RLock()


@contextmanager
def progress_context(callback: ProgressCallback | None) -> Iterator[None]:
    with _callback_lock:
        _callback_stack.append(callback)
    try:
        yield
    finally:
        with _callback_lock:
            _callback_stack.pop()


def current_progress_callback() -> ProgressCallback | None:
    with _callback_lock:
        if not _callback_stack:
            return None
        return _callback_stack[-1]


def emit_progress(event: FeatureBuildProgressEvent) -> None:
    callback = current_progress_callback()
    if callback is None:
        return
    callback(event)


def report_progress(
    *,
    stage: str,
    action: str,
    name: str = "",
    index: int | None = None,
    total: int | None = None,
    elapsed_seconds: float | None = None,
    message: str | None = None,
    **metrics: object,
) -> None:
    emit_progress(
        FeatureBuildProgressEvent(
            stage=stage,
            action=action,
            name=name,
            index=index,
            total=total,
            elapsed_seconds=elapsed_seconds,
            message=message,
            metrics=metrics,
        )
    )
