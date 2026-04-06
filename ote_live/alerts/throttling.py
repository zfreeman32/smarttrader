from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from ote_live.ingestion.base import ensure_utc, utc_now
from ote_live.storage import SQLiteLiveDataStore


@dataclass(frozen=True)
class NotificationThrottlePolicy:
    cooldown_seconds: float = 0.0
    dedupe_enabled: bool = True


@dataclass(frozen=True)
class NotificationThrottleDecision:
    allowed: bool
    reason: str
    cooldown_remaining_seconds: float | None = None


class NotificationThrottle:
    def __init__(self, store: SQLiteLiveDataStore) -> None:
        self.store = store

    def evaluate(
        self,
        *,
        channel: str,
        dedupe_key: str | None,
        cooldown_scope: str | None,
        policy: NotificationThrottlePolicy | None = None,
        now: datetime | None = None,
    ) -> NotificationThrottleDecision:
        resolved_policy = policy or NotificationThrottlePolicy()
        resolved_now = ensure_utc(now or utc_now())

        if resolved_policy.dedupe_enabled and dedupe_key and self._notification_already_sent(channel, dedupe_key):
            return NotificationThrottleDecision(
                allowed=False,
                reason="duplicate_dedupe_key",
                cooldown_remaining_seconds=None,
            )

        if cooldown_scope and resolved_policy.cooldown_seconds > 0:
            state = self.store.get_runtime_state(
                scope="alert_throttle",
                state_key=_cooldown_state_key(channel, cooldown_scope),
            )
            if state:
                last_sent_at = _parse_iso_datetime(state.get("last_sent_at_utc"))
                if last_sent_at is not None:
                    elapsed_seconds = (resolved_now - last_sent_at).total_seconds()
                    remaining_seconds = float(resolved_policy.cooldown_seconds) - elapsed_seconds
                    if remaining_seconds > 0:
                        return NotificationThrottleDecision(
                            allowed=False,
                            reason="cooldown_active",
                            cooldown_remaining_seconds=max(remaining_seconds, 0.0),
                        )

        return NotificationThrottleDecision(
            allowed=True,
            reason="allowed",
            cooldown_remaining_seconds=None,
        )

    def remember_sent(
        self,
        *,
        channel: str,
        dedupe_key: str | None,
        cooldown_scope: str | None,
        signal_decision_id: int | None = None,
        policy: NotificationThrottlePolicy | None = None,
        now: datetime | None = None,
    ) -> None:
        resolved_policy = policy or NotificationThrottlePolicy()
        if not cooldown_scope or resolved_policy.cooldown_seconds <= 0:
            return

        resolved_now = ensure_utc(now or utc_now())
        self.store.upsert_runtime_state(
            scope="alert_throttle",
            state_key=_cooldown_state_key(channel, cooldown_scope),
            payload={
                "channel": channel,
                "cooldown_scope": cooldown_scope,
                "dedupe_key": dedupe_key,
                "signal_decision_id": signal_decision_id,
                "last_sent_at_utc": resolved_now.isoformat(),
                "cooldown_seconds": float(resolved_policy.cooldown_seconds),
            },
        )

    def _notification_already_sent(self, channel: str, dedupe_key: str) -> bool:
        row = self.store.connection.execute(
            """
            SELECT id
            FROM notifications
            WHERE channel = ?
              AND dedupe_key = ?
              AND status IN ('queued', 'sent', 'delivered', 'success')
            LIMIT 1
            """,
            (channel, dedupe_key),
        ).fetchone()
        return row is not None


def build_signal_dedupe_key(*, channel: str, signal_decision_id: int) -> str:
    return f"{channel}:signal:{int(signal_decision_id)}"


def build_model_cooldown_scope(model_id: str, direction: str, decision: str) -> str:
    return f"{model_id}:{direction}:{decision}"


def _cooldown_state_key(channel: str, cooldown_scope: str) -> str:
    return f"{channel}:{cooldown_scope}"


def _parse_iso_datetime(value: object) -> datetime | None:
    if not value or not isinstance(value, str):
        return None
    try:
        return ensure_utc(datetime.fromisoformat(value))
    except ValueError:
        return None
