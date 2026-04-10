from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from email.message import EmailMessage
import os
from pathlib import Path
import smtplib
from typing import Any, Mapping, Protocol, Sequence
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from ote_live.alerts.throttling import (
    NotificationThrottle,
    NotificationThrottlePolicy,
    build_model_cooldown_scope,
    build_signal_dedupe_key,
)
from ote_live.ingestion.base import ensure_utc, utc_now
from ote_live.storage import LiveAuditRepository, SignalAuditTrail


@dataclass(frozen=True)
class EmailAlertMessage:
    recipients: tuple[str, ...]
    subject: str
    body_text: str


@dataclass(frozen=True)
class EmailDispatchResult:
    status: str
    channel: str
    dedupe_key: str
    notification_id: int
    reason: str | None = None
    transport_response: str | None = None


class EmailTransport(Protocol):
    def send_email(self, message: EmailAlertMessage) -> str | None:
        """Send an email alert and optionally return a provider message id."""


class NoOpEmailTransport:
    def send_email(self, message: EmailAlertMessage) -> str | None:
        return f"noop-email:{len(message.recipients)}"


class SmtpEmailTransport:
    def __init__(
        self,
        *,
        host: str,
        port: int,
        sender: str,
        username: str | None = None,
        password: str | None = None,
        use_tls: bool = True,
        timeout_seconds: float = 30.0,
    ) -> None:
        self.host = str(host)
        self.port = int(port)
        self.sender = str(sender)
        self.username = username
        self.password = password
        self.use_tls = bool(use_tls)
        self.timeout_seconds = float(timeout_seconds)

    @classmethod
    def from_environment(
        cls,
        *,
        prefix: str = "OTE_ALERT_EMAIL",
    ) -> "SmtpEmailTransport":
        host = os.getenv(f"{prefix}_SMTP_HOST")
        sender = os.getenv(f"{prefix}_SENDER")
        if not host or not sender:
            raise ValueError(
                f"Missing {prefix}_SMTP_HOST or {prefix}_SENDER for SMTP alert delivery."
            )
        port = int(os.getenv(f"{prefix}_SMTP_PORT", "587"))
        username = os.getenv(f"{prefix}_SMTP_USERNAME")
        password = os.getenv(f"{prefix}_SMTP_PASSWORD")
        use_tls = os.getenv(f"{prefix}_SMTP_USE_TLS", "1").strip().lower() not in {"0", "false", "no"}
        timeout_seconds = float(os.getenv(f"{prefix}_SMTP_TIMEOUT_SECONDS", "30"))
        return cls(
            host=host,
            port=port,
            sender=sender,
            username=username,
            password=password,
            use_tls=use_tls,
            timeout_seconds=timeout_seconds,
        )

    def send_email(self, message: EmailAlertMessage) -> str | None:
        payload = EmailMessage()
        payload["From"] = self.sender
        payload["To"] = ", ".join(message.recipients)
        if message.subject:
            payload["Subject"] = message.subject
        payload.set_content(message.body_text)

        with smtplib.SMTP(self.host, self.port, timeout=self.timeout_seconds) as client:
            if self.use_tls:
                client.starttls()
            if self.username:
                client.login(self.username, self.password or "")
            refused = client.send_message(payload)
        return "smtp-accepted" if not refused else "smtp-partial-refusal"


class LiveSignalEmailer:
    def __init__(
        self,
        *,
        audit_repository: LiveAuditRepository,
        throttle: NotificationThrottle,
        transport: EmailTransport,
        recipients: Sequence[str],
        policy: NotificationThrottlePolicy | None = None,
    ) -> None:
        self.audit_repository = audit_repository
        self.throttle = throttle
        self.transport = transport
        self.recipients = tuple(recipients)
        self.policy = policy or NotificationThrottlePolicy()

    def send_signal_alert(
        self,
        *,
        signal_decision_id: int,
        recipients: Sequence[str] | None = None,
        dashboard_url: str | None = None,
        screenshot_path: str | Path | None = None,
        dedupe_key: str | None = None,
        cooldown_scope: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        now: datetime | None = None,
    ) -> EmailDispatchResult:
        audit_trail = self.audit_repository.reconstruct_signal(
            signal_decision_id,
            include_bars=False,
        )
        resolved_now = ensure_utc(now or utc_now())
        resolved_recipients = tuple(recipients or self.recipients)
        resolved_dedupe_key = dedupe_key or build_signal_dedupe_key(
            channel="email",
            signal_decision_id=signal_decision_id,
        )
        resolved_cooldown_scope = cooldown_scope or build_model_cooldown_scope(
            audit_trail.signal.model_id,
            audit_trail.signal.direction,
            audit_trail.signal.decision,
        )

        message = render_signal_email_message(
            audit_trail,
            recipients=resolved_recipients,
            dashboard_url=dashboard_url,
            screenshot_path=screenshot_path,
        )
        payload = _build_notification_payload(
            audit_trail,
            message,
            dashboard_url=dashboard_url,
            screenshot_path=screenshot_path,
        )
        base_metadata = dict(metadata or {})
        base_metadata.update(
            {
                "recipients": list(resolved_recipients),
                "cooldown_scope": resolved_cooldown_scope,
            }
        )

        throttle_decision = self.throttle.evaluate(
            channel="email",
            dedupe_key=resolved_dedupe_key,
            cooldown_scope=resolved_cooldown_scope,
            policy=self.policy,
            now=resolved_now,
        )
        if not throttle_decision.allowed:
            base_metadata["throttle_reason"] = throttle_decision.reason
            if throttle_decision.cooldown_remaining_seconds is not None:
                base_metadata["cooldown_remaining_seconds"] = throttle_decision.cooldown_remaining_seconds
            notification_id = self.audit_repository.record_notification(
                signal_decision_id=signal_decision_id,
                channel="email",
                status="skipped",
                dedupe_key=resolved_dedupe_key,
                payload=payload,
                metadata=base_metadata,
                error_message=throttle_decision.reason,
                sent_at_utc=resolved_now,
            )
            return EmailDispatchResult(
                status="skipped",
                channel="email",
                dedupe_key=resolved_dedupe_key,
                notification_id=notification_id,
                reason=throttle_decision.reason,
                transport_response=None,
            )

        status = "sent"
        error_message: str | None = None
        transport_response: str | None = None
        try:
            transport_response = self.transport.send_email(message)
        except Exception as exc:  # pragma: no cover - transport behavior is caller-specific
            status = "failed"
            error_message = str(exc)

        notification_id = self.audit_repository.record_notification(
            signal_decision_id=signal_decision_id,
            channel="email",
            status=status,
            dedupe_key=resolved_dedupe_key,
            payload=payload,
            metadata=base_metadata,
            error_message=error_message,
            sent_at_utc=resolved_now,
        )
        if status == "sent":
            self.throttle.remember_sent(
                channel="email",
                dedupe_key=resolved_dedupe_key,
                cooldown_scope=resolved_cooldown_scope,
                signal_decision_id=signal_decision_id,
                policy=self.policy,
                now=resolved_now,
            )

        return EmailDispatchResult(
            status=status,
            channel="email",
            dedupe_key=resolved_dedupe_key,
            notification_id=notification_id,
            reason=error_message,
            transport_response=transport_response,
        )


def render_signal_email_message(
    audit_trail: SignalAuditTrail,
    *,
    recipients: Sequence[str],
    dashboard_url: str | None = None,
    screenshot_path: str | Path | None = None,
) -> EmailAlertMessage:
    signal = audit_trail.signal
    prediction = audit_trail.prediction
    runtime_manifest = audit_trail.runtime_manifest.manifest if audit_trail.runtime_manifest is not None else None
    signal_dashboard_url = _dashboard_signal_url(
        dashboard_url,
        signal_decision_id=audit_trail.signal_decision_id,
    )
    subject = (
        f"[OTE][{signal.direction.upper()}][{signal.decision.upper()}] "
        f"{signal.probability:.2%} @ {signal.timestamp:%Y-%m-%d %H:%M UTC}"
    )
    lines = [
        f"Signal decision id: {audit_trail.signal_decision_id}",
        f"Timestamp (UTC): {signal.timestamp.isoformat()}",
        f"Direction: {signal.direction}",
        f"Decision: {signal.decision}",
        f"Model: {signal.model_id}",
        f"Backend: {prediction.backend}",
        f"Probability: {signal.probability:.4f}",
        f"Threshold: {_format_optional_float(signal.threshold)}",
        f"Regime: {signal.regime or 'unknown'}",
        f"Source row idx: {signal.source_row_idx}",
        f"Reasons: {', '.join(signal.reasons) if signal.reasons else 'none'}",
    ]
    if runtime_manifest is not None:
        lines.extend(
            [
                f"Asset/timeframe: {runtime_manifest.asset} {runtime_manifest.timeframe}",
                f"Role/status: {runtime_manifest.role} / {runtime_manifest.status}",
            ]
        )
    if signal_dashboard_url:
        lines.append(f"Dashboard: {signal_dashboard_url}")
    if screenshot_path:
        lines.append(f"Screenshot: {Path(screenshot_path)}")

    return EmailAlertMessage(
        recipients=tuple(recipients),
        subject=subject,
        body_text="\n".join(lines),
    )


def _build_notification_payload(
    audit_trail: SignalAuditTrail,
    message: EmailAlertMessage,
    *,
    dashboard_url: str | None,
    screenshot_path: str | Path | None,
) -> dict[str, Any]:
    signal_dashboard_url = _dashboard_signal_url(
        dashboard_url,
        signal_decision_id=audit_trail.signal_decision_id,
    )
    return {
        "signal_decision_id": audit_trail.signal_decision_id,
        "subject": message.subject,
        "body_text": message.body_text,
        "recipients": list(message.recipients),
        "model_id": audit_trail.signal.model_id,
        "direction": audit_trail.signal.direction,
        "decision": audit_trail.signal.decision,
        "probability": audit_trail.signal.probability,
        "threshold": audit_trail.signal.threshold,
        "dashboard_url": signal_dashboard_url,
        "screenshot_path": str(screenshot_path) if screenshot_path is not None else None,
    }


def _format_optional_float(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.4f}"


def _dashboard_signal_url(base_url: str | None, *, signal_decision_id: int) -> str | None:
    if not base_url:
        return None
    split = urlsplit(base_url)
    query = dict(parse_qsl(split.query, keep_blank_values=True))
    query["signal_decision_id"] = str(int(signal_decision_id))
    return urlunsplit((split.scheme, split.netloc, split.path, urlencode(query), split.fragment))
