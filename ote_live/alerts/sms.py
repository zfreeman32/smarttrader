from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import os
from typing import Any, Mapping, Protocol, Sequence

from ote_live.alerts.emailer import EmailAlertMessage, EmailTransport, SmtpEmailTransport
from ote_live.alerts.throttling import (
    NotificationThrottle,
    NotificationThrottlePolicy,
    build_model_cooldown_scope,
    build_signal_dedupe_key,
)
from ote_live.ingestion.base import ensure_utc, utc_now
from ote_live.storage import LiveAuditRepository, SignalAuditTrail


@dataclass(frozen=True)
class SmsAlertMessage:
    recipients: tuple[str, ...]
    body_text: str


@dataclass(frozen=True)
class SmsDispatchResult:
    status: str
    channel: str
    dedupe_key: str
    notification_id: int
    reason: str | None = None
    transport_response: str | None = None


class SmsTransport(Protocol):
    def send_sms(self, message: SmsAlertMessage) -> str | None:
        """Send an SMS alert and optionally return a provider message id."""


class NoOpSmsTransport:
    def send_sms(self, message: SmsAlertMessage) -> str | None:
        return f"noop-sms:{len(message.recipients)}"


class EmailGatewaySmsTransport:
    def __init__(
        self,
        *,
        email_transport: EmailTransport,
        subject: str = "",
    ) -> None:
        self.email_transport = email_transport
        self.subject = str(subject or "").strip()

    @classmethod
    def from_environment(
        cls,
        *,
        email_prefix: str = "OTE_ALERT_EMAIL",
        subject_env_var: str = "OTE_ALERT_SMS_EMAIL_SUBJECT",
    ) -> "EmailGatewaySmsTransport":
        return cls(
            email_transport=SmtpEmailTransport.from_environment(prefix=email_prefix),
            subject=os.getenv(subject_env_var, ""),
        )

    def send_sms(self, message: SmsAlertMessage) -> str | None:
        email_message = EmailAlertMessage(
            recipients=message.recipients,
            subject=self.subject,
            body_text=message.body_text,
        )
        return self.email_transport.send_email(email_message)


class TwilioSmsTransport:
    def __init__(
        self,
        *,
        account_sid: str,
        auth_token: str,
        from_number: str | None = None,
        messaging_service_sid: str | None = None,
    ) -> None:
        if not from_number and not messaging_service_sid:
            raise ValueError("TwilioSmsTransport requires either from_number or messaging_service_sid.")
        self.account_sid = account_sid
        self.auth_token = auth_token
        self.from_number = from_number
        self.messaging_service_sid = messaging_service_sid
        try:
            from twilio.rest import Client
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "SMS delivery requires the 'twilio' package."
            ) from exc
        self._client_cls = Client

    @classmethod
    def from_environment(
        cls,
        *,
        prefix: str = "OTE_ALERT_SMS",
    ) -> "TwilioSmsTransport":
        account_sid = os.getenv(f"{prefix}_ACCOUNT_SID")
        auth_token = os.getenv(f"{prefix}_AUTH_TOKEN")
        from_number = os.getenv(f"{prefix}_FROM_NUMBER")
        messaging_service_sid = os.getenv(f"{prefix}_MESSAGING_SERVICE_SID")
        if not account_sid or not auth_token:
            raise ValueError(
                f"Missing {prefix}_ACCOUNT_SID or {prefix}_AUTH_TOKEN for SMS alert delivery."
            )
        return cls(
            account_sid=account_sid,
            auth_token=auth_token,
            from_number=from_number,
            messaging_service_sid=messaging_service_sid,
        )

    def send_sms(self, message: SmsAlertMessage) -> str | None:
        client = self._client_cls(self.account_sid, self.auth_token)
        last_sid: str | None = None
        for recipient in message.recipients:
            payload = {
                "body": message.body_text,
                "to": recipient,
            }
            if self.messaging_service_sid:
                payload["messaging_service_sid"] = self.messaging_service_sid
            else:
                payload["from_"] = self.from_number
            response = client.messages.create(**payload)
            last_sid = getattr(response, "sid", None)
        return last_sid


class RoutedSmsTransport:
    def __init__(
        self,
        *,
        phone_transport: SmsTransport | None = None,
        email_gateway_transport: SmsTransport | None = None,
    ) -> None:
        if phone_transport is None and email_gateway_transport is None:
            raise ValueError("RoutedSmsTransport requires at least one underlying transport.")
        self.phone_transport = phone_transport
        self.email_gateway_transport = email_gateway_transport

    def send_sms(self, message: SmsAlertMessage) -> str | None:
        phone_recipients = tuple(
            recipient for recipient in message.recipients if not is_email_gateway_sms_recipient(recipient)
        )
        email_gateway_recipients = tuple(
            recipient for recipient in message.recipients if is_email_gateway_sms_recipient(recipient)
        )
        responses: list[str] = []

        if phone_recipients:
            if self.phone_transport is None:
                raise ValueError("Phone SMS recipients were supplied, but no phone transport is configured.")
            phone_response = self.phone_transport.send_sms(
                SmsAlertMessage(
                    recipients=phone_recipients,
                    body_text=message.body_text,
                )
            )
            if phone_response:
                responses.append(phone_response)

        if email_gateway_recipients:
            if self.email_gateway_transport is None:
                raise ValueError("Email gateway SMS recipients were supplied, but no email gateway transport is configured.")
            email_response = self.email_gateway_transport.send_sms(
                SmsAlertMessage(
                    recipients=email_gateway_recipients,
                    body_text=message.body_text,
                )
            )
            if email_response:
                responses.append(email_response)

        return ",".join(responses) if responses else None


class LiveSignalSmsSender:
    def __init__(
        self,
        *,
        audit_repository: LiveAuditRepository,
        throttle: NotificationThrottle,
        transport: SmsTransport,
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
        dedupe_key: str | None = None,
        cooldown_scope: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        now: datetime | None = None,
    ) -> SmsDispatchResult:
        audit_trail = self.audit_repository.reconstruct_signal(
            signal_decision_id,
            include_bars=False,
        )
        resolved_now = ensure_utc(now or utc_now())
        resolved_recipients = tuple(recipients or self.recipients)
        resolved_dedupe_key = dedupe_key or build_signal_dedupe_key(
            channel="sms",
            signal_decision_id=signal_decision_id,
        )
        resolved_cooldown_scope = cooldown_scope or build_model_cooldown_scope(
            audit_trail.signal.model_id,
            audit_trail.signal.direction,
            audit_trail.signal.decision,
        )
        message = render_signal_sms_message(
            audit_trail,
            recipients=resolved_recipients,
        )
        payload = {
            "signal_decision_id": audit_trail.signal_decision_id,
            "body_text": message.body_text,
            "recipients": list(message.recipients),
            "model_id": audit_trail.signal.model_id,
            "direction": audit_trail.signal.direction,
            "decision": audit_trail.signal.decision,
            "probability": audit_trail.signal.probability,
            "threshold": audit_trail.signal.threshold,
        }
        base_metadata = dict(metadata or {})
        base_metadata.update(
            {
                "recipients": list(resolved_recipients),
                "cooldown_scope": resolved_cooldown_scope,
            }
        )

        throttle_decision = self.throttle.evaluate(
            channel="sms",
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
                channel="sms",
                status="skipped",
                dedupe_key=resolved_dedupe_key,
                payload=payload,
                metadata=base_metadata,
                error_message=throttle_decision.reason,
                sent_at_utc=resolved_now,
            )
            return SmsDispatchResult(
                status="skipped",
                channel="sms",
                dedupe_key=resolved_dedupe_key,
                notification_id=notification_id,
                reason=throttle_decision.reason,
                transport_response=None,
            )

        status = "sent"
        error_message: str | None = None
        transport_response: str | None = None
        try:
            transport_response = self.transport.send_sms(message)
        except Exception as exc:  # pragma: no cover - transport behavior is caller-specific
            status = "failed"
            error_message = str(exc)

        notification_id = self.audit_repository.record_notification(
            signal_decision_id=signal_decision_id,
            channel="sms",
            status=status,
            dedupe_key=resolved_dedupe_key,
            payload=payload,
            metadata=base_metadata,
            error_message=error_message,
            sent_at_utc=resolved_now,
        )
        if status == "sent":
            self.throttle.remember_sent(
                channel="sms",
                dedupe_key=resolved_dedupe_key,
                cooldown_scope=resolved_cooldown_scope,
                signal_decision_id=signal_decision_id,
                policy=self.policy,
                now=resolved_now,
            )

        return SmsDispatchResult(
            status=status,
            channel="sms",
            dedupe_key=resolved_dedupe_key,
            notification_id=notification_id,
            reason=error_message,
            transport_response=transport_response,
        )


def render_signal_sms_message(
    audit_trail: SignalAuditTrail,
    *,
    recipients: Sequence[str],
) -> SmsAlertMessage:
    signal = audit_trail.signal
    body_text = (
        f"OTE {signal.direction.upper()} {signal.decision.upper()} "
        f"model={signal.model_id} "
        f"p={signal.probability:.3f} thr={_format_optional_float(signal.threshold)} "
        f"regime={signal.regime or 'unknown'} "
        f"id={audit_trail.signal_decision_id}"
    )
    return SmsAlertMessage(
        recipients=tuple(recipients),
        body_text=body_text,
    )


def _format_optional_float(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.3f}"


def is_email_gateway_sms_recipient(value: str) -> bool:
    candidate = str(value or "").strip()
    if "@" not in candidate:
        return False
    local, _, domain = candidate.partition("@")
    return bool(local and "." in domain)
