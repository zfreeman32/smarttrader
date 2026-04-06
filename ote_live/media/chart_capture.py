from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol

from ote_live.dashboard.charts import build_audit_trail_figure
from ote_live.storage import LiveAuditRepository, SignalAuditTrail


@dataclass(frozen=True)
class CapturedChartArtifact:
    artifact_id: int
    signal_decision_id: int
    file_path: Path
    file_hash: str
    metadata: dict[str, Any]


class ChartImageRenderer(Protocol):
    def render_chart(self, audit_trail: SignalAuditTrail) -> bytes:
        """Render a signal chart screenshot and return PNG bytes."""


class ChartImageAnnotator(Protocol):
    def annotate(self, image_bytes: bytes, *, audit_trail: SignalAuditTrail) -> bytes:
        """Add signal metadata to PNG bytes and return the updated bytes."""


class PlotlySignalChartRenderer:
    def __init__(
        self,
        *,
        width: int = 1600,
        height: int = 900,
        scale: int = 1,
    ) -> None:
        self.width = int(width)
        self.height = int(height)
        self.scale = int(scale)

    def render_chart(self, audit_trail: SignalAuditTrail) -> bytes:
        try:
            import plotly.io as pio
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "Signal chart capture requires the 'plotly' package."
            ) from exc

        figure = build_audit_trail_figure(audit_trail)
        try:
            return pio.to_image(
                figure,
                format="png",
                width=self.width,
                height=self.height,
                scale=self.scale,
            )
        except Exception as exc:  # pragma: no cover - environment-specific dependency behavior
            raise RuntimeError(
                "Plotly PNG export failed. Install a supported static image engine such as kaleido."
            ) from exc


class SignalChartCaptureService:
    def __init__(
        self,
        *,
        audit_repository: LiveAuditRepository,
        output_root: str | Path,
        renderer: ChartImageRenderer,
        annotator: ChartImageAnnotator | None = None,
        artifact_type: str = "chart_screenshot",
        lookback_bars: int = 120,
    ) -> None:
        self.audit_repository = audit_repository
        self.output_root = Path(output_root)
        self.renderer = renderer
        self.annotator = annotator
        self.artifact_type = artifact_type
        self.lookback_bars = int(lookback_bars)

    def capture_signal_chart(
        self,
        *,
        signal_decision_id: int,
        audit_trail: SignalAuditTrail | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> CapturedChartArtifact:
        resolved_audit_trail = audit_trail or self.audit_repository.reconstruct_signal(
            signal_decision_id,
            include_bars=True,
            lookback_bars=self.lookback_bars,
        )
        image_bytes = self.renderer.render_chart(resolved_audit_trail)
        if self.annotator is not None:
            image_bytes = self.annotator.annotate(
                image_bytes,
                audit_trail=resolved_audit_trail,
            )

        file_hash = hashlib.sha256(image_bytes).hexdigest()
        file_path = self._artifact_path(resolved_audit_trail)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_bytes(image_bytes)

        artifact_metadata = {
            "model_id": resolved_audit_trail.signal.model_id,
            "direction": resolved_audit_trail.signal.direction,
            "decision": resolved_audit_trail.signal.decision,
            "timestamp_utc": resolved_audit_trail.signal.timestamp.isoformat(),
            "bar_count": len(resolved_audit_trail.bars),
            "renderer": type(self.renderer).__name__,
            "annotated": self.annotator is not None,
        }
        artifact_metadata.update(dict(metadata or {}))
        artifact_id = self.audit_repository.record_media_artifact(
            signal_decision_id=resolved_audit_trail.signal_decision_id,
            artifact_type=self.artifact_type,
            file_path=file_path,
            file_hash=file_hash,
            metadata=artifact_metadata,
        )
        return CapturedChartArtifact(
            artifact_id=artifact_id,
            signal_decision_id=resolved_audit_trail.signal_decision_id,
            file_path=file_path,
            file_hash=file_hash,
            metadata=artifact_metadata,
        )

    def _artifact_path(self, audit_trail: SignalAuditTrail) -> Path:
        signal = audit_trail.signal
        timestamp = signal.timestamp.strftime("%Y%m%dT%H%M%SZ")
        return (
            self.output_root
            / signal.timestamp.strftime("%Y")
            / signal.timestamp.strftime("%m")
            / signal.timestamp.strftime("%d")
            / f"{timestamp}_{_slugify(signal.model_id)}_{signal.direction}_{audit_trail.signal_decision_id}.png"
        )


def _slugify(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value)
