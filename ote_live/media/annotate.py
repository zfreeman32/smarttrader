from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO

from ote_live.storage import SignalAuditTrail


@dataclass(frozen=True)
class SignalImageAnnotationOptions:
    banner_height: int = 72
    line_height: int = 18
    padding: int = 12
    background_color: tuple[int, int, int] = (26, 31, 44)
    text_color: tuple[int, int, int] = (255, 255, 255)
    long_accent_color: tuple[int, int, int] = (25, 135, 84)
    short_accent_color: tuple[int, int, int] = (220, 53, 69)


class PillowSignalAnnotator:
    def __init__(self, options: SignalImageAnnotationOptions | None = None) -> None:
        self.options = options or SignalImageAnnotationOptions()
        try:
            from PIL import Image, ImageDraw, ImageFont
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "Signal image annotation requires the 'Pillow' package."
            ) from exc

        self._Image = Image
        self._ImageDraw = ImageDraw
        self._ImageFont = ImageFont

    def annotate(self, image_bytes: bytes, *, audit_trail: SignalAuditTrail) -> bytes:
        image = self._Image.open(BytesIO(image_bytes)).convert("RGBA")
        lines = build_signal_annotation_lines(audit_trail)
        banner_height = max(
            self.options.banner_height,
            self.options.padding * 2 + len(lines) * self.options.line_height,
        )
        canvas = self._Image.new(
            "RGBA",
            (image.width, image.height + banner_height),
            self.options.background_color + (255,),
        )
        canvas.paste(image, (0, banner_height))

        draw = self._ImageDraw.Draw(canvas)
        font = self._ImageFont.load_default()
        accent = (
            self.options.long_accent_color
            if audit_trail.signal.direction == "long"
            else self.options.short_accent_color
        )
        draw.rectangle((0, 0, 8, banner_height), fill=accent + (255,))

        y = self.options.padding
        x = self.options.padding + 10
        for line in lines:
            draw.text((x, y), line, fill=self.options.text_color, font=font)
            y += self.options.line_height

        output = BytesIO()
        canvas.convert("RGB").save(output, format="PNG")
        return output.getvalue()


def build_signal_annotation_lines(audit_trail: SignalAuditTrail) -> tuple[str, ...]:
    signal = audit_trail.signal
    threshold = "n/a" if signal.threshold is None else f"{signal.threshold:.4f}"
    return (
        f"{signal.direction.upper()} {signal.decision.upper()} | prob {signal.probability:.4f} | thr {threshold}",
        f"model {signal.model_id} | regime {signal.regime or 'unknown'} | signal_id {audit_trail.signal_decision_id}",
        f"timestamp {signal.timestamp.isoformat()} | source_row_idx {signal.source_row_idx}",
    )
