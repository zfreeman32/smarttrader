from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import pandas as pd

from frvp.setups.detector import detect_frvp_setups
from frvp.target_lanes import FRVP_SETUP_BROAD_FAMILY_BY_TYPE
from ict.setups.detector import detect_ict_setups
from ict.taxonomy import infer_ict_setup_family, normalize_ict_setup_type


_FRVP_SETUP_MODEL_PATTERN = re.compile(
    r"^frvp_(?P<direction>long|short)_(?P<family>reversal|continuation)_setup"
    r"(?P<setup_id>[1-6])_(?P<backend>xgb|tcn|lstm)_v(?P<version>\d+)\b",
    re.IGNORECASE,
)
_ICT_SETUP_MODEL_PATTERN = re.compile(
    r"^ict_(?P<direction>long|short)_(?P<family>reversal|continuation)_"
    r"(?P<setup_type>.+?)_(?P<backend>xgb|tcn|lstm)_v(?P<version>\d+)\b",
    re.IGNORECASE,
)
_ACRONYM_TOKENS = {
    "fvg": "FVG",
    "htf": "HTF",
    "ib": "IB",
    "ifvg": "IFVG",
    "mss": "MSS",
    "ob": "OB",
    "rth": "RTH",
}


@dataclass(frozen=True)
class SetupModelRoute:
    strategy: str
    direction: str
    family: str
    setup_type: str
    backend: str
    version: int
    setup_id: int | None = None

    @property
    def expected_side(self) -> int:
        return 1 if self.direction == "long" else -1

    @property
    def setup_label(self) -> str:
        if self.strategy == "FRVP" and self.setup_id is not None:
            return f"S{self.setup_id}"
        return _titleize_setup_type(self.setup_type)

    @property
    def display_label(self) -> str:
        return f"{self.strategy} {self.setup_label} {self.family}"


@dataclass(frozen=True)
class CurrentSetupEvent:
    fired: bool
    strategy: str
    setup_type: str | None = None
    setup_id: int | None = None
    setup_family: str | None = None
    setup_side: int | None = None
    confidence: float | None = None


@dataclass(frozen=True)
class SetupFamilyGateResult:
    route: SetupModelRoute | None
    matched: bool
    current_setup: CurrentSetupEvent | None = None
    forced_hold_reasons: tuple[str, ...] = ()
    error_type: str | None = None
    error_message: str | None = None

    @property
    def should_force_hold(self) -> bool:
        return self.route is not None and not self.matched


def infer_setup_model_route(model_id: object) -> SetupModelRoute | None:
    normalized = str(model_id or "").strip().lower()
    if not normalized:
        return None

    match = _FRVP_SETUP_MODEL_PATTERN.match(normalized)
    if match is not None:
        setup_id = int(match.group("setup_id"))
        return SetupModelRoute(
            strategy="FRVP",
            direction=match.group("direction"),
            family=match.group("family"),
            setup_type=str(setup_id),
            setup_id=setup_id,
            backend=match.group("backend"),
            version=int(match.group("version")),
        )

    match = _ICT_SETUP_MODEL_PATTERN.match(normalized)
    if match is None:
        return None
    setup_type = normalize_ict_setup_type(match.group("setup_type"))
    if not setup_type:
        return None
    return SetupModelRoute(
        strategy="ICT",
        direction=match.group("direction"),
        family=match.group("family"),
        setup_type=setup_type,
        setup_id=None,
        backend=match.group("backend"),
        version=int(match.group("version")),
    )


def is_setup_family_model_id(model_id: object) -> bool:
    return infer_setup_model_route(model_id) is not None


def resolve_setup_family_gate(
    model_id: object,
    *,
    policy_frame: pd.DataFrame,
) -> SetupFamilyGateResult:
    route = infer_setup_model_route(model_id)
    if route is None:
        return SetupFamilyGateResult(route=None, matched=True)

    try:
        current_setup = _resolve_current_setup(route.strategy, policy_frame)
    except Exception as exc:
        return SetupFamilyGateResult(
            route=route,
            matched=False,
            current_setup=None,
            forced_hold_reasons=("setup_family_gate_error",),
            error_type=type(exc).__name__,
            error_message=str(exc),
        )

    if current_setup is None or not current_setup.fired:
        return SetupFamilyGateResult(
            route=route,
            matched=False,
            current_setup=current_setup,
            forced_hold_reasons=("setup_family_gate_no_setup_fire",),
        )

    if not _route_matches_current_setup(route, current_setup):
        return SetupFamilyGateResult(
            route=route,
            matched=False,
            current_setup=current_setup,
            forced_hold_reasons=("setup_family_gate_mismatch",),
        )

    return SetupFamilyGateResult(
        route=route,
        matched=True,
        current_setup=current_setup,
    )


def format_setup_model_label(model_id: object) -> str | None:
    route = infer_setup_model_route(model_id)
    if route is None:
        return None
    return route.display_label


def _resolve_current_setup(strategy: str, policy_frame: pd.DataFrame) -> CurrentSetupEvent:
    if policy_frame.empty:
        return CurrentSetupEvent(fired=False, strategy=strategy)
    if strategy == "FRVP":
        return _resolve_current_frvp_setup(policy_frame)
    if strategy == "ICT":
        return _resolve_current_ict_setup(policy_frame)
    return CurrentSetupEvent(fired=False, strategy=strategy)


def _resolve_current_frvp_setup(policy_frame: pd.DataFrame) -> CurrentSetupEvent:
    if {"frvp_setup_type", "frvp_setup_side"}.issubset(policy_frame.columns):
        latest = policy_frame.iloc[-1]
        setup_id = _coerce_int(latest.get("frvp_setup_type"))
        setup_side = _coerce_int(latest.get("frvp_setup_side"))
        confidence = _coerce_float(latest.get("frvp_setup_confidence_rule"))
        fired = bool(setup_id and setup_side)
        family = FRVP_SETUP_BROAD_FAMILY_BY_TYPE.get(setup_id or 0)
        return CurrentSetupEvent(
            fired=fired,
            strategy="FRVP",
            setup_type=str(setup_id) if setup_id else None,
            setup_id=setup_id,
            setup_family=family,
            setup_side=setup_side,
            confidence=confidence,
        )

    detected = detect_frvp_setups(policy_frame)
    latest = detected.iloc[-1]
    setup_id = _coerce_int(latest.get("setup_type"))
    setup_side = _coerce_int(latest.get("setup_side"))
    fired = _coerce_bool(latest.get("fired"))
    return CurrentSetupEvent(
        fired=fired,
        strategy="FRVP",
        setup_type=str(setup_id) if setup_id else None,
        setup_id=setup_id,
        setup_family=FRVP_SETUP_BROAD_FAMILY_BY_TYPE.get(setup_id or 0),
        setup_side=setup_side,
        confidence=_coerce_float(latest.get("confidence")),
    )


def _resolve_current_ict_setup(policy_frame: pd.DataFrame) -> CurrentSetupEvent:
    direct_columns = {"ict_setup_type", "ict_setup_side"}
    if direct_columns.issubset(policy_frame.columns):
        latest = policy_frame.iloc[-1]
        setup_type = normalize_ict_setup_type(latest.get("ict_setup_type"))
        setup_side = _coerce_int(latest.get("ict_setup_side"))
        fired = _coerce_bool(latest.get("ict_setup_fired")) if "ict_setup_fired" in policy_frame.columns else bool(
            setup_type and setup_side
        )
        setup_family = str(latest.get("ict_setup_family") or "").strip().lower()
        return CurrentSetupEvent(
            fired=fired,
            strategy="ICT",
            setup_type=setup_type or None,
            setup_family=setup_family or infer_ict_setup_family(setup_type),
            setup_side=setup_side,
            confidence=_coerce_float(latest.get("ict_setup_confidence")),
        )

    detected = detect_ict_setups(policy_frame)
    latest = detected.iloc[-1]
    setup_type = normalize_ict_setup_type(latest.get("setup_type"))
    return CurrentSetupEvent(
        fired=_coerce_bool(latest.get("fired")),
        strategy="ICT",
        setup_type=setup_type or None,
        setup_family=str(latest.get("setup_family") or "").strip().lower() or infer_ict_setup_family(setup_type),
        setup_side=_coerce_int(latest.get("setup_side")),
        confidence=_coerce_float(latest.get("confidence")),
    )


def _route_matches_current_setup(route: SetupModelRoute, current_setup: CurrentSetupEvent) -> bool:
    if current_setup.setup_side != route.expected_side:
        return False
    if route.strategy != current_setup.strategy:
        return False
    if route.strategy == "FRVP":
        return current_setup.setup_id == route.setup_id
    if route.strategy == "ICT":
        if normalize_ict_setup_type(current_setup.setup_type) != route.setup_type:
            return False
        return (current_setup.setup_family or route.family) == route.family
    return False


def _titleize_setup_type(setup_type: str) -> str:
    words = []
    for token in str(setup_type).strip().lower().split("_"):
        if not token:
            continue
        words.append(_ACRONYM_TOKENS.get(token, token.title()))
    return " ".join(words) if words else "Setup"


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None or pd.isna(value):
        return False
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def _coerce_int(value: Any) -> int | None:
    if value is None or pd.isna(value):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _coerce_float(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


__all__ = [
    "CurrentSetupEvent",
    "SetupFamilyGateResult",
    "SetupModelRoute",
    "format_setup_model_label",
    "infer_setup_model_route",
    "is_setup_family_model_id",
    "resolve_setup_family_gate",
]
