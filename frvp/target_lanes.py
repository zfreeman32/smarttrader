from __future__ import annotations

from typing import Final


FRVP_SETUP_TYPES: Final[tuple[int, ...]] = (1, 2, 3, 4, 5, 6)
FRVP_SETUP_BROAD_FAMILY_BY_TYPE: Final[dict[int, str]] = {
    1: "reversal",
    2: "continuation",
    3: "continuation",
    4: "reversal",
    5: "continuation",
    6: "reversal",
}
FRVP_POOLED_TARGET_FAMILIES: Final[tuple[str, ...]] = (
    "frvp_reversal",
    "frvp_continuation",
)


def pooled_target_family(setup_type: int) -> str:
    """Return the legacy pooled target family for one numbered FRVP setup."""

    normalized = int(setup_type)
    try:
        broad_family = FRVP_SETUP_BROAD_FAMILY_BY_TYPE[normalized]
    except KeyError as exc:
        raise ValueError(f"Unsupported FRVP setup type: {normalized}") from exc
    return f"frvp_{broad_family}"


def setup_target_family(setup_type: int) -> str:
    """Return the additive setup-specific target-family slug."""

    normalized = int(setup_type)
    return f"{pooled_target_family(normalized)}_setup{normalized}"


def target_column(direction: str, target_family: str) -> str:
    normalized_direction = str(direction).strip().lower()
    if normalized_direction not in {"long", "short"}:
        raise ValueError(f"Unsupported FRVP target direction: {direction!r}")
    return f"label_{normalized_direction}_{str(target_family).strip().lower()}"


FRVP_SETUP_TARGET_FAMILIES: Final[tuple[str, ...]] = tuple(
    setup_target_family(setup_type) for setup_type in FRVP_SETUP_TYPES
)
FRVP_POOLED_DIRECT_TARGET_COLUMNS: Final[tuple[str, ...]] = tuple(
    target_column(direction, family)
    for family in FRVP_POOLED_TARGET_FAMILIES
    for direction in ("long", "short")
)
FRVP_SETUP_TARGET_COLUMNS: Final[tuple[str, ...]] = tuple(
    target_column(direction, setup_target_family(setup_type))
    for setup_type in FRVP_SETUP_TYPES
    for direction in ("long", "short")
)
FRVP_META_TARGET_COLUMNS: Final[tuple[str, ...]] = (
    "label_long_frvp_meta",
    "label_short_frvp_meta",
)
FRVP_DIRECT_TARGET_COLUMNS: Final[tuple[str, ...]] = (
    *FRVP_POOLED_DIRECT_TARGET_COLUMNS,
    *FRVP_SETUP_TARGET_COLUMNS,
)
FRVP_TARGET_COLUMNS: Final[tuple[str, ...]] = (
    *FRVP_DIRECT_TARGET_COLUMNS,
    *FRVP_META_TARGET_COLUMNS,
)


__all__ = [
    "FRVP_DIRECT_TARGET_COLUMNS",
    "FRVP_META_TARGET_COLUMNS",
    "FRVP_POOLED_DIRECT_TARGET_COLUMNS",
    "FRVP_POOLED_TARGET_FAMILIES",
    "FRVP_SETUP_BROAD_FAMILY_BY_TYPE",
    "FRVP_SETUP_TARGET_COLUMNS",
    "FRVP_SETUP_TARGET_FAMILIES",
    "FRVP_SETUP_TYPES",
    "FRVP_TARGET_COLUMNS",
    "pooled_target_family",
    "setup_target_family",
    "target_column",
]
