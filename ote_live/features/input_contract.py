"""Qualification metadata, separate from the model's numeric/missing-value API."""
from __future__ import annotations

import re
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from ote_live.features.manifest import LiveRuntimeManifest


INPUT_CONTRACT_VERSION = "es-causal-inputs-v2"
_TRANSFORM = re.compile(r"^(.+)_(?:lag_\d+|roll_(?:mean|std)_\d+|zscore_\d+|pct_rank_\d+|atr_norm|sigma_norm_\d+|winsor_\d+)$")
_EVENT_FEATURE = re.compile(
    r"(?:^dist_to_(?:bull|bear)_(?:fvg|order_block)|^dist_to_equal_(?:high|low)_pool|"
    r"(?:^|_)bars_since_|^ict_nearest_|^ict_latest_|^ict_sweep_(?:level|penetration)|"
    r"^ict_dol_|^ict_dist_to_swing_|^htf_(?:30m|1h)_(?:latest_swing|dist_to_swing))"
)
_SESSION_OPEN = ("ict_rth_open", "ict_rth_gap_open", "ict_open_0830", "ict_midnight_open", "ict_ib_high", "ict_ib_low")


def _event_missing_is_expected(name: str, frame: pd.DataFrame) -> bool:
    # A transform needs its own history; an event-like name alone cannot
    # excuse an uninitialized lag, normalization, or rolling window.
    if not _TRANSFORM.match(name) and _EVENT_FEATURE.search(name):
        return True
    lag = re.match(r"^(.+)_lag_(\d+)$", name)
    if lag and lag[1] in frame.columns:
        offset = int(lag[2])
        if len(frame) <= offset:
            return False
        source = frame.iloc[:-offset] if offset else frame
        return pd.isna(source.iloc[-1][lag[1]]) and _event_missing_is_expected(lag[1], source)
    # A constant event feature has an undefined z-score, including the sparse
    # FRVP S3 channel-slope helper highlighted in the September audit.
    match = re.match(r"^(.+)_zscore_(\d+)$", name)
    if match and match[1] in frame.columns:
        window = int(match[2])
        values = pd.to_numeric(frame[match[1]].tail(window), errors="coerce")
        return len(values) == window and np.isfinite(values).all() and values.nunique() == 1
    return False


def _base_feature(name: str) -> str:
    while match := _TRANSFORM.match(name):
        name = match[1]
    return name


def evaluate_model_input_contract(
    manifest: LiveRuntimeManifest,
    frame: pd.DataFrame,
    *,
    missing_producer_features: Sequence[str] = (),
    fallback_features: Sequence[str] = (),
    producer_contracts: Mapping[str, Any] | None = None,
) -> dict[str, object]:
    """Check every row consumed by the frozen lag/sequence inference window."""
    kwargs = dict(missing_producer_features=missing_producer_features,
                  fallback_features=fallback_features, producer_contracts=producer_contracts)
    result = _evaluate_row_input_contract(manifest, frame, **kwargs)
    offsets = (
        range(max(1, manifest.context_requirements.window_size))
        if manifest.backend in {"tcn", "lstm"}
        else sorted(set([0, *manifest.feature_manifest.lag_steps]))
    )
    history_failures = []
    for offset in offsets:
        if offset == 0:
            continue
        if offset >= len(frame):
            history_failures.append({"row_offset": offset, "reasons": ["missing_history_or_input"]})
            continue
        status = _evaluate_row_input_contract(manifest, frame.iloc[:-offset], **kwargs)
        # Lineage is model-wide and already reported on the latest row.
        reasons = [reason for reason in status["reasons"] if reason != "corrected_feature_lineage_unverified"]
        if reasons:
            history_failures.append({"row_offset": offset, "reasons": reasons,
                                     "feature_states": {key: status["feature_states"][key] for key in reasons}})
    if history_failures:
        result["reasons"].append("historical_required_inputs_failed")
        result["diagnostic_only"] = True
        result["status"] = "diagnostic_only"
    result["history_failures"] = history_failures
    result["evaluated_row_count"] = sum(offset < len(frame) for offset in offsets)
    return result


def _evaluate_row_input_contract(
    manifest: LiveRuntimeManifest,
    frame: pd.DataFrame,
    *,
    missing_producer_features: Sequence[str] = (),
    fallback_features: Sequence[str] = (),
    producer_contracts: Mapping[str, Any] | None = None,
) -> dict[str, object]:
    """Classify current required inputs without treating every NaN as a defect.

    Legacy scores remain reproducible; a passing contract is a separate
    prerequisite for qualification. Producer attestations for external helpers
    require identity/version, verified parity, and a source timestamp equal to
    the evaluated row, so an old helper cannot silently be forward-filled.
    """
    categories: dict[str, list[str]] = {
        name: [] for name in (
            "available", "event_dependent_missing", "not_yet_observed",
            "missing_producer", "unverified_producer", "missing_history_or_input", "stale_input",
        )
    }
    reasons: list[str] = []
    selected = manifest.feature_manifest.selected_feature_names
    missing_producers = set(missing_producer_features)
    fallbacks = set(fallback_features)
    last = frame.iloc[-1] if not frame.empty else pd.Series(dtype=float)
    source_time = last.get("datetime", last.get("timestamp"))
    for name in selected:
        base = _base_feature(name)
        if name not in frame.columns or {name, base}.intersection(missing_producers | fallbacks):
            categories["missing_producer"].append(name)
            continue
        if base.startswith("htf_confluence_"):
            producer = dict((producer_contracts or {}).get(name) or {})
            if not (producer.get("producer_id") and producer.get("version") and producer.get("parity_verified") is True):
                categories["unverified_producer"].append(name)
                continue
            source_timestamps = producer.get("source_timestamps")
            if not isinstance(source_timestamps, (list, tuple)):
                source_timestamps = [producer.get("source_timestamp")]
            observed = pd.to_datetime(source_timestamps, errors="coerce", utc=True)
            current = pd.to_datetime(source_time, errors="coerce", utc=True)
            if pd.isna(current) or current not in observed:
                categories["stale_input"].append(name)
                continue
        htf_prefixes = [prefix for prefix in ("htf_30m", "htf_1h") if base.startswith(prefix + "_")]
        if base == "htf_alignment_score":
            htf_prefixes = ["htf_30m", "htf_1h"]
        if str(manifest.asset).upper() == "ES" and htf_prefixes:
            stale = False
            for prefix in htf_prefixes:
                age = last.get(f"{prefix}_source_age_minutes", np.nan)
                complete = last.get(f"{prefix}_source_complete", 0)
                if pd.isna(complete) or complete != 1 or pd.isna(age):
                    categories["missing_history_or_input"].append(name)
                    stale = True
                    break
                if not np.isfinite(float(age)) or float(age) >= (30 if prefix == "htf_30m" else 60) or float(age) < 0:
                    categories["stale_input"].append(name)
                    stale = True
                    break
            if stale:
                continue
        value = last.get(name)
        if isinstance(value, (int, float, np.number)) and not pd.isna(value) and not np.isfinite(value):
            categories["missing_history_or_input"].append(name)
        elif pd.isna(value):
            if _event_missing_is_expected(name, frame):
                categories["event_dependent_missing"].append(name)
            elif name == base and name.startswith(_SESSION_OPEN) and _session_value_not_yet_observable(name, last):
                categories["not_yet_observed"].append(name)
            else:
                categories["missing_history_or_input"].append(name)
        else:
            categories["available"].append(name)

    for name in ("missing_producer", "unverified_producer", "missing_history_or_input", "stale_input"):
        if categories[name]:
            reasons.append(name)
    corrected_inputs = any(name.startswith(("htf_", "ict_")) for name in selected)
    if str(manifest.asset).upper() == "ES" and corrected_inputs and manifest.feature_manifest.input_contract_version != INPUT_CONTRACT_VERSION:
        reasons.append("corrected_feature_lineage_unverified")
    return {
        "contract_version": INPUT_CONTRACT_VERSION,
        "model_id": manifest.model_id,
        "status": "diagnostic_only" if reasons else "passed",
        "diagnostic_only": bool(reasons),
        "reasons": reasons,
        "feature_states": categories,
        "fallback_features": sorted(set(selected).intersection(fallbacks)),
        "required_feature_count": len(selected),
    }


def _session_value_not_yet_observable(name: str, row: pd.Series) -> bool:
    timestamp = pd.to_datetime(row.get("datetime", row.get("timestamp")), errors="coerce", utc=True)
    if pd.isna(timestamp):
        return False
    local = timestamp.tz_convert("America/New_York")
    minute = local.hour * 60 + local.minute
    if name.startswith(("ict_rth_open", "ict_rth_gap_open")):
        return minute < 570 or minute >= 960
    if name.startswith("ict_open_0830"):
        return minute < 510
    if name.startswith(("ict_ib_high", "ict_ib_low")):
        return not bool(row.get("ict_ib_complete", False))
    # A missing midnight bar after midnight is missing history, not an event.
    return False
