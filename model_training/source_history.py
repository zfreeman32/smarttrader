"""Versioned source-bar windows and frozen missing-value handling.

Label selection chooses endpoints, never the rows inside a window. The saved
scaler carries the same imputation contract used during preparation/training.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

MODE = "consecutive_observed_source_bars_v1"


def file_sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def contract_sha256(contract):
    return hashlib.sha256(json.dumps(contract, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def validate_contract(contract, feature_names):
    if contract.get("mode") != MODE or contract.get("feature_names") != list(feature_names):
        raise ValueError("Source-history mode or feature order mismatch")
    fills = contract.get("fill_values", {})
    if set(fills) != set(feature_names) or not np.isfinite(list(fills.values())).all():
        raise ValueError("Source-history contract requires finite frozen fills for every feature")
    if contract.get("bar_minutes") != 5 or contract.get("missing_value_rule") != "nonfinite_to_frozen_fill":
        raise ValueError("Unsupported source-history clock or imputation rule")


def impute_matrix(matrix, contract):
    values = np.asarray(matrix, dtype=np.float32).copy()
    fills = np.asarray([contract["fill_values"][name] for name in contract["feature_names"]], dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != len(fills):
        raise ValueError("Frozen imputation feature count mismatch")
    bad = ~np.isfinite(values)
    if bad.any():
        values[bad] = np.broadcast_to(fills, values.shape)[bad]
    return np.ascontiguousarray(values)


def selected_contract(contract, feature_names):
    """Freeze exactly the fitted feature order when a trial selects a subset."""
    validate_contract(contract, contract["feature_names"])
    names = list(feature_names)
    if not names or len(set(names)) != len(names) or not set(names) <= set(contract["feature_names"]):
        raise ValueError("Invalid selected history features")
    return {**contract, "feature_names": names,
            "fill_values": {name: contract["fill_values"][name] for name in names}}


class FrozenHistoryScaler:
    """Pickleable runtime adapter: fixed fills, then the fitted training scaler."""

    def __init__(self, scaler, contract):
        validate_contract(contract, contract["feature_names"])
        self.scaler = scaler
        self.history_contract = contract
        self.n_features_in_ = len(contract["feature_names"])

    def transform(self, matrix):
        return self.scaler.transform(impute_matrix(matrix, self.history_contract))


def indexed_lag_view(history, endpoints, selected_features, scaler, lag_steps, delta_feature_count, clip):
    """Gather only causal source rows; scaling is supplied, never fitted here."""
    ends = np.asarray(endpoints, dtype=np.int64)
    lags = list(lag_steps)
    if not lags or lags[0] != 0 or min(lags) < 0 or len(set(lags)) != len(lags):
        raise ValueError("Lag steps must be unique, nonnegative and start at zero")
    if ends.ndim != 1 or (len(ends) and (ends.min() < max(lags) or ends.max() >= len(history))):
        raise ValueError("Missing causal source history for a label endpoint")
    parts = []
    for lag in lags:
        raw = history[np.ix_(ends - lag, selected_features)]
        scaled = np.asarray(scaler.transform(raw), dtype=np.float32)
        if clip > 0:
            np.clip(scaled, -clip, clip, out=scaled)
        parts.append(scaled)
    count = min(delta_feature_count, len(selected_features))
    if count > 0:
        parts.extend(parts[0][:, :count] - part[:, :count] for part in parts[1:].copy())
    return np.ascontiguousarray(np.concatenate(parts, axis=1), dtype=np.float32)
