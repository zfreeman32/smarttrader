"""Explicit, hash-addressed access to pre-B1 evaluation mathematics."""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import json

import pandas as pd

from model_testing.outcome_contracts import content_hash, file_hash


LEGACY_ID = "legacy_label_horizon_or_breakout_event_v0"


def require_legacy_opt_in(enabled: bool) -> None:
    if not enabled:
        raise ValueError(
            "This evaluator uses the pre-B1 label-horizon/breakout-event contract. "
            "For causal outcomes use scripts/run_outcome_evaluation.py with frozen signal evidence. "
            "Historical reproduction requires explicit legacy_outcomes=True / --legacy-outcomes; "
            "legacy results cannot qualify a new B1 baseline."
        )


def legacy_snapshot(config, *, direction: str, training_summary: dict | None = None) -> dict:
    body = dict(contract_id=LEGACY_ID, contract_version=0, outcome_surface="legacy_economic_proxy",
        b1_comparable=False, direction=direction, threshold_config=asdict(config),
        training_summary=training_summary or {},
        training_label_version=(training_summary or {}).get("training_label_version", "UNKNOWN_LEGACY"),
        training_label_hash=(training_summary or {}).get("training_label_hash"),
        training_summary_snapshot_hash=content_hash(training_summary or {}),
        availability="UNVERIFIED_HISTORICAL", entry="signal_close_or_saved_breakout_entry",
        timeout="source_row_plus_label_horizon_or_saved_breakout_exit", overlaps="independent_emitted_events",
        costs="legacy_config_unchanged", gaps="row_lookup_without_complete_path_attestation")
    body["contract_hash"] = content_hash(body)
    return body


def annotate_legacy(frame: pd.DataFrame, snapshot: dict) -> pd.DataFrame:
    result = frame.copy()
    result["evaluation_contract_id"] = snapshot["contract_id"]
    result["evaluation_contract_version"] = 0
    result["evaluation_contract_hash"] = snapshot["contract_hash"]
    result["outcome_surface"] = "legacy_economic_proxy"
    result["b1_comparable"] = False
    if "target" in result:
        result["training_label_outcome"] = result["target"]
    result["training_label_version"] = snapshot["training_label_version"]
    result["training_label_hash"] = snapshot["training_label_hash"]
    return result


def save_legacy_snapshot(output_dir: Path, snapshot: dict, *, artifact_dir: Path) -> None:
    # Metadata is additive; no original trained artifacts or labels are modified.
    payload = dict(snapshot)
    payload["artifact_metadata_hashes"] = {
        name: file_hash(artifact_dir / name)
        for name in ("training_summary.json", "features.json", "metadata.json")
        if (artifact_dir / name).is_file()
    }
    (output_dir / "outcome_contract.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
