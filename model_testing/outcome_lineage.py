"""Read-only family/lineage adapters. Unknown historical label provenance stays unknown."""
from __future__ import annotations

import json
from pathlib import Path

from model_testing.outcome_contracts import CATALOG, ROOT, content_hash, file_hash, load_contracts
from model_testing.outcome_engine import SignalEvidence, TIME_FIELDS, utc
from model_testing.outcome_scope import load_scope


def target_binding(target: str, model_id: str = "") -> tuple[str, str, str]:
    target = target.removeprefix("label_")
    side, _, family = target.partition("_")
    if side not in {"long", "short"}:
        raise ValueError(f"Unregistered target spelling: {target}")
    if family.startswith("frvp_"):
        return "frvp", side, family.removeprefix("frvp_")
    if family.startswith("ict_"):
        return "ict", side, family.removeprefix("ict_")
    if family in {"ote", "ote_meta", "ote_union", "reversal", "continuation", "continuation_pullback", "breakout", "breakout_sustained"}:
        family = {"ote_meta": "meta", "ote_union": "union", "continuation_pullback": "continuation"}.get(family, family)
        if "union" in model_id:
            family = "union"
        elif "meta" in model_id and family == "ote":
            family = "meta"
        return "ote", side, family
    raise ValueError(f"No B1 target binding: {target}")


def signal_from_available_inputs(payload: dict) -> SignalEvidence:
    """Use with saved detector/prediction/policy evidence, NEVER a label row alone.

    All source timestamps must be supplied explicitly (including confirmation
    and HTF completion); this adapter only computes the latest decision time.
    Legacy prediction CSVs without those fields fail validation instead of
    inventing same-bar feature/prediction availability.
    """
    payload = dict(payload)
    required = [key for key in TIME_FIELDS if key not in {"known_at_ts", "event_anchor_ts"}]
    known = max(utc(payload[key]) for key in required)
    if "known_at_ts" in payload:
        known = max(known, utc(payload["known_at_ts"]))
    payload["known_at_ts"] = known.to_pydatetime()
    return SignalEvidence.model_validate(payload)


def inspect_training_lineage(artifact_dir: Path) -> dict:
    summary_path = artifact_dir / "training_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8-sig")) if summary_path.exists() else {}
    label_version = summary.get("training_label_version", "UNKNOWN_LEGACY")
    label_hash = summary.get("training_label_hash")
    return dict(training_label_version=label_version, training_label_hash=label_hash,
                training_label_provenance="verified" if label_hash and label_version != "UNKNOWN_LEGACY" else "unknown_legacy",
                training_summary_snapshot_hash=file_hash(summary_path) if summary_path.exists() else None,
                prepared_root=summary.get("config", {}).get("prepared_root"),
                label_horizon_assumptions=summary.get("label_horizon_assumptions", {}),
                target=summary.get("target_name", summary.get("target", artifact_dir.name)))


def registry_inventory(root: Path = ROOT, *, catalog_path: Path = CATALOG) -> dict:
    """Every root registry, including research/historical roster variants."""
    contracts = load_contracts(catalog_path)
    scope = load_scope()
    rows = []
    for registry_path in sorted((root / "models").glob("*registry*.json")):
        payload = json.loads(registry_path.read_text(encoding="utf-8-sig"))
        for model in payload.get("models", []):
            if not isinstance(model, dict):
                continue
            artifact = Path(model["artifact_path"])
            artifact = artifact if artifact.is_absolute() else root / artifact
            lineage = inspect_training_lineage(artifact)
            target = str(lineage["target"])
            model_id = model["model_id"]
            try:
                strategy, side, family = target_binding(target, model_id)
                matches = sorted(c.contract_id for c in contracts.values()
                                 if c.strategy == strategy and family in c.model_families
                                 and c.side in {"long_or_short", side})
                error = None if matches else "NO_CONTRACT"
            except ValueError as exc:
                strategy, side, family, matches, error = None, None, None, [], str(exc)
            rows.append(dict(registry=str(registry_path.relative_to(root)), registry_hash=file_hash(registry_path),
                model_id=model_id, artifact_path=model["artifact_path"], registry_record_hash=content_hash(model),
                status=model.get("status"), backend=model.get("backend"), strategy=strategy, side=side,
                model_family=family, contract_ids=matches, binding_error=error, **lineage))
            rows[-1]["baseline_contract_status"] = {
                key: scope.reason(strategy=strategy, model_family=family, setup_family=contracts[key].setup_family)
                     or ("unresolved_risk" if contracts[key].risk.rule == "UNRESOLVED" else "included") for key in matches}
    return dict(schema_version="b1-registry-audit-v2", baseline_scope_id=scope.scope_id, baseline_scope_hash=scope.sha256, registrations=len(rows),
                unique_targets=sorted({row["target"] for row in rows}), rows=rows)


def baseline_completion_audit(root: Path = ROOT, *, catalog_path: Path = CATALOG) -> dict:
    """B1 completeness is explicit economics OR an explicit diagnostic exclusion.

    This checks all catalogue bindings, including research families without a
    currently deployed model, plus every registered target. It is not a B2
    lineage clearance, a policy fitness test, or a scored-collection gate.
    """
    contracts, scope = load_contracts(catalog_path), load_scope()
    bindings = []
    for c in contracts.values():
        for family in c.model_families:
            reason = scope.reason(strategy=c.strategy, model_family=family, setup_family=c.setup_family)
            bindings.append(dict(contract_id=c.contract_id, contract_hash=c.sha256, model_family=family,
                                 status=reason or ("unresolved_risk" if c.risk.rule == "UNRESOLVED" else "included")))
    inventory = registry_inventory(root, catalog_path=catalog_path)
    unresolved = [b for b in bindings if b['status'] == 'unresolved_risk']
    failures = [r for r in inventory['rows'] if r['binding_error']]
    return dict(schema_version='b1-completion-audit-v1', catalog=str(catalog_path.relative_to(ROOT))
                if catalog_path.is_relative_to(ROOT) else str(catalog_path),
                catalog_hash=file_hash(catalog_path), catalog_lock_hash=file_hash(catalog_path.with_suffix('.lock.json')),
                baseline_scope_id=scope.scope_id, baseline_scope_hash=scope.sha256,
                contract_count=len(contracts), defined_risk_count=sum(c.risk.rule != 'UNRESOLVED' for c in contracts.values()),
                excluded_risk_placeholder_count=sum(c.risk.rule == 'UNRESOLVED' for c in contracts.values()),
                unresolved_included_bindings=unresolved, registry_binding_failures=failures,
                registrations=inventory['registrations'], unique_targets=inventory['unique_targets'],
                complete=bool(bindings and inventory['rows'] and not unresolved and not failures),
                scored_collection_enabled=False, bindings=bindings)
