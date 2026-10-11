"""B2 evidence inventory. Reads saved artifacts; never trains or clears a model.

An observed content hash identifies bytes NOW. It is not a retrospective
attestation that those bytes were consumed by a historical training process.
Path references, current-source dependencies, and numerical witnesses are kept
separate for that reason. No pickle/torch/model deserialization is performed.
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any, Iterable

SCHEMA = "b2-lineage-inventory-v1"
TRANSFORM = re.compile(r"^(.+)_(?:lag_\d+|roll_(?:mean|std)_\d+|zscore_\d+|pct_rank_\d+|atr_norm|sigma_norm_\d+|winsor_\d+)$")
REFERENCE_BASES = {
    "ict_prior_rth_high", "ict_prior_rth_low", "ict_prior_rth_close",
    "ict_prior_week_high", "ict_prior_week_low", "ict_rth_open", "ict_rth_gap_open",
}
SOURCE_FILES = [
    "ict/structure/liquidity.py", "ict/feature_sets/ict_context.py",
    "ict/feature_sets/ict_interactions.py", "ict/detectors/sweeps.py",
    "ict/detectors/premium_discount.py", "ict/detectors/bos_choch.py", "ict/structure/market_structure.py",
    "ict/detectors/displacement.py", "ict/detectors/order_blocks.py",
    "ict/setups/detector.py", "ict/labeling/ict_labeling_engine.py",
    "ict/pipelines/es_primary_phase06.py", "frvp/setups/detector.py",
    "frvp/pipelines/es_primary_phase04.py", "data/labeling/frvp_labeling_engine.py",
    "data/labeling/reversal_labeling_engine.py", "data/labeling/labeling_engine.py",
    "data/labeling/ote_continuation_pullback_labeling_engine.py",
    "data/labeling/ote_breakout_labeling_engine.py", "features/feature_sets/htf_context.py",
    "features/transforms.py", "features/config.py", "features/builder.py", "features/feature_sets/continuation_pullback.py",
    "preprocessing/pipeline.py", "preprocessing/feature_selection.py",
    "preprocessing/feature_importance.py", "model_training/ote_training/ote_xgboost_pipeline.py",
    "model_training/ote_training/feature_ranking.py", "ote_live/features/incremental_engine.py",
    "ote_live/features/input_contract.py", "ote_live/features/manifest.py",
    "ote_live/dashboard/view_registry.py", "ote_live/scripts/run_es_live_collector.py",
    "scripts/materialize_ict_setup_prepared_root.py", "scripts/materialize_frvp_recency_prepared_root.py",
    "model_testing/outcome_atr.py", "configs/outcome_contracts/baseline_scope_v1.json",
    "configs/outcome_contracts/b1_v3.json",
    "model_testing/artifact_lineage_audit.py", "scripts/audit_b2_lineage.py",
]


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def json_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def feature_base(name: str) -> str:
    while match := TRANSFORM.match(name):
        name = match[1]
    return name


def feature_routes(name: str) -> list[str]:
    """Conservative, named dependency routes, not a claim of changed values.

    Do NOT flag every ict_* feature: swing-only ranges, BOS/CHoCH, ordinary order
    blocks, raw displacement and FVG
geometry do not depend on the faulty session reference values merely because
    their producer accepts the same frame. MSS and breaker routes use sweep state;
    actual sensitivity must still be determined by a corrected diff.
"""
    base = feature_base(name)
    result = []
    if base in REFERENCE_BASES or base.removesuffix("_dist_atr") in REFERENCE_BASES:
        result.append("ICT_REFERENCE_DIRECT")
    if (base.startswith(("ict_sweep_", "ict_failed_buy_side_sweep", "ict_failed_sell_side_sweep",
                         "ict_buy_side_sweep", "ict_sell_side_sweep", "ict_dol_"))
            or re.match(r"ict_bars_since_(?:buy_side_sweep|sell_side_sweep|sweep_any)", base)):
        result.append("ICT_REFERENCE_SWEEP_OR_DOL")
    if (re.match(r"ict_(?:mss|bars_since_mss)", base) or "breaker" in base
            or base.startswith("ict_") and any(s in base for s in ("sweep_plus", "after_sweep"))):
        result.append("ICT_REFERENCE_STATEFUL_DEPENDENCY_REVIEW")
    if (base.startswith(("htf_30m_", "htf_1h_", "continuation_")) or base == "htf_alignment_score"
            or base.startswith(("interaction_htf_trend_", "interaction_htf_ltf_"))
            or re.match(r"interaction_(?:london|newyork|overlap)_(?:bull|bear)_setup$", base)):
        result.append("HTF_FEATURE_PUBLICATION")
    if base.startswith(("htf_prev_day_", "htf_prev_week_", "htf_rolling_daily_", "htf_rolling_weekly_",
                        "htf_dist_to_prev_day_", "htf_dist_to_prev_week_", "htf_dist_to_rolling_daily_", "htf_dist_to_rolling_weekly_")):
        result.append("HTF_CALENDAR_REFERENCE_SEMANTICS")
    if base.startswith("htf_confluence_"):
        result.append("LIVE_CONFLUENCE_PRODUCER_MISMATCH")
    return result


def walk_strings(value: Any, pointer: str = "") -> Iterable[tuple[str, str]]:
    if isinstance(value, dict):
        for key, item in value.items():
            yield from walk_strings(item, pointer + "/" + str(key))
    elif isinstance(value, list):
        for i, item in enumerate(value):
            yield from walk_strings(item, pointer + "/" + str(i))
    elif isinstance(value, str):
        yield pointer, value


def looks_like_path(pointer: str, value: str) -> bool:
    if not value or "\n" in value or value.startswith(("http:", "https:")):
        return False
    # Avoid classifying feature names and prose as paths.
    return ("/" in value or "\\" in value) and bool(
        re.search(r"(?:path|file|root|dir|csv|dataset|summary|metadata|manifest|registry|recipe|source|events|labels)", pointer, re.I)
    ) and not any(c in value for c in ("*", "<", ">"))


class Inventory:
    def __init__(self, root: Path):
        self.root = root.resolve()
        self.nodes: dict[str, dict] = {}
        self.documents: dict[str, Any] = {}
        self.references: list[dict] = []
        self.edges: list[dict] = []
        self.errors: list[dict] = []
        self.discovery_exclusions: list[dict] = []

    def canonical(self, value: str | Path) -> str:
        text = str(value).replace("\\", "/")
        p = Path(text)
        if p.is_absolute():
            try:
                return p.resolve().relative_to(self.root).as_posix()
            except ValueError:
                return text
        return p.as_posix().removeprefix("./")

    def path(self, value: str | Path) -> Path:
        p = Path(self.canonical(value))
        return p if p.is_absolute() else self.root / p

    def node(self, value: str | Path, role: str = "referenced") -> str:
        key = self.canonical(value)
        if key not in self.nodes:
            p = self.path(key)
            # Never follow stale foreign-workspace paths outside this checkout.
            local = p.resolve().is_relative_to(self.root)
            exists = local and p.exists()
            stat = p.stat() if exists else None
            self.nodes[key] = dict(path=key, local=local, exists=exists,
                                   kind="directory" if exists and p.is_dir() else "file",
                                   bytes=stat.st_size if exists and p.is_file() else None,
                                   mtime_ns=stat.st_mtime_ns if exists else None,
                                   roles=[], sha256=None, hash_status="pending" if exists and p.is_file() else "not_a_local_file")
        if role not in self.nodes[key]["roles"]:
            self.nodes[key]["roles"].append(role)
        return key

    def edge(self, parent: str | Path, child: str | Path, relationship: str,
             evidence: str | Path, pointer: str = "", confidence: str = "declared_reference") -> None:
        if not parent or not child:
            return
        a, b = self.node(parent), self.node(child)
        if a != b:
            self.edges.append(dict(parent=a, child=b, relationship=relationship,
                                   evidence_file=self.canonical(evidence), evidence_pointer=pointer,
                                   confidence=confidence))

    def document(self, p: Path, role: str = "metadata") -> Any:
        key = self.node(p, role)
        if key in self.documents:
            return self.documents[key]
        if not self.nodes[key]["exists"]:
            return {}
        try:
            data = read_json(p)
        except (ValueError, OSError) as exc:
            self.errors.append(dict(path=key, error=str(exc)))
            return {}
        self.documents[key] = data
        for pointer, value in walk_strings(data):
            if looks_like_path(pointer, value):
                # Content-addressed preservation objects are relative to the
                # containing capture manifest, not the repository root.
                owner_relative = pointer.endswith("/object") and value.replace("\\", "/").startswith("objects/")
                resolved = self.node(p.parent / value if owner_relative else value)
                self.references.append(dict(document=key, pointer=pointer, declared_path=value,
                                            canonical_path=resolved, exists=self.nodes[resolved]["exists"],
                                            resolution="capture_manifest_relative" if owner_relative else "repository_or_absolute_reference"))
        return data

    def discover(self) -> None:
        for base in ("models", "artifacts", "data/prepared", "data/features", "data/labeling/labeled_data", "ote_live/runtime_manifests"):
            for p in sorted((self.root / base).rglob("*.json")):
                if "objects" in p.parts or p.name in {"model.json", "training_history.json", "cv_fold_manifest.json"}:
                    continue
                self.document(p)
        # Research placebos have trained models outside models/. Discover those
        # without treating every evaluation report as an additional trained model.
        for base in ("model_testing/reports", "model_training"):
            for p in sorted((self.root / base).rglob("training_summary.json")):
                self.document(p, "training_summary")
                if p.with_name("model_config.json").exists():
                    self.document(p.with_name("model_config.json"))
        for base in ("tmp", "research"):
            for p in sorted((self.root / base).rglob("training_summary.json")):
                if p.with_name("model.json").is_file() or p.with_name("model.pt").is_file():
                    self.document(p, "temporary_or_research_training_summary")
                    if p.with_name("model_config.json").is_file():
                        self.document(p.with_name("model_config.json"))
                else:
                    self.discovery_exclusions.append(dict(path=self.canonical(p), reason="temporary_metadata_without_saved_model_weights"))
        for base in ("artifacts", "data/features", "data/prepared", "data/labeling/labeled_data"):
            for p in sorted((self.root / base).rglob("*")):
                if p.is_file() and p.suffix in {".csv", ".gz", ".parquet"} and "objects" not in p.parts:
                    self.node(p, "saved_dataset_or_diagnostic_table")
        # Referenced metadata outside these roots (policies, alternate label runs,
        # pipeline summaries) is part of the trace, even if not centrally registered.
        while True:
            pending = [k for k, n in self.nodes.items() if n["exists"] and n["kind"] == "file"
                       and k.endswith(".json") and k not in self.documents
                       and Path(k).name not in {"model.json", "training_history.json", "cv_fold_manifest.json"}]
            if not pending:
                break
            old = len(self.documents)
            for k in pending:
                self.document(self.path(k))
            if len(self.documents) == old:
                break
        for source in SOURCE_FILES:
            self.node(source, "current_source_not_historical_attestation")

    def infer_data_file(self, metadata: str) -> str | None:
        p = self.path(metadata)
        if not p.name.endswith(".metadata.json"):
            return None
        stem = str(p)[:-len(".metadata.json")]
        candidates = [Path(stem), Path(stem + ".gz")] if stem.endswith(".csv") else [Path(stem + ".csv"), Path(stem + ".csv.gz")]
        found = [q for q in candidates if q.is_file()]
        if len(found) == 1:
            return self.node(found[0], "dataset")
        return None  # An ambiguous sibling is never silently chosen.

    def build_data_edges(self) -> None:
        for key, d in list(self.documents.items()):
            if not isinstance(d, dict):
                continue
            p = self.path(key)
            output = self.infer_data_file(key)
            if output:
                self.edge(key, output, "metadata_describes", key, confidence="sidecar_name_and_unique_existing_file")
                # Sources include the nested ICT label source omitted by older B1 inventory.
                for pointer, value in walk_strings(d):
                    field = pointer.rsplit("/", 1)[-1]
                    if field in {"source_path", "source_metadata_file", "upstream_source_path", "upstream_metadata_file", "label_source_path"} and value:
                        self.edge(value, output, "dataset_input", key, pointer)
            if p.name == "summary.json" and (p.parent.name == "prepared" or "data/prepared/" in key):
                prepared_root = p.parent
                self.node(prepared_root, "prepared_root")
                for field in ("input_file", "metadata_file"):
                    if d.get(field):
                        self.edge(d[field], prepared_root, "preparation_input", key, "/" + field)
                self.edge(key, prepared_root, "prepared_metadata", key)
            if p.name == "report.json" and p.with_name("features.json").exists():
                self.edge(p.parent.parent, p.parent, "prepared_target", key)
                for field, value in (d.get("source_lineage") or {}).items():
                    if isinstance(value, str) and looks_like_path(field, value):
                        self.edge(value, p.parent, "target_source_reference", key, "/source_lineage/" + field)
            if p.name == "recency_weighting_summary.json":
                if d.get("base_prepared_root"):
                    self.edge(d["base_prepared_root"], p.parent, "recency_materialization_parent", key, "/base_prepared_root")
                if d.get("source_dataset"):
                    self.edge(d["source_dataset"], p.parent, "recency_timestamp_source", key, "/source_dataset")
                self.edge(key, p.parent, "materialization_metadata", key)
            if p.name == "phase04_gate_report.json":
                a = d.get("artifacts", {})
                merged = a.get("phase02_clean_dataset_csv")
                for field in ("phase02_feature_csv", "phase03_labels_csv"):
                    if merged and a.get(field):
                        self.edge(a[field], merged, "frvp_feature_label_merge", key, "/artifacts/" + field)
                raw = d.get("config", {}).get("input_path")
                for field in ("phase03_labels_csv", "phase03_events_csv"):
                    if raw and a.get(field):
                        self.edge(raw, a[field], "frvp_labeling_raw_input", key, "/config/input_path")
                        self.edge(key, a[field], "pipeline_saved_configuration", key)
                # Events and labels are parallel outputs, not each other's input.
            if p.name in {"phase06_summary.json", "ict_setup_phase06_summary.json"}:
                a = d.get("artifacts", {})
                merged = a.get("cleaned_dataset_csv") or a.get("merged_dataset_csv")
                for field in ("features_csv", "labels_csv", "events_csv"):
                    if merged and a.get(field):
                        self.edge(a[field], merged, "ict_preparation_input", key, "/artifacts/" + field)
                if d.get("base_root") and merged:
                    base = self.path(d["base_root"])
                    self.edge(base / "phase03_labeling/ict_es_events.csv", merged, "ict_specialist_event_parent", key, "/base_root", "saved_base_root_plus_materializer_contract")
                    self.edge(base / "phase04_prepared/ict_es_phase06_merged_dataset.csv", merged, "ict_specialist_feature_parent", key, "/base_root", "saved_base_root_plus_materializer_contract")

    def ancestors(self, child: str) -> list[str]:
        parents: dict[str, set[str]] = defaultdict(set)
        for edge in self.edges:
            parents[edge["child"]].add(edge["parent"])
        seen, queue = set(), deque([child])
        while queue:
            for parent in parents[queue.popleft()]:
                if parent != child and parent not in seen:
                    seen.add(parent)
                    queue.append(parent)
        return sorted(seen)

    def hash_files(self, progress=print) -> None:
        files = [(k, n) for k, n in self.nodes.items() if n["exists"] and n["kind"] == "file"]
        total = sum(n["bytes"] for _, n in files)
        done = 0
        last_report = 0
        progress(f"Hashing {len(files)} referenced files ({total / 1e9:.2f} GB); source files remain read-only.", flush=True)
        for key, node in sorted(files):
            p = self.path(key)
            before = p.stat()
            try:
                node["sha256"] = digest(p)
                after = p.stat()
                stable = (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns)
                same_as_inventory = (before.st_size, before.st_mtime_ns) == (node["bytes"], node["mtime_ns"])
                node["hash_status"] = "observed_stable" if stable and same_as_inventory else "changed_during_inventory"
            except OSError as exc:
                node["hash_status"] = "read_error"
                self.errors.append(dict(path=key, error=str(exc)))
            done += node["bytes"]
            if done - last_report >= 5e9:
                progress(f"Hashed {done / 1e9:.1f}/{total / 1e9:.1f} GB", flush=True)
                last_report = done


def family(target: str) -> str:
    if "_frvp_" in target:
        return "frvp"
    if "_ict_" in target:
        return "ict"
    if target.startswith(("long_", "short_")):
        return "ote"
    return "unclassified"


def build_inventory(root: Path, *, hash_files: bool = True) -> dict:
    inv = Inventory(root)
    inv.discover()
    inv.build_data_edges()
    registrations, runtime, model_paths = [], [], set()
    for key, d in list(inv.documents.items()):
        if not isinstance(d, dict):
            continue
        if key.startswith("models/") and "registry" in Path(key).name:
            for i, row in enumerate(d.get("models", [])):
                if not isinstance(row, dict) or not row.get("artifact_path"):
                    continue
                path = inv.node(row["artifact_path"], "model_bundle")
                model_paths.add(path)
                registrations.append(dict(registry=key, pointer=f"/models/{i}", model_id=row.get("model_id"),
                                          artifact_path=path, status=row.get("status"), backend=row.get("backend"),
                                          registration_hash=json_hash(row), record=row))
        if key.startswith("ote_live/runtime_manifests/"):
            manifests = [d] if "artifact_references" in d else d.get("models", [])
            for i, m in enumerate(manifests):
                if not isinstance(m, dict) or not m.get("artifact_references", {}).get("artifact_dir"):
                    continue
                path = inv.node(m["artifact_references"]["artifact_dir"], "model_bundle")
                model_paths.add(path)
                for field, value in m["artifact_references"].items():
                    if value:
                        inv.node(value, "runtime_artifact_reference")
                runtime.append(dict(manifest=key, pointer="" if manifests == [d] else f"/models/{i}",
                                    model_id=m.get("model_id"), artifact_path=path,
                                    selected_feature_names=m.get("feature_manifest", {}).get("selected_feature_names", []),
                                    input_contract_version=m.get("feature_manifest", {}).get("input_contract_version"),
                                    registry=m.get("registry_path"), live_policy=m.get("live_policy"),
                                    artifact_references=m.get("artifact_references"),
                                    source_lineage=m.get("source_lineage"), record_hash=json_hash(m)))
    # Discover unregistered runs and incomplete/orphan model bundles as well.
    for key in list(inv.documents):
        if Path(key).name == "training_summary.json":
            model_paths.add(inv.node(Path(key).parent, "model_bundle"))
    for pattern in ("model_config.json", "model.pt", "model.json", "scaler.joblib", "best_checkpoint.pt"):
        for p in (root / "models").rglob(pattern):
            model_paths.add(inv.node(p.parent, "model_bundle"))
    auxiliary = []
    for p in sorted((root / "model_training").rglob("*.keras")):
        key = inv.node(p, "auxiliary_saved_model")
        companions = [inv.node(q, "auxiliary_model_metadata_or_scaler") for q in sorted(p.parent.iterdir())
                      if q.is_file() and q.suffix in {".json", ".pkl", ".npy"}]
        auxiliary.append(dict(model_file=key, companions=companions, registered_runtime_dependency_found=False,
                              disposition="unverified_auxiliary_artifact_no_declared_primary_model_edge",
                              baseline_clearance=False,
                              evidence_gap="No training_summary/prepared-parent attestation in primary schema; no inferred tie to identically named htf_confluence helpers."))
    models = []
    for key in sorted(model_paths):
        p = inv.path(key)
        summary = inv.document(p / "training_summary.json", "training_summary")
        config = inv.document(p / "model_config.json", "model_config")
        target = summary.get("target", summary.get("target_name", p.name))
        strategy = family(target)
        selected = summary.get("selected_feature_names") or config.get("selected_feature_names") or []
        files = []
        if inv.nodes[key]["exists"] and p.is_dir():
            for q in sorted(p.iterdir()):
                if q.is_file():
                    files.append(inv.node(q, "model_bundle_file"))
        prepared_dir = summary.get("prepared_dir")
        if not prepared_dir and summary.get("config", {}).get("prepared_root"):
            prepared_dir = str(Path(summary["config"]["prepared_root"]) / target)
        prepared_key = inv.node(prepared_dir, "prepared_target") if prepared_dir else None
        report, features, prepared_files = {}, [], []
        if prepared_key:
            pp = inv.path(prepared_key)
            inv.edge(prepared_key, key, "training_prepared_input", p / "training_summary.json", "/prepared_dir")
            inv.edge(pp.parent, pp, "prepared_target", p / "training_summary.json", "/prepared_dir")
            report = inv.document(pp / "report.json")
            ff = inv.document(pp / "features.json")
            features = ff.get("features", []) if isinstance(ff, dict) else ff
            if inv.nodes[prepared_key]["exists"] and pp.is_dir():
                for q in sorted(pp.iterdir()):
                    if q.is_file():
                        prepared_files.append(inv.node(q, "prepared_target_file"))
                        inv.edge(q, pp, "prepared_split_or_selection_artifact", p / "training_summary.json", "/prepared_dir", "observed_current_directory_member")
            for field, value in (summary.get("source_lineage") or {}).items():
                if isinstance(value, str) and looks_like_path(field, value):
                    inv.edge(value, prepared_key, "training_saved_source_reference", p / "training_summary.json", "/source_lineage/" + field)
        if summary.get("event_window_source"):
            inv.edge(summary["event_window_source"], key, "training_event_window_input", p / "training_summary.json", "/event_window_source")
        exposures = []
        for feature in selected:
            for route in feature_routes(feature):
                exposures.append(dict(feature=feature, base=feature_base(feature), route=route,
                                      evidence="saved_selected_feature_names", impact="dependency_present_value_change_not_yet_measured"))
        # Label exposure is independent of final selected-feature membership.
        label_route = {"frvp": "FRVP_STRUCTURAL_ATR_EVENT_LABEL_PATH", "ict": "ICT_SETUP_LABEL_REFERENCE_AND_HTF_PATH",
                       "ote": "OTE_1H_30M_LABEL_PUBLICATION_PATH"}.get(strategy)
        reg = [r for r in registrations if r["artifact_path"] == key]
        manifests = [r for r in runtime if r["artifact_path"] == key]
        gaps = []
        if not summary:
            gaps.append("missing_training_summary")
        if not prepared_key or not inv.nodes[prepared_key]["exists"]:
            gaps.append("missing_prepared_target")
        if not summary.get("training_label_hash") or not summary.get("training_label_version"):
            gaps.append("historical_training_label_hash_or_version_missing")
        gaps.append("training_time_input_hash_and_producer_attestation_not_established")
        for manifest in manifests:
            if manifest["selected_feature_names"] != selected:
                gaps.append("runtime_selected_features_differ_from_training_summary:" + manifest["manifest"])
            for field in ("model_file", "scaler_file", "calibrator_file"):
                value = manifest["artifact_references"].get(field)
                if value and not inv.path(value).is_file():
                    gaps.append("missing_runtime_" + field + ":" + inv.canonical(value))
        models.append(dict(artifact_path=key, strategy=strategy, target=target,
                           backend=summary.get("backend", config.get("backend")), model_type=config.get("model_type"),
                           purpose="research_placebo" if "placebo_readouts" in key else "temporary_saved_training_run" if key.startswith("tmp/") else "trained_candidate_or_historical_model",
                           model_ids=sorted({r["model_id"] for r in reg + manifests if r.get("model_id")}),
                           registration_count=len(reg), runtime_reference_count=len(manifests),
                           registrations=[{k:r[k] for k in ("registry", "pointer", "model_id", "status")} for r in reg],
                           runtime_manifests=sorted({r["manifest"] for r in manifests}),
                           prepared_dir=prepared_key, source_lineage=summary.get("source_lineage"),
                           prepared_summary=summary.get("prepared_summary"),
                           feature_selection_config={k:summary.get("config", {}).get(k) for k in ("max_loaded_features", "top_feature_min", "top_feature_max", "n_trials", "random_seed")},
                           selected_feature_names=selected, prepared_feature_count=len(features),
                           prepared_feature_routes={route:sorted(f for f in features if route in feature_routes(f))
                                                    for route in sorted({route for f in features for route in feature_routes(f)})},
                           selected_feature_dependencies=exposures, label_dependency=label_route,
                           lag_steps=summary.get("lag_steps"), window_size=summary.get("window_size"),
                           context_rows=summary.get("context_rows"), delta_feature_count=summary.get("delta_feature_count"),
                           split_counts=report.get("split_counts", summary.get("report", {}).get("split_counts")),
                           split_date_ranges=report.get("split_date_ranges", summary.get("report", {}).get("split_date_ranges")),
                           row_identity=summary.get("row_identity"), cv_splitter=summary.get("cv_splitter"),
                           label_horizon_assumptions=summary.get("label_horizon_assumptions"),
                           calibration=summary.get("calibration", config.get("calibration")), threshold=summary.get("threshold"),
                           model_files=files, prepared_files=prepared_files,
                           evidence_gaps=sorted(set(gaps)), disposition="unverified_pending_corrective_B2_review",
                           baseline_clearance=False))
    # Late-discovered sidecars may add graph paths; remove exact duplicate edges.
    inv.build_data_edges()
    inv.edges = [json.loads(s) for s in sorted({json.dumps(e, sort_keys=True) for e in inv.edges})]
    for model in models:
        ancestors = inv.ancestors(model["artifact_path"])
        model["ancestor_paths"] = ancestors
        model["missing_ancestor_paths"] = [a for a in ancestors if not inv.nodes[a]["exists"]]
        model["label_source_paths"] = [a for a in ancestors if re.search(r"(?:labels|events).*\.csv(?:\.gz)?$", a)]
        model["raw_source_paths"] = [a for a in ancestors if a.startswith(("data/futures_data/", "data/currency_data/"))]
        if model["missing_ancestor_paths"]:
            model["evidence_gaps"].append("missing_referenced_ancestors")
    if hash_files:
        inv.hash_files()
    capture_checks = []
    for document, data in inv.documents.items():
        if not document.endswith("baseline_manifest.json") or not isinstance(data, dict):
            continue
        for logical, record in data.get("files", {}).items():
            if isinstance(record, dict) and record.get("source") and record.get("sha256"):
                source = inv.canonical(record["source"])
                observed = inv.nodes.get(source, {}).get("sha256")
                capture_checks.append(dict(capture=document, logical_name=logical, source=source,
                                           captured_sha256=record["sha256"], observed_sha256=observed,
                                           matches_capture=observed == record["sha256"] if observed else None,
                                           meaning="capture equality only; not training-time input proof"))
    for model in models:
        fingerprint_files = [k for k in model["model_files"] if Path(k).name in
                             {"model.json", "model.pt", "scaler.joblib", "calibrator.joblib", "model_config.json"}]
        fingerprint = {k.rsplit("/", 1)[-1]:inv.nodes[k]["sha256"] for k in fingerprint_files}
        model["observed_bundle_sha256"] = json_hash(fingerprint) if fingerprint and all(fingerprint.values()) else None
    routes = Counter(route for m in models for route in {e["route"] for e in m["selected_feature_dependencies"]})
    return dict(schema_version=SCHEMA, scope="all locally discovered saved model bundles, registry entries, runtime manifests and referenced ancestry",
                proof_boundary="Current hashes and declared references do not prove historical consumption or numerical impact. No model is certified by this inventory.",
                summary=dict(model_bundles=len(models), registrations=len(registrations),
                             registered_artifact_paths=len({r["artifact_path"] for r in registrations}),
                             auxiliary_saved_weight_files=len(auxiliary), research_placebo_bundles=sum(m["purpose"] == "research_placebo" for m in models),
                             temporary_saved_training_bundles=sum(m["purpose"] == "temporary_saved_training_run" for m in models),
                             unregistered_model_bundles=sum(not m["registration_count"] for m in models),
                             runtime_records=len(runtime), runtime_artifact_paths=len({r["artifact_path"] for r in runtime}),
                             strategy_counts=dict(Counter(m["strategy"] for m in models)),
                             selected_dependency_model_counts=dict(routes), nodes=len(inv.nodes), edges=len(inv.edges),
                             missing_referenced_files=sum(not n["exists"] for n in inv.nodes.values()),
                             verified_clean_models=0, retrained_models=0, baseline_releases=0),
                models=models, auxiliary_models=auxiliary, registrations=registrations, runtime_records=runtime,
                historical_capture_checks=capture_checks,
                nodes=[inv.nodes[k] for k in sorted(inv.nodes)], edges=inv.edges,
                declared_references=inv.references, discovery_exclusions=inv.discovery_exclusions, parse_errors=inv.errors)


def write_outputs(payload: dict, output: Path) -> None:
    # Refuse replacement of any previous audit. Inputs can never be the destination.
    output.mkdir(parents=True, exist_ok=False)
    (output / "inventory.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    fields = ["artifact_path", "model_ids", "strategy", "target", "backend", "registration_count", "runtime_reference_count",
              "prepared_dir", "label_source_paths", "raw_source_paths", "selected_dependency_routes", "label_dependency",
              "observed_bundle_sha256", "missing_ancestor_paths", "evidence_gaps", "disposition", "baseline_clearance"]
    with (output / "models.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for model in payload["models"]:
            row = {k:model.get(k) for k in fields}
            row["selected_dependency_routes"] = sorted({e["route"] for e in model["selected_feature_dependencies"]})
            writer.writerow({k:json.dumps(v) if isinstance(v, (dict, list)) else v for k, v in row.items()})
    for name, rows, columns in (
        ("edges", payload["edges"], ["parent", "child", "relationship", "evidence_file", "evidence_pointer", "confidence"]),
        ("files", payload["nodes"], ["path", "kind", "exists", "bytes", "sha256", "hash_status", "roles"]),
        ("registrations", payload["registrations"], ["registry", "pointer", "model_id", "artifact_path", "status", "backend", "registration_hash"]),
        ("runtime", payload["runtime_records"], ["manifest", "pointer", "model_id", "artifact_path", "registry", "input_contract_version", "record_hash"]),
    ):
        with (output / (name + ".csv")).open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow({k:json.dumps(v) if isinstance(v, (list, dict)) else v for k,v in row.items()})
    consumers: dict[str, list[str]] = defaultdict(list)
    for model in payload["models"]:
        for ancestor in model["ancestor_paths"]:
            consumers[ancestor].append(model["artifact_path"])
    with (output / "dataset_consumers.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["dataset", "exists", "sha256", "declared_model_descendants", "coverage"])
        writer.writeheader()
        for node in payload["nodes"]:
            if re.search(r"\.(?:csv(?:\.gz)?|parquet)$", node["path"]):
                descendants = sorted(consumers[node["path"]])
                writer.writerow(dict(dataset=node["path"], exists=node["exists"], sha256=node["sha256"],
                                     declared_model_descendants=json.dumps(descendants),
                                     coverage="declared_ancestry" if descendants else "no_declared_descendant_found_not_a_clean_verdict"))
    feature_rows = [dict(artifact_path=m["artifact_path"], **e) for m in payload["models"] for e in m["selected_feature_dependencies"]]
    with (output / "selected_feature_dependencies.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["artifact_path", "feature", "base", "route", "evidence", "impact"])
        writer.writeheader()
        writer.writerows(feature_rows)
    files = sorted(p for p in output.iterdir() if p.is_file())
    (output / "audit_files.sha256.json").write_text(json.dumps({p.name:digest(p) for p in files}, indent=2) + "\n", encoding="utf-8")
