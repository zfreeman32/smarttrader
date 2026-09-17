"""Read-only ES presentation of immutable setup and shadow-decision records."""
from __future__ import annotations

from dataclasses import dataclass
from html import escape
import json

import pandas as pd

from ict.setups.setup_types import ICTSetupType
from ote_live.dashboard.charts import (
    _add_frvp_overlay_traces, _add_ict_overlay_traces, _visible_runtime_state,
    build_price_signal_figure,
)
from ote_live.ingestion.base import timeframe_to_timedelta
from ote_live.policies.shadow import SHADOW_POLICY_CONTRACT, SETUP_MATCH_CONTRACT


def setup_layer_options() -> list[dict]:
    return [
        {"label": "All setup types", "value": "all"},
        *[{"label": f"FRVP S{i}", "value": f"FRVP:{i}"} for i in range(1, 7)],
        *[{"label": f"ICT {item.value.replace('_', ' ')}", "value": f"ICT:{item.value}"}
          for item in ICTSetupType if item != ICTSetupType.NONE],
    ]


@dataclass
class ESChartData:
    collection_version: str | None
    collection_options: list[dict]
    setups: list[dict]
    decisions: list[dict]
    history_available: bool


def fetch_es_chart_data(store, *, bars, asset, timeframe, strategy, model_ids,
                        collection_version="latest", runtime_manifest_hashes=None) -> ESChartData:
    """Load the visible source window without the dashboard's 40/120 event caps.

    Select a single explicit collection. Legacy DBs do not acquire setup tables
    merely by being viewed. Setup records include revisions and rejected events.
    """
    options = [{"label": "Latest collection in chart window", "value": "latest"}]
    available = store.connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='setup_events'"
    ).fetchone() is not None
    if bars.empty:
        return ESChartData(None, options, [], [], available)
    times = pd.to_datetime(bars["timestamp"], utc=True)
    start, end = times.min().isoformat(), times.max().isoformat()
    setups = []
    if available:
        for row in store.connection.execute(
            "SELECT * FROM setup_events WHERE asset=? AND timeframe=? AND strategy=? "
            "AND source_timestamp_utc BETWEEN ? AND ? ORDER BY id",
            (asset, timeframe, strategy, start, end),
        ):
            setups.append({**dict(row), "payload": json.loads(row["payload_json"])})
    decisions = []
    if model_ids:
        placeholders = ','.join('?' for _ in model_ids)
        params = [asset, timeframe, start, end, *model_ids]
        manifest_filter = ""
        if runtime_manifest_hashes is not None:
            hashes = tuple(runtime_manifest_hashes)
            manifest_filter = (" AND rm.manifest_hash IN (" + ','.join('?' for _ in hashes) + ")"
                               if hashes else " AND 0")
            params.extend(hashes)
        rows = store.connection.execute(
            "SELECT sd.id, sd.prediction_id, sd.model_id, sd.collection_version, sd.signal_json, "
            "mp.prediction_json, fs.snapshot_json FROM signal_decisions sd "
            "JOIN model_predictions mp ON mp.id=sd.prediction_id "
            "JOIN feature_snapshots fs ON fs.id=mp.feature_snapshot_id "
            "LEFT JOIN runtime_manifests rm ON rm.id=COALESCE(sd.runtime_manifest_id,mp.runtime_manifest_id,fs.runtime_manifest_id) "
            "WHERE fs.asset=? AND fs.timeframe=? AND sd.timestamp_utc BETWEEN ? AND ? "
            f"AND sd.model_id IN ({placeholders}){manifest_filter} ORDER BY sd.id", params,
        )
        for row in rows:
            decisions.append({**dict(row), "signal": json.loads(row["signal_json"]),
                              "prediction": json.loads(row["prediction_json"]),
                              "snapshot": json.loads(row["snapshot_json"])})
    versions = {}
    for row in setups:
        version = row["collection_version"]
        versions[version] = max(versions.get(version, ""), row["observed_at_utc"])
    for row in decisions:
        version = row["collection_version"]
        prediction = row["prediction"]
        stamp = prediction.get("prediction_recorded_at_utc") or prediction["timestamp"]
        versions[version] = max(versions.get(version, ""), stamp)
    ordered = sorted(versions, key=lambda v: (pd.Timestamp(versions[v]), v), reverse=True)
    options.extend({"label": v, "value": v} for v in ordered)
    version = (ordered[0] if ordered else None) if collection_version in {None, "latest"} else collection_version
    if version is not None and version not in versions:
        options.append({"label": f"{version} (no events in window)", "value": version})
    selected_setups = [s for s in setups if s["collection_version"] == version]
    if selected_setups:
        links = store.connection.execute(
            "SELECT l.setup_event_id, l.prediction_id FROM setup_prediction_links l "
            "JOIN setup_events s ON s.id=l.setup_event_id "
            "WHERE s.asset=? AND s.timeframe=? AND s.strategy=? AND s.collection_version=? "
            "AND s.source_timestamp_utc BETWEEN ? AND ? ORDER BY l.id",
            (asset, timeframe, strategy, version, start, end),
        )
        by_id = {s["id"]: s for s in selected_setups}
        for link in links:
            by_id[link["setup_event_id"]].setdefault("prediction_ids", []).append(link["prediction_id"])
    return ESChartData(version, options, selected_setups,
                       [d for d in decisions if d["collection_version"] == version], available)


def visible_setups(data: ESChartData, *, strategy, setup_types=None, layers=()) -> list[dict]:
    rows = data.setups
    if "revisions" not in layers:
        latest = {row["event_key"]: row for row in sorted(rows, key=lambda r: r["revision"])}
        rows = list(latest.values())
    types = {"all"} if setup_types is None else set(setup_types)
    return [row for row in rows
            if ("all" in types or f"{strategy}:{row['setup_type']}" in types)
            and ("rejected" in layers or (row["selected"] and not row["payload"].get("research")))]


def matching_qualified_decisions(data: ESChartData, setups: list[dict]) -> list[dict]:
    """Use persisted qualification and exact immutable matches, never probability alone."""
    visible_keys = {s["event_key"] for s in setups}
    # A later correction must not erase a decision that was qualified against
    # the original observation. Keep its original event ID and disclose changes.
    by_id = {s["id"]: s for s in data.setups if s["event_key"] in visible_keys}
    latest = {s["event_key"]: s for s in sorted(data.setups, key=lambda s: s["revision"])}
    result = []
    for row in data.decisions:
        evaluation = row["signal"].get("shadow_evaluation") or {}
        match = evaluation.get("setup_match") or {}
        if (evaluation.get("contract_version") != SHADOW_POLICY_CONTRACT
                or match.get("contract_version") != SETUP_MATCH_CONTRACT
                or evaluation.get("qualified_shadow_entry") is not True
                or evaluation.get("policy_candidate_eligible") is not True
                or match.get("matched") is not True):
            continue
        prediction = row["prediction"]
        recorded = _timestamp(prediction.get("prediction_recorded_at_utc"))
        source = row["snapshot"].get("observation_metadata", {}).get("source_bar", {})
        eligibility = row["snapshot"].get("observation_metadata", {}).get("bar_eligibility", {})
        if eligibility.get("eligible") is not True or source.get("is_complete") is not True:
            continue
        matched = []
        for event_id in match.get("matched_event_ids", []):
            event = by_id.get(event_id)
            if (event is None or not event["selected"] or event["payload"].get("research")
                    or event["event_kind"] == "invalidated" or recorded is None):
                continue
            if (event["collection_version"] != row["collection_version"]
                    or event["source_bar_version"] != source.get("bar_version")
                    or event["payload"].get("source_bar", {}).get("instrument_id") != source.get("instrument_id")
                    or _timestamp(event["source_timestamp_utc"]) != _timestamp(prediction["timestamp"])
                    or event["setup_side"] != (1 if prediction["direction"] == "long" else -1)
                    or _timestamp(event["observed_at_utc"]) > recorded):
                continue
            matched.append(event_id)
        if matched:
            details = [{"event_id": event_id, "rule_confidence": by_id[event_id]["rule_confidence"],
                        "latest_event_kind": latest[by_id[event_id]["event_key"]]["event_kind"],
                        "latest_revision": latest[by_id[event_id]["event_key"]]["revision"]}
                       for event_id in matched]
            result.append({**row, "matching_setup_ids": matched, "matching_setup_details": details})
    return result


def build_es_price_figure(bars, *, data, strategy, timeframe, runtime_state=None,
                          mode="setups", setup_types=None, layers=()):
    import plotly.graph_objects as go

    fig = build_price_signal_figure(bars, pd.DataFrame(), title=f"ES {timeframe} {strategy} setups and qualified decisions")
    if "levels" in layers:
        overlay = _add_frvp_overlay_traces if strategy == "FRVP" else _add_ict_overlay_traces
        overlay(fig, bars=bars, runtime_state=_visible_runtime_state(runtime_state, bars))
    setups = visible_setups(data, strategy=strategy, setup_types=setup_types, layers=layers)
    # Keep every type independently toggleable, including rare/research types.
    groups = {}
    for row in setups:
        state = setup_mark_state(row, timeframe)
        key = (row["setup_type"], row["setup_side"], state)
        groups.setdefault(key, []).append(row)
    for (setup_type, side, state), rows in groups.items():
        long = side > 0
        type_label = f"S{setup_type}" if strategy == "FRVP" else setup_type.replace("_", " ")
        symbol = "triangle-up" if long else "triangle-down"
        if "invalidated" in state:
            symbol = "x"
        elif any(s in state for s in ("incomplete", "provisional", "rejected", "research")):
            symbol += "-open"
        elif "revised" in state:
            symbol = "diamond"
        fig.add_trace(go.Scatter(
            x=[r["source_timestamp_utc"] for r in rows],
            y=[r["payload"].get("source_bar", {}).get("low" if long else "high") for r in rows],
            mode="markers", name=f"{strategy} {type_label} {'long' if long else 'short'} · {state}",
            marker=dict(symbol=symbol, size=11, color="#22c55e" if long else "#ef4444"),
            hovertext=[setup_hover(r, timeframe) for r in rows], hoverinfo="text",
        ))
    qualified = matching_qualified_decisions(data, setups)
    for row in qualified:
        evaluation = row["signal"]["shadow_evaluation"]
        # Only an explicitly recorded executable price/time can be drawn as entry.
        # Otherwise show the decision at its actual recording time, on a setup reference.
        entry_at = evaluation.get("executable_entry_at")
        price = evaluation.get("executable_entry_price")
        has_entry = (_timestamp(entry_at) is not None and price is not None
                     and _timestamp(entry_at) >= _timestamp(row["prediction"]["prediction_recorded_at_utc"]))
        source = row["snapshot"].get("observation_metadata", {}).get("source_bar", {})
        fig.add_trace(go.Scatter(
            x=[entry_at if has_entry else row["prediction"]["prediction_recorded_at_utc"]],
            y=[price if has_entry else source.get("close")], mode="markers",
            name="Qualified shadow entry" if has_entry else "Qualified shadow decision (reference price)",
            marker=dict(symbol="star", size=14, color="#38bdf8"),
            hovertext=[decision_hover(row, timeframe)], hoverinfo="text",
        ))
    if mode == "research":
        diagnostic_groups = {}
        for row in data.decisions:
            signal = row["signal"]
            evaluation = signal.get("shadow_evaluation") or {}
            crossed = evaluation.get("threshold_crossed") is True
            if not evaluation:
                probability, threshold = signal.get("probability"), signal.get("threshold")
                crossed = probability is not None and threshold is not None and probability >= threshold
            if not crossed:
                continue
            source = row["snapshot"].get("observation_metadata", {}).get("source_bar", {})
            match = evaluation.get("setup_match", {}).get("matched") is True
            diagnostic_groups.setdefault(match, []).append((row, source))
        for match, points in diagnostic_groups.items():
            fig.add_trace(go.Scatter(
                x=[row["prediction"]["timestamp"] for row, _ in points],
                y=[source.get("close") or _bar_close(bars, row["prediction"]["timestamp"]) for row, source in points],
                mode="markers", name="Setup-matched threshold diagnostic" if match else "Off-setup / unverified threshold diagnostic",
                marker=dict(symbol="circle-open", size=8, color="#a78bfa"),
                hovertext=[decision_hover(row, timeframe) for row, _ in points], hoverinfo="text",
            ))
    if not bars.empty and "is_complete" in bars:
        incomplete = bars.loc[bars["is_complete"].eq(False)]
        if not incomplete.empty:
            fig.add_trace(go.Scatter(x=incomplete["timestamp"], y=incomplete["close"], mode="markers",
                                     name="Incomplete / provisional bar", marker=dict(symbol="square-open", color="#fbbf24", size=10)))
    fig.update_layout(showlegend=True, uirevision=f"es-a8-{strategy}-{data.collection_version}",
                      hovermode="closest", height=660,
                      legend=dict(orientation="h", yanchor="top", y=-0.15))
    if qualified and fig.layout.xaxis.range:
        first, last = fig.layout.xaxis.range
        latest_decision = max(max(_timestamp(r["signal"]["shadow_evaluation"].get("executable_entry_at")
                                            or r["prediction"]["prediction_recorded_at_utc"]),
                                  _timestamp(r["prediction"]["prediction_recorded_at_utc"])) for r in qualified)
        fig.update_xaxes(range=[first, max(_timestamp(last), latest_decision + pd.Timedelta(minutes=1))])
    status = ("No immutable setup history available." if not data.history_available else
              f"{len(setups)} setup observations · {len(qualified)} matching qualified decisions.")
    fig.add_annotation(x=0, y=1.06, xref="paper", yref="paper", showarrow=False, xanchor="left",
                       text=status, font=dict(size=12))
    return fig


def _timestamp(value):
    parsed = pd.to_datetime(value, utc=True, errors="coerce")
    return None if pd.isna(parsed) else parsed


def _delay(observed, source, timeframe):
    observed, source = _timestamp(observed), _timestamp(source)
    if observed is None or source is None:
        return "unknown"
    seconds = (observed - source - timeframe_to_timedelta(timeframe)).total_seconds()
    return f"{seconds:.1f}s after bar close" if seconds >= 0 else f"{-seconds:.1f}s before bar close"


def setup_mark_state(row, timeframe):
    source = row["payload"].get("source_bar", {})
    states = [row["event_kind"]]
    complete = source.get("is_complete")
    if complete is False:
        states.append("incomplete/provisional")
    elif complete is not True:
        states.append("provisional (completion unknown)")
    elif _timestamp(row["observed_at_utc"]) < _timestamp(row["source_timestamp_utc"]) + timeframe_to_timedelta(timeframe):
        states.append("provisional (bar not closed)")
    if row["payload"].get("research"):
        states.append("research")
    if not row["selected"]:
        states.append("rejected")
    return " · ".join(states)


def setup_hover(row, timeframe):
    source = row["payload"].get("source_bar", {})
    lines = [f"{row['strategy']} {row['setup_type']} · {row['setup_family']}",
             f"Rule confidence: {_number(row['rule_confidence'])}",
             f"State: {setup_mark_state(row, timeframe)} · revision {row['revision']}",
             f"Source bar: {row['source_timestamp_utc']}",
             f"Setup observed: {row['observed_at_utc']}",
             f"Setup observation delay: {_delay(row['observed_at_utc'], row['source_timestamp_utc'], timeframe)}",
             f"First bar observation: {source.get('first_observed_at') or 'unknown'}",
             f"First bar observation delay: {_delay(source.get('first_observed_at'), row['source_timestamp_utc'], timeframe)}",
             f"First setup observation: {row['first_observed_at_utc']}",
             f"Feed: {source.get('feed_type') or 'unknown'} · {source.get('observation_kind') or 'unknown'}",
             f"Source version: {row['source_bar_version']}", f"Setup event ID: {row['id']}",
             f"Associated prediction IDs (including diagnostics): {row.get('prediction_ids', [])}",
             f"Collection: {row['collection_version']}"]
    geometry = json.loads(row["geometry_json"])
    for key in ("entry_price", "anchor_level", "stop_reference", "target_reference"):
        if geometry.get(key) is not None:
            lines.append(f"Setup {key.replace('_', ' ')}: {geometry[key]}")
    reasons = row["payload"].get("rejection_reasons") or []
    if reasons:
        lines.append(f"Rejection reasons: {', '.join(reasons)}")
    return "<br>".join(escape(str(line)) for line in lines)


def decision_hover(row, timeframe):
    prediction, signal = row["prediction"], row["signal"]
    evaluation = signal.get("shadow_evaluation") or {}
    metadata = row["snapshot"].get("observation_metadata", {})
    eligibility = metadata.get("bar_eligibility", {})
    lines = [f"Model: {row['model_id']}", f"Model probability: {_number(signal.get('probability'))}",
             f"Raw model score: {_number(prediction.get('raw_score'))}", f"Recorded threshold: {_number(signal.get('threshold'))}",
             f"Source bar: {prediction['timestamp']}",
             f"Prediction recorded: {prediction.get('prediction_recorded_at_utc') or 'unknown'}",
             f"Decision delay: {_delay(prediction.get('prediction_recorded_at_utc'), prediction['timestamp'], timeframe)}",
             f"Observation: {eligibility.get('evaluation_kind', 'unknown')}",
             f"Qualified shadow entry: {evaluation.get('qualified_shadow_entry') is True}",
             f"Matched setup IDs: {evaluation.get('setup_match', {}).get('matched_event_ids', [])}",
             f"Prediction ID: {row['prediction_id']} · Decision ID: {row['id']}",
             f"Collection: {row['collection_version']}",
             f"Rejections: {', '.join(evaluation.get('rejection_reasons', signal.get('reasons', [])))}",
             f"Prerequisites: {', '.join(evaluation.get('prerequisite_reasons', []))}"]
    priority = evaluation.get("research_priority")
    if priority:
        lines.append(f"Recorded research priority: {priority['label']} ({priority['contract_version']})")
        lines.append(priority["note"])
    for setup in row.get("matching_setup_details", []):
        lines.append(f"Matched setup {setup['event_id']} rule confidence: {_number(setup['rule_confidence'])}; "
                     f"latest state: {setup['latest_event_kind']} revision {setup['latest_revision']}")
    return "<br>".join(escape(str(line)) for line in lines)


def _number(value):
    return "unknown" if value is None or pd.isna(value) else f"{float(value):.4f}"


def _bar_close(bars, timestamp):
    matches = bars.loc[pd.to_datetime(bars["timestamp"], utc=True).eq(_timestamp(timestamp))]
    return None if matches.empty else matches.iloc[-1]["close"]
