"""Section 3 deterministic smoke replay (R4) for the 16 selected versions.

Roster-driven: B4 v2 freeze, route bindings v2, B3 protocol v3, engine b1-v5.
It verifies routing, entries, exits, costs, ledger accounting and replay hashes.
Profitability is recorded as advisory only and is never interpreted, tuned on or
used as a veto. Nothing here installs, registers, scores live or promotes.

Stages (one job at a time, each safe to re-run):
  setup            choose windows from setup presence and the calendar only
  run --run a|b    per model: verify bytes -> score -> route -> surfaces -> ledgers,
                   then the stream's instrument-capacity ledger
  report           compare run a with run b and write the audits and the summary
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from model_testing.outcome_contracts import content_hash, file_hash, load_contracts
from model_testing.outcome_diagnostics import ModelSetupContext, SetupObservation
from model_testing.outcome_engine import (COST_CASES, ENGINE_VERSION, EvaluationTape, SignalEvidence, TIME_FIELDS,
                                          position_costs)
from model_testing.outcome_routes import load_route_bindings, resolve_model_setup_route
from model_testing.outcome_surfaces import (LEDGER_KINDS, NOT_DEFINED, build_stream_surfaces, guard_report_table,
                                            independent_ledger_id, ledger_summary, persist_stream_surfaces)

RUN_DATE = "20261002"
COLLECTION_ID = f"section3_smoke_replay_{RUN_DATE}"
OUTS = {"ES": ROOT / f"artifacts/s3smoke_{RUN_DATE}_v1", "EURUSD": ROOT / f"artifacts/s3smoke_eurusd_{RUN_DATE}_v1"}
OUT = OUTS["ES"]  # rebound to the requested stream in __main__; streams never share a root
STREAM = "ES"
AUDIT_SUFFIX = {"ES": "", "EURUSD": "_v2"}  # the EURUSD not_run audits of the first pass are kept, not overwritten
UNITS = {"ES": "ticks", "EURUSD": "pips"}
# Pass 2 (user-authorized 2026-10-02): IDs 22 and 26 also score the reversal confirmation bars that their saved
# held-out predictions omit. It has its own root; 21/25/23 model outputs are read from the first EURUSD root.
PASS = 1
FX_FIRST_ROOT = OUTS["EURUSD"]
FX_PASS2_ROOT = ROOT / f"artifacts/s3smoke_eurusd_{RUN_DATE}_v2"
CONFIRMATION_BAR_IDS = (22, 26)
FX_CALENDAR = ROOT / "configs/eurusd_5m_expected_open_calendar_v1.json"
OTE_EVENT_FAMILIES = {"reversal": "reversal", "continuation_pullback": "continuation", "breakout": "breakout"}
PROTOCOL = ROOT / "configs/b3_b4_candidate_review_v3_20261002.json"
FREEZE = ROOT / "configs/b4_broad_full_session_shadow_freeze_v2.json"
BINDINGS = ROOT / "configs/broad_full_session_shadow_route_bindings_v2.json"
ES_CALENDAR_ID = "pmc_5_1_3_cme_globex_equity_plus_carter_v1"
ARTIFACT_FILES = dict(model_config="model_config.json", training_summary="training_summary.json",
                      oof_predictions="oof_predictions.csv", test_predictions="test_predictions.csv",
                      cv_fold_manifest_json="cv_fold_manifest.json", cv_fold_manifest_csv="cv_fold_manifest.csv")
EURUSD_NOT_RUN = ("EURUSD smoke replay is not_run until one versioned EURUSD 5m expected-open calendar is pinned "
                  "(B3 v3 smoke_replay.eurusd_calendar_rule, decision D6)")
TRADE_FIELDS = ("entry_ts", "entry_price", "exit_price", "position_exit_ts", "gross_pnl", "total_cost", "net_pnl")


def dump(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False, default=str) + "\n", encoding="utf-8")


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def roster() -> list[dict]:
    protocol, freeze = load(PROTOCOL), load(FREEZE)
    bindings = {b.queue_id: b for b in load_route_bindings().bindings}
    frozen = {entry["queue_id"]: entry for entry in freeze["frozen_replacements"]}
    order = protocol["section3_roster"]["selection_order"]
    if sorted(order) != sorted(frozen) or sorted(order) != sorted(bindings):
        raise ValueError("Selection order, B4 v2 freeze and route bindings v2 name different models")
    return [dict(queue_id=q, model_id=bindings[q].model_id, stream=bindings[q].instrument, binding=bindings[q],
                 freeze=frozen[q], threshold=float(frozen[q]["threshold"]["value"]),
                 model_dir=ROOT / frozen[q]["artifact_root"],
                 version_hash=frozen[q]["artifacts"]["reviewed_artifact_set_sha256"]) for q in order]


def route_id(model, setup_family):
    return f"{model['queue_id']}:{setup_family}:{model['binding'].setup_bindings[setup_family]}"


def verify_bytes(model) -> dict:
    """Hash every frozen artifact file before anything is loaded from it."""
    artifacts, pins = model["freeze"]["artifacts"], model["freeze"]["source_data_pins"]
    names = dict(ARTIFACT_FILES, model=artifacts["model_file"], scaler=artifacts["scaler_file"],
                 calibrator=artifacts["calibrator_file"])
    files = {}
    for key, expected in artifacts["sha256"].items():
        if expected is None:
            if names[key] is not None and (model["model_dir"] / names[key]).exists():
                raise ValueError(f"{key} is present but not frozen")
            continue
        actual = file_hash(model["model_dir"] / names[key])
        if actual != expected:
            raise ValueError(f"{names[key]} does not match the B4 v2 freeze")
        files[names[key]] = actual
    if file_hash(ROOT / pins["prepared_manifest"]) != pins["prepared_manifest_sha256"]:
        raise ValueError("Prepared manifest does not match the B4 v2 freeze")
    return dict(b4_freeze_files_verified=files, prepared_manifest_sha256=pins["prepared_manifest_sha256"],
                reviewed_artifact_set_sha256=model["version_hash"])


# ---------------------------------------------------------------- ES inputs (setup presence, calendar, bars)

def es_source():
    from scripts.review_b1_candidate_outcomes import source_bars
    return source_bars()[1]


def es_schedule(start, end):
    from scripts.review_b1_candidate_outcomes import build_schedule
    return build_schedule(start, end)


def es_opportunities(model) -> pd.DataFrame:
    """Held-out predictions with the detector's contemporaneous same-side setup. No score, label or outcome."""
    from scripts import run_b2_retraining as old
    binding, side = model["binding"], (1 if model["binding"].side == "long" else -1)
    prepared = load(ROOT / model["freeze"]["source_data_pins"]["prepared_manifest"])
    dataset = ROOT / Path(prepared["source_dataset"].replace("\\", "/"))
    if file_hash(dataset) != prepared["source_dataset_sha256"]:
        raise ValueError("Source dataset does not match the prepared manifest")
    endpoints = pd.read_csv(model["model_dir"] / "test_predictions.csv", usecols=["source_row_idx"]).source_row_idx
    columns = ["source_row_idx", "datetime", "atr_14"] + (
        ["frvp_setup_type", "frvp_setup_side"] if binding.strategy == "frvp" else [])
    rows = pd.read_parquet(dataset, columns=columns).set_index("source_row_idx", drop=False).loc[endpoints]
    rows = rows.reset_index(drop=True)
    rows["datetime"] = pd.to_datetime(rows.datetime, utc=True)
    rows["stop_reference"] = np.nan
    rows["target_reference"] = np.nan
    if binding.strategy == "frvp":
        matched = rows.frvp_setup_side.eq(side) & rows.frvp_setup_type.gt(0)
        rows["setup_family"] = ("setup" + rows.frvp_setup_type.astype("Int64").astype(str)).where(matched)
        detector = "b2_prepared_frvp_detector_columns"
    else:
        old.require_hash(old.EVENTS, old.EVENTS_SHA256)
        events = pd.read_csv(old.EVENTS, usecols=["setup_type", "setup_side", "event_time", "signal_index",
                                                  "stop_reference", "target_reference"])
        events = events.loc[events.setup_side.eq(side)]
        if events.duplicated("signal_index").any():
            raise ValueError("More than one same-side fired ICT event on one bar: the selected route is ambiguous")
        events = events.set_index("signal_index").reindex(rows.source_row_idx)
        if (pd.to_datetime(events.event_time, utc=True).dropna().to_numpy()
                != rows.datetime[events.event_time.notna().to_numpy()].to_numpy()).any():
            raise ValueError("ICT event time is not the decision bar")
        rows["setup_family"] = events.setup_type.to_numpy()
        rows["stop_reference"] = events.stop_reference.to_numpy()
        rows["target_reference"] = events.target_reference.to_numpy()
        detector = "b2_corrected_ict_fired_events"
    declared = rows.setup_family.isin(list(binding.setup_bindings))
    rows["match_state"] = np.where(rows.setup_family.isna(), "no_same_side_setup",
                                   np.where(declared, "selected_same_side_setup", "undeclared_setup_family"))
    rows["detector_version"] = detector
    return rows


def fx_calendar() -> dict:
    lock = load(FX_CALENDAR.with_suffix(".lock.json"))
    if file_hash(FX_CALENDAR) != lock["sha256"]:
        raise ValueError("EURUSD calendar does not match its lock")
    return load(FX_CALENDAR)


def fx_schedule(start, end) -> pd.DataFrame:
    """Expected EURUSD 5m opens from the pinned rule-based calendar. Bars are never consulted."""
    rule = fx_calendar()
    opens = pd.date_range(pd.Timestamp(start).ceil("5min"), pd.Timestamp(end), freq="5min", inclusive="left")
    minutes = opens.hour * 60 + opens.minute
    open_minute = int(rule["weekly_open_utc"]["time"][:2]) * 60
    close_minute = int(rule["weekly_close_utc"]["time"][:2]) * 60
    keep = ~((opens.weekday == 5) | ((opens.weekday == 6) & (minutes < open_minute))
             | ((opens.weekday == 4) & (minutes >= close_minute)))
    for holiday in rule["holiday_closures"]:
        for year in range(opens[0].year - 1, opens[-1].year + 2):
            day = pd.Timestamp(year=year, month=holiday["month"], day=holiday["day"], tz="UTC")
            keep &= ~((opens >= day - pd.Timedelta(hours=2)) & (opens < day + pd.Timedelta(hours=23)))
    opens = opens[keep]
    session = (opens + pd.Timedelta(hours=3)).normalize()
    session = session.where(session.weekday != 5, session - pd.Timedelta(days=1))
    return pd.DataFrame({"open_ts": opens, "session_id": session.strftime("%Y-%m-%d")})


def fx_publication(model) -> dict:
    return load(ROOT / model["freeze"]["source_data_pins"]["prepared_manifest"])["corrected_publication"]


def fx_source(model) -> pd.DataFrame:
    """The pinned B2 EURUSD source tape (bar-open UTC). Its hash comes from the model's prepared manifest."""
    pins = fx_publication(model)
    path = ROOT / Path(pins["label_artifact"].replace("\\", "/"))
    if file_hash(path) != pins["label_artifact_sha256"]:
        raise ValueError("EURUSD source tape does not match the prepared manifest")
    bars = pd.read_csv(path, usecols=["timestamp", "open", "high", "low", "close"])
    bars["open_ts"] = pd.to_datetime(bars.pop("timestamp"), utc=True)
    bars["is_complete"] = True
    return bars


def fx_opportunities(model) -> pd.DataFrame:
    """Held-out predictions with the causal OTE base events confirmed on the decision bar. No score or label."""
    binding, side = model["binding"], model["binding"].side
    pins = fx_publication(model)
    events_path = (ROOT / Path(pins["label_artifact"].replace("\\", "/"))).parent / "events.csv.gz"
    events = pd.read_csv(events_path, usecols=["label_family", "swing_type", "confirm_time", "confirm_index"])
    events["family"] = events.label_family.map(OTE_EVENT_FAMILIES)
    events["side"] = np.where(events.swing_type.eq("low"), "long", "short")
    if events.family.isna().any() or events.duplicated(["confirm_index", "side", "family"]).any():
        raise ValueError("Unknown or duplicated OTE base event")
    prepared = (ROOT / model["freeze"]["source_data_pins"]["prepared_manifest"]).parent / "prepared" / model["freeze"]["target"]
    history = pd.read_parquet(prepared / "source_history.parquet", columns=["source_row_idx", "datetime"])
    closes = pd.to_datetime(history.datetime, utc=True) + pd.Timedelta(minutes=5)
    if (pd.to_datetime(events.confirm_time, utc=True).to_numpy() != closes.iloc[events.confirm_index].to_numpy()).any():
        raise ValueError("OTE event confirmation is not the close of its decision bar")
    grouped = events.sort_values("family").groupby(["confirm_index", "side"]).family.agg("|".join).unstack("side")
    saved_ids = pd.read_csv(model["model_dir"] / "test_predictions.csv", usecols=["source_row_idx"]).source_row_idx
    endpoints = saved_ids
    if PASS == 2 and model["queue_id"] in CONFIRMATION_BAR_IDS:
        # Declared-route confirmation bars strictly inside the saved held-out endpoint range (after the purge).
        own = events.loc[events.side.eq(side) & events.family.isin(list(binding.setup_bindings))
                         & events.confirm_index.between(saved_ids.min(), saved_ids.max())].confirm_index
        endpoints = pd.Series(sorted(set(saved_ids) | set(own)), name="source_row_idx")
    rows = history.set_index("source_row_idx", drop=False).loc[endpoints].reset_index(drop=True)
    rows["saved_prediction"] = rows.source_row_idx.isin(saved_ids).to_numpy()
    rows["datetime"] = pd.to_datetime(rows.datetime, utc=True)
    other = "short" if side == "long" else "long"
    rows["same_side_families"] = grouped[side].reindex(rows.source_row_idx).to_numpy()
    rows["other_side_families"] = grouped[other].reindex(rows.source_row_idx).to_numpy()
    single = rows.same_side_families.notna() & ~rows.same_side_families.fillna("").str.contains("|", regex=False)
    rows["setup_family"] = rows.same_side_families.where(single)
    declared = rows.setup_family.isin(list(binding.setup_bindings))
    rows["match_state"] = np.where(rows.same_side_families.isna(), "no_same_side_setup",
        np.where(~single, "ambiguous_multiple_same_side_setups",
                 np.where(declared, "selected_same_side_setup", "undeclared_setup_family")))
    rows["detector_version"] = "b2_corrected_ote_base_events"
    return rows


def opportunities_of(model) -> pd.DataFrame:
    rows = es_opportunities(model) if model["stream"] == "ES" else fx_opportunities(model)
    if "saved_prediction" not in rows:
        rows["saved_prediction"] = True
    return rows


def es_weeks(bars: pd.DataFrame, start, end, schedule=None) -> pd.DataFrame:
    """One row per trading week of the stream's pinned calendar, with calendar-only attributes."""
    zones = ("America/New_York",) if schedule is None else ("America/New_York", "Europe/London")
    schedule = es_schedule(start, end) if schedule is None else schedule
    session = pd.to_datetime(schedule.session_id)
    schedule["week"] = (session - pd.to_timedelta(session.dt.weekday, unit="D")).dt.date.astype(str)
    schedule["present"] = schedule.open_ts.isin(bars.open_ts)
    schedule["roll"] = False
    if "coordinate_id" in bars:
        observed = bars.set_index("open_ts").coordinate_id
        change = observed.ne(observed.shift()) & observed.shift().notna()
        schedule["roll"] = schedule.open_ts.map(change).fillna(False).astype(bool)
    weeks = schedule.groupby("week").agg(first_open=("open_ts", "min"), last_open=("open_ts", "max"),
        expected=("open_ts", "size"), present=("present", "sum"), es_roll=("roll", "any"),
        sessions=("session_id", lambda s: sorted(set(s)))).reset_index()
    offset = weeks.first_open.map(lambda t: tuple(t.tz_convert(zone).utcoffset() for zone in zones))
    weeks["dst_transition"] = offset.ne(offset.shift()) & offset.shift().notna()
    weeks["planned_closure"] = weeks.expected < weeks.expected.mode().iloc[0]
    weeks["complete"] = weeks.present.eq(weeks.expected)
    return weeks, schedule


def choose_windows(model, opportunities, weeks, schedule, cap) -> dict:
    """Greedy set cover with earliest-week tie-break (B3 v3 window rule). Setup presence and calendar only."""
    week_of = schedule.set_index("open_ts").week
    matched = opportunities.loc[opportunities.match_state.eq("selected_same_side_setup")].copy()
    matched["week"] = matched.datetime.map(week_of)
    new_york = matched.datetime.dt.tz_convert("America/New_York")
    minutes = new_york.dt.hour * 60 + new_york.dt.minute
    matched["segment"] = np.where((minutes >= 570) & (minutes < 960), "rth_segment", "overnight_segment")
    daytime = "rth_segment"
    if model["stream"] == "EURUSD":
        from model_testing.outcome_engine import session_at
        daytime = "london_new_york_segment"
        matched["segment"] = [daytime if session_at(t) in {"london", "overlap", "new_york"} else "overnight_segment"
                              for t in matched.datetime]
    held_out = (opportunities.datetime.min(), opportunities.datetime.max())
    candidates = weeks.loc[weeks.complete & weeks.first_open.ge(held_out[0]) & weeks.last_open.le(held_out[1])]
    covers = {week: set() for week in candidates.week}
    for row in matched.loc[matched.week.isin(covers)].itertuples():
        covers[row.week].update({f"route:{route_id(model, row.setup_family)}", row.segment})
    for row in candidates.itertuples():
        covers[row.week].update(name for name in ("planned_closure", "dst_transition", "es_roll") if getattr(row, name))
    required = [f"route:{route_id(model, family)}" for family in model["binding"].setup_bindings] + [
        "overnight_segment", daytime, "planned_closure", "dst_transition"] + (["es_roll"] if model["stream"] == "ES" else [])
    coverable = set().union(*covers.values()) if covers else set()
    chosen, covered = [], set()
    while len(chosen) < cap and covered != coverable:
        gains = {week: len(elements - covered) for week, elements in covers.items() if week not in chosen}
        best = max(gains.values())
        week = min(week for week, gain in gains.items() if gain == best)
        chosen.append(week)
        covered |= covers[week]
    chosen.sort()
    detail = candidates.set_index("week").loc[chosen]
    element_weeks = {element: [week for week in chosen if element in covers[week]] for element in required}
    unexercised = {element: ("no_eligible_week_in_held_out_period" if element not in coverable
                             else "week_cap_reached") for element in required if not element_weeks[element]}
    return dict(queue_id=model["queue_id"], model_id=model["model_id"], status="selected",
        held_out_period=[t.isoformat() for t in held_out], candidate_weeks=len(candidates),
        weeks=[dict(week=week, first_open=detail.first_open[week].isoformat(),
                    last_open=detail.last_open[week].isoformat(), sessions=detail.sessions[week]) for week in chosen],
        element_weeks={element: found for element, found in element_weeks.items() if found},
        routes_exercised=[e[6:] for e in required if e.startswith("route:") and element_weeks[e]],
        routes_unexercised={e[6:]: reason for e, reason in unexercised.items() if e.startswith("route:")},
        calendar_elements_unexercised={e: reason for e, reason in unexercised.items() if not e.startswith("route:")},
        held_out_predictions=len(opportunities),
        held_out_setup_matches=int(opportunities.match_state.eq("selected_same_side_setup").sum()))


def setup_eurusd() -> None:
    path = OUT / "windows.json"
    if path.exists():
        print(f"windows already chosen: {file_hash(path)}")
        return
    protocol = load(PROTOCOL)
    rule, calendar = protocol["smoke_replay"]["window_rule"], fx_calendar()
    models = [m for m in roster() if m["stream"] == "EURUSD"]
    carried = {}
    if PASS == 2:
        first = load(FX_FIRST_ROOT / "windows.json")["streams"]["EURUSD"]["models"]
        carried = {key: value for key, value in first.items() if int(key) not in CONFIRMATION_BAR_IDS}
        models = [m for m in models if m["queue_id"] in CONFIRMATION_BAR_IDS]
    bars = fx_source(models[0])
    opportunities = {m["queue_id"]: opportunities_of(m) for m in models}
    start = min(frame.datetime.min() for frame in opportunities.values()) - pd.Timedelta(days=7)
    end = bars.open_ts.max() + pd.Timedelta(minutes=5)
    schedule = fx_schedule(start, end)
    in_range = bars.loc[bars.open_ts.ge(start) & bars.open_ts.lt(end)]
    outside = in_range.loc[~in_range.open_ts.isin(schedule.open_ts), "open_ts"]
    weeks, schedule = es_weeks(bars, start, end, schedule)
    chosen = {str(m["queue_id"]): choose_windows(m, opportunities[m["queue_id"]], weeks, schedule,
                                                 rule["cap_weeks_per_model"]) for m in models}
    for window in chosen.values():
        window["prediction_universe"] = ("saved held-out predictions plus the declared-route confirmation bars inside "
                                         "their range" if PASS == 2 else "saved held-out predictions")
    chosen.update(carried)
    for model in models:  # a family the base-event detector never emits cannot be exercised
        window = chosen[str(model["queue_id"])]
        for route in window["routes_unexercised"]:
            if route.split(":")[1] not in OTE_EVENT_FAMILIES.values():
                window["routes_unexercised"][route] = "no_detector_emits_this_setup_family"
    windows = dict(collection_id=COLLECTION_ID, protocol=str(PROTOCOL.relative_to(ROOT)).replace("\\", "/"),
        protocol_sha256=file_hash(PROTOCOL), b4_freeze_sha256=file_hash(FREEZE), route_bindings_sha256=file_hash(BINDINGS),
        window_rule=rule, supersedes="EURUSD not_run entry in " + str((OUTS["ES"] / "windows.json").relative_to(ROOT)).replace("\\", "/"),
        pass_2=(dict(authorized="User, 2026-10-02: for IDs 22 and 26 the production runner may score reversal "
                    "confirmation bars that are absent from the saved held-out predictions.",
                    scope=list(CONFIRMATION_BAR_IDS), carried_unchanged_from=str((FX_FIRST_ROOT / "windows.json").relative_to(ROOT)).replace("\\", "/"),
                    carried_windows_sha256=file_hash(FX_FIRST_ROOT / "windows.json"),
                    caveat="Those scores cannot be compared with saved predictions; every saved bar in the same window is.")
                if PASS == 2 else None),
        d6_resolution="User authorized a provisional rule-based EURUSD calendar on 2026-10-02; pinned and locked before any EURUSD outcome was computed.",
        readings=dict(
            trading_week="sessions of the pinned calendar grouped by the Monday of their session date",
            candidate_week="every expected open present in the source bars; first and last open inside the "
                           "model's first-to-last held-out endpoint",
            setup_match="an OTE base event (reversal, continuation pullback or breakout) of the model's side whose "
                        "confirmation is the close of the decision bar; exactly one same-side family on the bar is "
                        "the detector-selected setup; two or more are ambiguous and stay non-executable markouts",
            route_element="covered only by an unambiguous detector-selected match",
            overnight_and_london_new_york="decision bar opens outside / inside the London, overlap or New York session "
                                          "of the frozen session clock",
            planned_closure="a week with fewer expected opens than the modal week of the calendar",
            dst_transition="first expected open has a different New York or London UTC offset than the previous week's",
            inputs_used="saved held-out endpoint row IDs, OTE base-event family/side/confirmation bar, calendar, bar "
                        "presence; no score, threshold pass, label, quality tier, outcome or P&L"),
        streams=dict(EURUSD=dict(calendar_id=calendar["calendar_id"], calendar_sha256=file_hash(FX_CALENDAR),
            calendar_weeks=len(weeks), complete_weeks=int(weeks.complete.sum()),
            source_bars_outside_calendar=[t.isoformat() for t in outside], models=chosen)))
    windows["windows_hash"] = content_hash(windows)
    dump(path, windows)
    state = dict(windows_sha256=file_hash(path), models={key: "pending" for key in chosen})
    dump(OUT / "state.json", state)
    for key, value in chosen.items():
        print(key, [w["week"] for w in value["weeks"]], "cand", value["candidate_weeks"], "routes", len(value["routes_exercised"]),
              "unexercised", len(value["routes_unexercised"]), list(value["calendar_elements_unexercised"]))
    print("weeks", len(weeks), "complete", int(weeks.complete.sum()), "outside", len(outside), "windows_sha256", state["windows_sha256"])


def setup() -> None:
    if STREAM == "EURUSD":
        return setup_eurusd()
    path = OUT / "windows.json"
    if path.exists():
        print(f"windows already chosen: {file_hash(path)}")
        return
    protocol, models = load(PROTOCOL), roster()
    rule = protocol["smoke_replay"]["window_rule"]
    bars = es_source()
    es_models = [m for m in models if m["stream"] == "ES"]
    opportunities = {m["queue_id"]: es_opportunities(m) for m in es_models}
    start = min(frame.datetime.min() for frame in opportunities.values()) - pd.Timedelta(days=7)
    end = bars.open_ts.max() + pd.Timedelta(minutes=5)
    weeks, schedule = es_weeks(bars, start, end)
    es = {str(m["queue_id"]): choose_windows(m, opportunities[m["queue_id"]], weeks, schedule,
                                             rule["cap_weeks_per_model"]) for m in es_models}
    windows = dict(collection_id=COLLECTION_ID, protocol=str(PROTOCOL.relative_to(ROOT)).replace("\\", "/"),
        protocol_sha256=file_hash(PROTOCOL), b4_freeze_sha256=file_hash(FREEZE), route_bindings_sha256=file_hash(BINDINGS),
        window_rule=rule,
        readings=dict(
            trading_week="sessions of the pinned calendar grouped by the Monday of their session date",
            candidate_week="every expected open present in the source bars; first and last open inside the "
                           "model's first-to-last held-out endpoint",
            overnight_and_rth="a same-side setup match whose decision bar opens outside / inside 09:30-16:00 New York",
            planned_closure="a week with fewer expected opens than the modal week of the calendar",
            dst_transition="first expected open has a different New York UTC offset than the previous week's",
            es_roll="a source bar whose contract coordinate differs from the previous bar's",
            inputs_used="saved held-out endpoint row IDs, detector setup columns / fired ICT events, calendar, "
                        "bar presence, contract coordinates; no score, threshold pass, label, outcome or P&L"),
        streams=dict(
            ES=dict(calendar_id=ES_CALENDAR_ID, calendar_weeks=len(weeks), complete_weeks=int(weeks.complete.sum()),
                    models=es),
            EURUSD=dict(status="not_run", reason=EURUSD_NOT_RUN, calendar_id=None,
                        models={str(m["queue_id"]): dict(queue_id=m["queue_id"], model_id=m["model_id"],
                            status="not_run", weeks=[], routes_exercised=[],
                            routes_unexercised={route_id(m, family): "stream_not_run_no_pinned_calendar"
                                                for family in m["binding"].setup_bindings})
                                for m in models if m["stream"] == "EURUSD"})))
    windows["windows_hash"] = content_hash(windows)
    dump(path, windows)
    state = dict(windows_sha256=file_hash(path),
                 models={str(m["queue_id"]): "pending" for m in models})
    dump(OUT / "state.json", state)
    for key, value in es.items():
        print(key, [w["week"] for w in value["weeks"]], "routes", len(value["routes_exercised"]),
              "unexercised", len(value["routes_unexercised"]), list(value["calendar_elements_unexercised"]))
    print("windows_sha256", state["windows_sha256"])


# ---------------------------------------------------------------- per-model replay

def model_root(queue_id) -> Path:
    carried = PASS == 2 and STREAM == "EURUSD" and queue_id not in CONFIRMATION_BAR_IDS
    return FX_FIRST_ROOT if carried else OUT


def set_status(queue_id, status):
    state = load(OUT / "state.json")
    state["models"][str(queue_id)] = status
    dump(OUT / "state.json", state)


def es_tape(windows):
    bars = es_source()
    chosen = [week for model in windows["streams"]["ES"]["models"].values() for week in model["weeks"]]
    start = min(pd.Timestamp(week["first_open"]) for week in chosen) - pd.Timedelta(days=30)
    end = min(bars.open_ts.max() + pd.Timedelta(minutes=5),
              max(pd.Timestamp(week["last_open"]) for week in chosen) + pd.Timedelta(days=7))
    tape = EvaluationTape(bars.loc[bars.open_ts.ge(start) & bars.open_ts.lt(end)].copy(), es_schedule(start, end),
        timeframe_minutes=5, calendar_id=ES_CALENDAR_ID,
        calendar_source="pandas_market_calendars 5.1.3 CME Globex Equity; B1 review frozen calendar",
        coverage_start=start, coverage_end=end, require_completion=True)
    from model_testing.outcome_atr import completed_structural_atr
    return tape, completed_structural_atr(tape, instrument="ES")


def fx_tape(windows, model):
    from model_testing.outcome_atr import completed_structural_atr
    bars = fx_source(model)
    chosen = [week for entry in windows["streams"]["EURUSD"]["models"].values() for week in entry["weeks"]]
    start = min(pd.Timestamp(week["first_open"]) for week in chosen) - pd.Timedelta(days=30)
    end = min(bars.open_ts.max() + pd.Timedelta(minutes=5),
              max(pd.Timestamp(week["last_open"]) for week in chosen) + pd.Timedelta(days=7))
    schedule = fx_schedule(start, end)
    bars = bars.loc[bars.open_ts.ge(start) & bars.open_ts.lt(end) & bars.open_ts.isin(schedule.open_ts)].copy()
    tape = EvaluationTape(bars, schedule, timeframe_minutes=5, calendar_id=fx_calendar()["calendar_id"],
        calendar_source=f"configs/{FX_CALENDAR.name} sha256 {file_hash(FX_CALENDAR)}",
        coverage_start=start, coverage_end=end, require_completion=True)
    return tape, {minutes: completed_structural_atr(tape, timeframe_minutes=minutes, instrument="EURUSD")
                  for minutes in (60, 30)}


def score(model, window) -> pd.DataFrame:
    """Saved-model production runner on full consecutive source history, for the window's endpoints only."""
    summary = load(model["model_dir"] / "training_summary.json")
    if model["stream"] == "ES":
        from scripts.audit_b2_candidate_history import runtime
        columns = None
    else:
        from scripts.run_b2_ote_candidate_history import runtime_model as runtime
        columns = ["source_row_idx", "datetime", *summary["selected_feature_names"]]
    prepared = (ROOT / model["freeze"]["source_data_pins"]["prepared_manifest"]).parent / "prepared" / model["freeze"]["target"]
    contract = load(prepared / "source_history_contract.json")
    if file_hash(prepared / "source_history.parquet") != contract["source_history_sha256"]:
        raise ValueError("Source history does not match its frozen contract")
    opportunities = opportunities_of(model)
    in_window = pd.Series(False, index=opportunities.index)
    for week in window["weeks"]:
        in_window |= opportunities.datetime.between(pd.Timestamp(week["first_open"]), pd.Timestamp(week["last_open"]))
    rows = opportunities.loc[in_window].reset_index(drop=True)
    runner, history, context = runtime(model["model_dir"], summary), pd.read_parquet(prepared / "source_history.parquet", columns=columns), summary["context_rows"]
    predictions = [runner.predict_latest(history.iloc[end - context:end + 1]) for end in rows.source_row_idx]
    rows["raw_score"] = [float(p.raw_score) for p in predictions]
    rows["calibrated_probability"] = [float(p.calibrated_probability) for p in predictions]
    rows["threshold"] = model["threshold"]
    rows["threshold_passed"] = rows.calibrated_probability >= model["threshold"]
    known = rows.loc[rows.saved_prediction]
    saved = pd.read_csv(model["model_dir"] / "test_predictions.csv").set_index("source_row_idx").loc[known.source_row_idx]
    np.testing.assert_allclose(known.raw_score, saved.raw_probability, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(known.calibrated_probability, saved.calibrated_probability, rtol=1e-5, atol=1e-6)
    np.testing.assert_array_equal(known.threshold_passed, saved.predicted_label.astype(bool))
    return rows


def es_signals(model, scores, tape, hourly, protocol_sha256):
    """One signal per prediction on its detector-selected same-side route. Unmatched rows stay in the census."""
    from model_testing.outcome_atr import structural_atr_at
    binding, signals, census = model["binding"], [], []
    for row in scores.itertuples():
        event_id = f"s3smoke-{model['queue_id']}-{int(row.source_row_idx)}"
        state, coordinate = row.match_state, None
        if state == "selected_same_side_setup":
            if pd.Timestamp(row.datetime) in tape.bars.index:
                coordinate = str(tape.bars.coordinate_id[pd.Timestamp(row.datetime)])
            else:
                state = "decision_bar_missing_from_tape"
        census.append(dict(stream="ES", queue_id=model["queue_id"], model_id=model["model_id"], event_id=event_id,
            source_row_idx=int(row.source_row_idx), decision_bar_open_ts=row.datetime, match_state=state,
            setup_family=row.setup_family, in_surfaces=state == "selected_same_side_setup",
            raw_score=row.raw_score, calibrated_probability=row.calibrated_probability,
            threshold=row.threshold, threshold_passed=row.threshold_passed))
        if state != "selected_same_side_setup":
            continue
        contract = resolve_model_setup_route(model_id=model["model_id"], setup_family=row.setup_family,
                                             instrument="ES", side=binding.side)
        close = pd.Timestamp(row.datetime) + pd.Timedelta(minutes=5)
        known = (close + pd.Timedelta(seconds=1)).to_pydatetime()
        proof, atr, extra = None, float(row.atr_14), {}
        if binding.strategy == "frvp":
            proof = structural_atr_at(hourly, known, coordinate_id=coordinate)
            atr = None if proof is None else proof.value
        else:
            extra = {name: (float(value) if pd.notna(value) else None) for name, value in
                     (("stop_reference", row.stop_reference), ("target_reference", row.target_reference))}
        clocks = {field: known for field in TIME_FIELDS}
        clocks.update(event_anchor_ts=pd.Timestamp(row.datetime).to_pydatetime(), signal_bar_close_ts=close.to_pydatetime(),
                      confirmation_ts=close.to_pydatetime(), feature_max_source_close_ts=close.to_pydatetime(),
                      atr_available_at=known if proof is None else proof.available_at_ts)
        audit = ModelSetupContext(collection_id=COLLECTION_ID, opportunity_id=event_id,
            prediction_id=event_id + "-prediction", available_at_ts=known, coverage="all_model_observations",
            setup_coverage="complete", expected_model_ids=(model["model_id"],),
            model_artifact_hash=model["version_hash"], policy_hash=protocol_sha256, raw_score=row.raw_score,
            calibrated_probability=row.calibrated_probability, threshold=model["threshold"],
            threshold_source="b4_broad_full_session_shadow_freeze_v2", threshold_passed=bool(row.threshold_passed),
            context_source_hash=model["freeze"]["artifacts"]["sha256"]["test_predictions"],
            setups=(SetupObservation(setup_event_id=event_id + "-setup", strategy=binding.strategy,
                setup_family=row.setup_family, side=binding.side, known_at_ts=known, selected=True,
                detector_version=row.detector_version),))
        signals.append(SignalEvidence(event_id=event_id, model_id=model["model_id"],
            model_version=model["model_dir"].parent.parent.name, model_family=binding.model_family,
            strategy=binding.strategy, setup_family=row.setup_family, instrument="ES", side=binding.side,
            training_label_outcome=None, training_label_version="not_used_for_admission",
            training_label_provenance="unknown_legacy", **clocks, availability_status="verified",
            availability_evidence_refs=("historical B1 tape and predeclared bar-close-plus-one-second offline timing; production arrival not certified",),
            atr=atr, atr_definition=contract.risk.atr_definition, atr_provenance=proof, coordinate_id=coordinate,
            policy_eligible=True, policy_rejection_reason="", quantity=1.0, audit_context=audit, **extra))
    return signals, pd.DataFrame(census)


def fx_signals(model, scores, tape, atr, protocol_sha256):
    """One signal per prediction and declared same-side family. Two or more same-side families are ambiguous:
    none is detector-selected, so they are non-executable counterfactual markouts only."""
    from model_testing.outcome_atr import structural_atr_at
    binding, signals, census = model["binding"], [], []
    other_side = "short" if binding.side == "long" else "long"
    families = lambda value: value.split("|") if isinstance(value, str) and value else []
    for row in scores.itertuples():
        base = f"s3smoke-{model['queue_id']}-{int(row.source_row_idx)}"
        same, other, state = families(row.same_side_families), families(row.other_side_families), row.match_state
        declared = [family for family in same if family in binding.setup_bindings]
        if declared and pd.Timestamp(row.datetime) not in tape.bars.index:
            state, declared = "decision_bar_missing_from_tape", []
        census.append(dict(stream="EURUSD", queue_id=model["queue_id"], model_id=model["model_id"], event_id=base,
            source_row_idx=int(row.source_row_idx), decision_bar_open_ts=row.datetime, match_state=state,
            same_side_families="|".join(same), in_surfaces=bool(declared),
            saved_prediction=bool(getattr(row, "saved_prediction", True)), raw_score=row.raw_score,
            calibrated_probability=row.calibrated_probability, threshold=row.threshold,
            threshold_passed=row.threshold_passed))
        if not declared:
            continue
        close = pd.Timestamp(row.datetime) + pd.Timedelta(minutes=5)
        known = (close + pd.Timedelta(seconds=1)).to_pydatetime()
        setups = tuple(SetupObservation(setup_event_id=f"{base}-{side}-{family}-setup", strategy="ote",
                setup_family=family, side=side, known_at_ts=known, selected=len(group) == 1,
                detector_version=row.detector_version)
            for side, group in ((binding.side, same), (other_side, other)) for family in group)
        for family in declared:
            event_id = base if len(same) == 1 else f"{base}-{family}"
            contract = resolve_model_setup_route(model_id=model["model_id"], setup_family=family,
                                                 instrument="EURUSD", side=binding.side)
            minutes = 60 if contract.risk.atr_definition == "completed_1h_structural_atr" else 30
            proof = structural_atr_at(atr[minutes], known, coordinate_id=None)
            clocks = {field: known for field in TIME_FIELDS}
            clocks.update(event_anchor_ts=pd.Timestamp(row.datetime).to_pydatetime(), signal_bar_close_ts=close.to_pydatetime(),
                          confirmation_ts=close.to_pydatetime(), feature_max_source_close_ts=close.to_pydatetime(),
                          atr_available_at=known if proof is None else proof.available_at_ts)
            audit = ModelSetupContext(collection_id=COLLECTION_ID, opportunity_id=base, prediction_id=base + "-prediction",
                available_at_ts=known, coverage="all_model_observations", setup_coverage="complete",
                expected_model_ids=(model["model_id"],), model_artifact_hash=model["version_hash"],
                policy_hash=protocol_sha256, raw_score=row.raw_score, calibrated_probability=row.calibrated_probability,
                threshold=model["threshold"], threshold_source="b4_broad_full_session_shadow_freeze_v2",
                threshold_passed=bool(row.threshold_passed),
                context_source_hash=model["freeze"]["artifacts"]["sha256"]["test_predictions"], setups=setups)
            signals.append(SignalEvidence(event_id=event_id, model_id=model["model_id"],
                model_version=model["model_dir"].parent.parent.name, model_family=binding.model_family, strategy="ote",
                setup_family=family, instrument="EURUSD", side=binding.side, training_label_outcome=None,
                training_label_version="not_used_for_admission", training_label_provenance="unknown_legacy", **clocks,
                availability_status="verified",
                availability_evidence_refs=("pinned B2 EURUSD source tape and predeclared bar-close-plus-one-second offline timing; production arrival not certified",),
                atr=None if proof is None else proof.value, atr_definition=contract.risk.atr_definition, atr_provenance=proof,
                coordinate_id=None, policy_eligible=True, policy_rejection_reason="", quantity=1.0, audit_context=audit))
    return signals, pd.DataFrame(census)


def signals_of(model, scores, tape, atr, protocol_sha256):
    return (es_signals if model["stream"] == "ES" else fx_signals)(model, scores, tape, atr, protocol_sha256)


def build(stream, models, signals, tape, protocol):
    return build_stream_surfaces(stream, signals, tape,
        thresholds={m["model_id"]: m["threshold"] for m in models},
        model_hashes={m["model_id"]: m["version_hash"] for m in models},
        selection_order=protocol["section3_roster"]["selection_order"], protocol_sha256=file_hash(PROTOCOL),
        full_policy={m["model_id"]: None for m in models})


def close(a, b) -> bool:
    return bool(np.isclose(float(a), float(b), rtol=0, atol=1e-9))


def ledger_checks(root: Path, manifest: dict, stream: str, unit: str) -> dict:
    """Accounting checks on the persisted CSVs (after the round trip), one ledger at a time."""
    contracts, problems, unresolved = load_contracts(), [], 0
    for ledger_id, summary in manifest["ledger_summaries"].items():
        table = pd.read_csv(root / "ledgers" / f"{ledger_id}.csv")
        if not len(table):
            continue
        guard_report_table(table)
        if set(table.stream) != {stream} or set(table.unit) != {unit}:
            problems.append(f"{ledger_id}: foreign stream or unit")
        taken = table.loc[table.position_taken.astype(bool)]
        if not set(taken.position_status) <= {"CLOSED", "UNRESOLVED"}:
            problems.append(f"{ledger_id}: admitted row with no position status")
        closed, open_rows = table.loc[table.position_status.eq("CLOSED")], table.loc[table.position_status.eq("UNRESOLVED")]
        unresolved += len(open_rows)
        if open_rows[["gross_pnl", "net_pnl", "total_cost"]].notna().any().any():
            problems.append(f"{ledger_id}: an open position carries realized P&L or cost")
        for column in ("gross_pnl", "total_cost", "net_pnl"):
            if not close(closed[column].sum() if len(closed) else 0.0, summary[column]):
                problems.append(f"{ledger_id}: per-trade {column} does not sum to the ledger total")
        if summary["closed"] != len(closed) or summary["unresolved_open_exposure"] != len(open_rows):
            problems.append(f"{ledger_id}: closed/open counts differ from the ledger summary")
        for row in closed.itertuples():
            cost = contracts[row.outcome_contract_id].costs
            expected = position_costs(entry_ts=row.entry_ts, exit_ts=row.position_exit_ts, cost=cost,
                                      quantity=row.quantity, accounting_unit=unit)
            factor = COST_CASES[row.cost_case]
            sign = 1 if row.side == "long" else -1
            multiplier = row.quantity * (cost.cash_per_unit if unit == "cash" else 1)
            gross = sign * (row.exit_price - row.entry_price) / cost.price_increment * multiplier
            if not (close(row.total_cost, expected["total_cost"] * factor)
                    and close(row.total_cost, row.spread_cost + row.slippage_cost + row.commission_cost)
                    and close(row.net_pnl, row.gross_pnl - row.total_cost) and close(row.gross_pnl, gross)):
                problems.append(f"{ledger_id}: {row.event_id} cost or P&L differs from the contract rates")
    for path in sorted((root / "markouts").rglob("*.csv")):
        table = pd.read_csv(path)
        if len(table) and (set(table.stream) != {stream} or set(table.unit) != {unit}
                           or {"gross_pnl", "net_pnl", "total_cost"} & set(table.columns)):
            problems.append(f"{path.parent.name}: foreign stream/unit or a P&L column in a markout store")
    return dict(problems=problems, unresolved_rows=unresolved)


def route_coverage(root: Path, model, window) -> list[dict]:
    queue_id = model["queue_id"]
    stream = model["stream"]
    store = root / "markouts" / f"mk.{stream}.{queue_id}.raw_score_gross_forward_response"
    forward, paper = pd.read_csv(store / "forward_response.csv"), pd.read_csv(store / "paper_contract_outcome.csv")
    ledgers = {variant: pd.read_csv(root / "ledgers" / f"{independent_ledger_id(stream, queue_id, variant, 'base')}.csv")
               for variant in ("setup_only", "setup_threshold")}
    rows = []
    for family in model["binding"].setup_bindings:
        rid = route_id(model, family)
        f, p = forward.loc[forward.route_id.eq(rid)], paper.loc[paper.route_id.eq(rid)]
        unresolved = paper.loc[paper.route_id.eq(rid) & ~paper.paper_resolved.astype(bool)]
        admitted = {variant: int(t.loc[t.route_id.eq(rid), "position_taken"].astype(bool).sum()) if len(t) else 0
                    for variant, t in ledgers.items()}
        admitted["full_policy"] = NOT_DEFINED
        rows.append(dict(route_id=rid, exercised=rid in window["routes_exercised"], setup_matches_in_window=len(f),
            executable_routes=int(f.executable_route.astype(bool).sum()) if len(f) else 0,
            counterfactual_markouts=int(f.counterfactual_markout_only.astype(bool).sum()) if len(f) else 0,
            admitted_per_variant=admitted,
            paper_contract_outcomes={str(k): int(v) for k, v in p.paper_contract_outcome.value_counts().sort_index().items()},
            data_gap_and_censor_reasons={str(k): int(v) for k, v in unresolved.paper_outcome_reason.fillna("").value_counts().sort_index().items()},
            unresolved=len(unresolved)))
    return rows


def run_model(model, window, tape, hourly, protocol, out: Path):
    report_path = out / "report.json"
    if report_path.exists():
        scores = pd.read_csv(out / "scores.csv", parse_dates=["datetime"], float_precision="round_trip")  # exact floats
        for column in ("setup_family", "same_side_families", "other_side_families"):
            if column in scores:
                scores[column] = scores[column].astype(object).where(scores[column].notna(), None)
        return signals_of(model, scores, tape, hourly, file_hash(PROTOCOL))[0], load(report_path)
    if out.exists():
        shutil.rmtree(out)  # an interrupted attempt of this harness; nothing else writes here
    out.mkdir(parents=True)
    stream = model["stream"]
    verified = verify_bytes(model)
    if not window["weeks"]:
        raise ValueError("no eligible window: no recorded held-out prediction has a detector-selected same-side "
                         "setup match on a declared route, so no route can be exercised")
    scores = score(model, window)
    scores.to_csv(out / "scores.csv", index=False)
    signals, census = signals_of(model, scores, tape, hourly, file_hash(PROTOCOL))
    census.to_csv(out / "census.csv", index=False)
    if not signals:
        raise ValueError("no setup match inside the chosen window")
    result = build(stream, [model], signals, tape, protocol)
    # The instrument-capacity ledger belongs to the stream step, never to one model.
    result["ledgers"] = {key: value for key, value in result["ledgers"].items()
                         if value.empty or value.ledger_kind.iloc[0] == LEDGER_KINDS[0]}
    if any(key.startswith("cap.") for key in result["ledgers"]):
        result["ledgers"] = {key: value for key, value in result["ledgers"].items() if not key.startswith("cap.")}
    manifest = persist_stream_surfaces(result, out)
    checks = ledger_checks(out / stream, manifest, stream, manifest["unit"])
    if "saved_prediction" not in scores:
        scores["saved_prediction"] = True
    known = scores.loc[scores.saved_prediction.astype(bool)]
    saved = pd.read_csv(model["model_dir"] / "test_predictions.csv").set_index("source_row_idx").loc[known.source_row_idx]
    report = dict(queue_id=model["queue_id"], model_id=model["model_id"], stream=stream, unit=manifest["unit"],
        engine_version=ENGINE_VERSION, **verified,
        scoring=dict(runner="ote_live RuntimeModelRunner via the B2 candidate-history loader",
            history="full consecutive source history (context rows before each endpoint)",
            endpoints_scored=len(scores), endpoints_verified_against_saved_predictions=len(known),
            endpoints_without_a_saved_prediction=int(len(scores) - len(known)),
            unverified_note=("scores on bars without a saved prediction cannot be compared with saved output; the same "
                             "loaded model is verified on every saved bar of the same window") if len(known) < len(scores) else None,
            raw_score_max_abs_error=float(np.max(np.abs(known.raw_score.to_numpy() - saved.raw_probability.to_numpy()))) if len(known) else None,
            threshold=model["threshold"], threshold_decisions_equal_saved=True),
        census={str(k): int(v) for k, v in census.match_state.value_counts().sort_index().items()} if len(census) else {},
        signals=len(signals), full_policy=manifest["full_policy"], surface_hash=manifest["surface_hash"],
        files=manifest["files"], scores_sha256=file_hash(out / "scores.csv"), census_sha256=file_hash(out / "census.csv"),
        calendar_id=manifest["calendar_id"], calendar_hash=manifest["calendar_hash"],
        accounting=checks, route_coverage=route_coverage(out / stream, model, window),
        advisory_ledger_summaries=manifest["ledger_summaries"])
    report["canonical_hash"] = content_hash({key: report[key] for key in (
        "surface_hash", "files", "scores_sha256", "census_sha256", "calendar_hash", "full_policy")})
    dump(report_path, report)
    return signals, report


def run_stream_capacity(stream, models, signals, tape, protocol, out: Path, model_reports) -> dict:
    report_path = out / "report.json"
    if report_path.exists():
        return load(report_path)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    manifest = persist_stream_surfaces(build(stream, models, signals, tape, protocol), out)
    root = out / stream
    checks = ledger_checks(root, manifest, stream, manifest["unit"])
    capacity = pd.read_csv(root / "ledgers" / f"cap.{stream}.instrument_policy.base.csv")
    reconcile, problems = {}, list(checks["problems"])
    for model in models:
        queue_id = model["queue_id"]
        # Independence: a model's own ledgers are byte-identical whether built alone or beside the others.
        for name, digest in model_reports[queue_id]["files"].items():
            if manifest["files"].get(name) != digest:
                problems.append(f"{queue_id}: {name} differs between the model build and the stream build")
        own = pd.read_csv(root / "ledgers" / f"{independent_ledger_id(stream, queue_id, 'setup_threshold', 'base')}.csv")
        cap = capacity.loc[capacity.queue_id.eq(queue_id)] if len(capacity) else capacity
        if len(cap) and set(cap.surface) != {"setup_threshold"}:
            problems.append(f"{queue_id}: capacity feed is not labeled setup_threshold")
        both = cap.loc[cap.position_taken.astype(bool)].merge(
            own.loc[own.position_taken.astype(bool)], on="event_id", suffixes=("_cap", "_ind")) if len(cap) and len(own) else pd.DataFrame()
        mismatched = 0
        for row in both.itertuples():
            for field in TRADE_FIELDS:
                a, b = getattr(row, field + "_cap"), getattr(row, field + "_ind")
                if not ((pd.isna(a) and pd.isna(b)) or a == b or (not isinstance(a, str) and close(a, b))):
                    mismatched += 1
                    break
        if mismatched:
            problems.append(f"{queue_id}: {mismatched} trades differ between the capacity and independent ledgers")
        reconcile[str(queue_id)] = dict(capacity_rows=len(cap),
            capacity_admitted=int(cap.position_taken.astype(bool).sum()) if len(cap) else 0,
            independent_admitted=int(own.position_taken.astype(bool).sum()) if len(own) else 0,
            trades_in_both=len(both), trades_mismatched=mismatched)
    capacity_files = {name: digest for name, digest in manifest["files"].items() if "/cap." in name}
    report = dict(stream=stream, unit=manifest["unit"], engine_version=ENGINE_VERSION,
        models=[m["queue_id"] for m in models], calendar_id=manifest["calendar_id"], calendar_hash=manifest["calendar_hash"],
        surface_hash=manifest["surface_hash"], capacity_files=capacity_files, problems=problems, reconcile=reconcile,
        unresolved_rows=checks["unresolved_rows"],
        advisory_ledger_summaries={key: value for key, value in manifest["ledger_summaries"].items() if key.startswith("cap.")})
    report["canonical_hash"] = content_hash({"capacity_files": capacity_files, "calendar_hash": report["calendar_hash"]})
    dump(report_path, report)
    return report


def run(run_id: str) -> None:
    windows, protocol, state = load(OUT / "windows.json"), load(PROTOCOL), load(OUT / "state.json")
    if file_hash(OUT / "windows.json") != state["windows_sha256"]:
        raise ValueError("Windows changed after they were chosen")
    models = roster()
    es_models = [m for m in models if m["stream"] == STREAM]
    for model in models:
        if STREAM == "ES" and model["stream"] == "EURUSD":  # first pass only: bytes verified, nothing scored
            try:
                dump(OUT / run_id / f"m{model['queue_id']}" / "report.json", dict(queue_id=model["queue_id"],
                    model_id=model["model_id"], stream="EURUSD", status="not_run", reason=EURUSD_NOT_RUN,
                    **verify_bytes(model)))
            except Exception as error:
                dump(OUT / run_id / f"m{model['queue_id']}" / "error.json", dict(error=repr(error)))
    tape, hourly = es_tape(windows) if STREAM == "ES" else fx_tape(windows, es_models[0])
    print(STREAM, "tape", tape.coverage_start, tape.coverage_end, len(tape.bars), flush=True)
    passed, signals, reports = [], [], {}
    for model in es_models:
        queue_id = model["queue_id"]
        out = model_root(queue_id) / run_id / f"m{queue_id}"
        if run_id == "a":
            set_status(queue_id, "running")
        try:
            model_signals, report = run_model(model, windows["streams"][STREAM]["models"][str(queue_id)], tape, hourly, protocol, out)
            passed.append(model)
            signals += model_signals
            reports[queue_id] = report
            print(queue_id, "ok", report["signals"], report["canonical_hash"][:12], len(report["accounting"]["problems"]), flush=True)
        except Exception as error:  # a failing model stays diagnostic and does not block the others
            (OUT / run_id).mkdir(parents=True, exist_ok=True)
            dump(OUT / run_id / f"m{queue_id}.error.json", dict(queue_id=queue_id, error=repr(error),
                                                                traceback=traceback.format_exc()))
            print(queue_id, "FAILED", repr(error)[:200], flush=True)
    stream = run_stream_capacity(STREAM, passed, signals, tape, protocol, OUT / run_id / f"stream_{STREAM}", reports)
    print("stream", STREAM, stream["canonical_hash"][:12], len(stream["problems"]), flush=True)
    print(f"RUN {run_id} COMPLETE", flush=True)


# ---------------------------------------------------------------- compare, audits, summary

def report() -> None:
    windows, models = load(OUT / "windows.json"), roster()
    audits, streams = [], {}
    suffix = AUDIT_SUFFIX[STREAM]
    for stream in OUTS:
        pair = [OUTS[stream] / run_id / f"stream_{stream}" / "report.json" for run_id in ("a", "b")]
        if all(path.exists() for path in pair):
            a, b = (load(path) for path in pair)
            streams[stream] = dict(a, replay_hash_a=a["canonical_hash"], replay_hash_b=b["canonical_hash"],
                                   replay_identical=a["canonical_hash"] == b["canonical_hash"] and a["models"] == b["models"])
    for model in models:
        queue_id, stream = model["queue_id"], model["stream"]
        if stream != STREAM and STREAM != "ES":  # the other stream's evidence is read, never rewritten
            audits.append(load(ROOT / f"docs/audits/section3_smoke_{model['model_id']}_{RUN_DATE}{AUDIT_SUFFIX[stream]}.json"))
            continue
        window = windows["streams"][stream]["models"][str(queue_id)]
        base = dict(audit=f"section3_smoke_replay_{RUN_DATE}", queue_id=queue_id, model_id=model["model_id"], stream=stream,
            protocol_sha256=windows["protocol_sha256"], b4_freeze_sha256=windows["b4_freeze_sha256"],
            route_bindings_sha256=windows["route_bindings_sha256"], windows_sha256=file_hash(OUT / "windows.json"),
            engine_version=ENGINE_VERSION, window=window, full_policy=NOT_DEFINED,
            authority=dict(scored_collection=False, paper=False, alerts=False, broker_orders=False, promotion=False),
            profitability="advisory only; recorded, not interpreted")
        paths = [model_root(queue_id) / run_id / f"m{queue_id}" / "report.json" for run_id in ("a", "b")]
        if stream != STREAM:
            a = load(paths[0]) if paths[0].exists() else {}
            audit = dict(base, status="not_run", scored=False, reason=EURUSD_NOT_RUN,
                         bytes_verified_against_b4_v2=bool(a.get("b4_freeze_files_verified")),
                         b4_freeze_files_verified=a.get("b4_freeze_files_verified"))
        elif not all(path.exists() for path in paths):
            errors = [load(p) for p in (OUT / run_id / f"m{queue_id}.error.json" for run_id in ("a", "b")) if p.exists()]
            audit = dict(base, status="failed", scored=False, classification="diagnostic_unscored",
                         bytes_verified_against_b4_v2=bool(verify_bytes(model)),
                         failures=[e["error"] for e in errors] or ["run output missing"])
        else:
            a, b = (load(path) for path in paths)
            capacity = streams.get(stream, {})
            reconcile = capacity.get("reconcile", {}).get(str(queue_id))
            capacity_problems = [p for p in capacity.get("problems", []) if p.startswith(f"{queue_id}:")]
            checks = dict(
                bytes_verified_against_b4_v2=True,
                production_runner_matches_saved_predictions=True,
                replay_hashes_identical=a["canonical_hash"] == b["canonical_hash"],
                per_trade_pnl_sums_to_ledger_totals=not any("sum to the ledger" in p or "counts differ" in p for p in a["accounting"]["problems"]),
                costs_applied_once_at_contract_rates=not any("contract rates" in p for p in a["accounting"]["problems"]),
                open_positions_carried_as_open=not any("open position" in p or "position status" in p for p in a["accounting"]["problems"]),
                independent_and_capacity_ledgers_reconcile=bool(reconcile) and not capacity_problems,
                capacity_replay_identical=bool(capacity.get("replay_identical")),
                no_cross_stream_aggregation=not any("foreign" in p for p in a["accounting"]["problems"]),
                full_policy_not_defined_without_ledger=all(v.get("status") == NOT_DEFINED for v in a["full_policy"].values())
                    and not any(".full_policy." in name for name in a["files"]))
            failures = [name for name, ok in checks.items() if not ok] + a["accounting"]["problems"] + capacity_problems
            audit = dict(base, status="passed" if not failures else "failed", scored=False, checks=checks, failures=failures,
                unit=a["unit"], calendar_id=a["calendar_id"], calendar_hash=a["calendar_hash"],
                canonical_hash_run_a=a["canonical_hash"], canonical_hash_run_b=b["canonical_hash"],
                surface_hash=a["surface_hash"], reviewed_artifact_set_sha256=a["reviewed_artifact_set_sha256"],
                b4_freeze_files_verified=a["b4_freeze_files_verified"], scoring=a["scoring"], census=a["census"],
                signals=a["signals"], open_positions_rows=a["accounting"]["unresolved_rows"],
                route_coverage=a["route_coverage"], capacity_reconcile=reconcile,
                capacity_ledger_hash=capacity.get("canonical_hash"),
                advisory_ledger_summaries=a["advisory_ledger_summaries"], files=a["files"],
                output_root=str((model_root(queue_id) / "a" / f"m{queue_id}").relative_to(ROOT)).replace("\\", "/"))
            if failures:
                audit["classification"] = "diagnostic_unscored"
        if stream == STREAM or STREAM == "ES":
            dump(ROOT / f"docs/audits/section3_smoke_{model['model_id']}_{RUN_DATE}{AUDIT_SUFFIX[stream] if stream == STREAM else ''}.json", audit)
            set_status(queue_id, audit["status"])
        audits.append(audit)
    lines = [f"# Section 3 smoke replay - {RUN_DATE[:4]}-{RUN_DATE[4:6]}-{RUN_DATE[6:]}", "",
        "Deterministic smoke replay (R4) of the 16 selected versions on engine `" + ENGINE_VERSION + "`, B4 v2 freeze, "
        "route bindings v2 and B3 protocol v3. Windows were chosen from setup presence and the calendar only and hashed "
        "before any outcome was computed. P&L and trade counts are advisory: recorded, not interpreted. All 16 full "
        "policies are `not_defined` (no ledger, no rows, no totals). ES and EURUSD are reported separately and never combined.", "",
        *[f"- {name} windows: `{(root / 'windows.json').relative_to(ROOT).as_posix()}` sha256 `{file_hash(root / 'windows.json')}`"
          for name, root in OUTS.items() if (root / "windows.json").exists() and (name == STREAM or STREAM != "ES")],
        f"- Protocol sha256 `{windows['protocol_sha256']}`; B4 v2 sha256 `{windows['b4_freeze_sha256']}`; route bindings v2 sha256 `{windows['route_bindings_sha256']}`",
        f"- Outputs: `<stream root>/a` and `/b` (two independent processes); per-model audits `docs/audits/section3_smoke_<model_id>_{RUN_DATE}.json` (ES) and `..._{RUN_DATE}_v2.json` (EURUSD, when run)", ""]
    if STREAM == "EURUSD":
        calendar = fx_calendar()
        lines += [f"EURUSD calendar: `{FX_CALENDAR.relative_to(ROOT).as_posix()}` (`{calendar['calendar_id']}`, sha256 `{file_hash(FX_CALENDAR)}`), "
                  "a provisional rule-based calendar authorized by the user on 2026-10-02 and pinned before any EURUSD outcome was computed. "
                  "The ES section below restates the unchanged first-pass evidence.", ""]
        if PASS == 2:
            lines += ["Pass 2 (user-authorized 2026-10-02): for IDs 22 and 26 the production runner also scored the reversal "
                      "confirmation bars that their saved held-out predictions omit. Those scores cannot be compared with saved "
                      "predictions; every saved bar in the same window was compared and matched. IDs 21, 25 and 23 model outputs "
                      "are unchanged from the first EURUSD pass; the EURUSD instrument-capacity ledger was rebuilt with all five models.", ""]
    for stream, unit in (("ES", "ticks"), ("EURUSD", "pips")):
        lines += [f"## {stream} ({unit})", ""]
        if all(a["status"] == "not_run" for a in audits if a["stream"] == stream):
            lines += [f"Stream status: `not_run`. {EURUSD_NOT_RUN}. Frozen bytes were verified; nothing was scored or simulated.", ""]
        elif stream in streams:
            s = streams[stream]
            lines += [f"Instrument-capacity ledger: models {s['models']}; replay identical: {s['replay_identical']}; "
                      f"hash `{s['replay_hash_a']}`; problems: {len(s['problems'])}.", ""]
            for ledger_id, summary in s["advisory_ledger_summaries"].items():
                lines.append(f"- advisory `{ledger_id}`: opportunities {summary['opportunities']}, admitted {summary['admitted']}, "
                             f"closed {summary['closed']}, open {summary['unresolved_open_exposure']}, gross {summary['gross_pnl']:.2f}, "
                             f"cost {summary['total_cost']:.2f}, net {summary['net_pnl']:.2f} {unit}")
            lines.append("")
        lines += ["| ID | Model | Status | Weeks | Routes exercised | Routes unexercised | Replay hash (a = b) |", "|---|---|---|---|---|---|---|"]
        for audit in (a for a in audits if a["stream"] == stream):
            window = audit["window"]
            status = audit["status"] + (" (diagnostic)" if audit["status"] == "failed" else "")
            digest = f"`{audit['canonical_hash_run_a'][:16]}` {'=' if audit['checks']['replay_hashes_identical'] else '!='} `{audit['canonical_hash_run_b'][:16]}`" if "checks" in audit else "-"
            lines.append(f"| {audit['queue_id']} | `{audit['model_id']}` | {status} | {', '.join(w['week'] for w in window['weeks']) or '-'} | "
                         f"{len(window['routes_exercised'])} | {len(window['routes_unexercised'])} | {digest} |")
        lines.append("")
        for audit in (a for a in audits if a["stream"] == stream):
            window = audit["window"]
            lines.append(f"### ID {audit['queue_id']} `{audit['model_id']}` - {audit['status']}")
            lines.append("")
            if audit.get("failures"):
                lines.append("- Failures: " + "; ".join(audit["failures"]))
            if audit["status"] == "not_run":
                lines.append("- Unexercised routes (stream not run): " + ", ".join(f"`{r}`" for r in window["routes_unexercised"]))
                lines.append("")
                continue
            lines.append("- Exercised routes: " + (", ".join(f"`{r}`" for r in window["routes_exercised"]) or "none"))
            lines.append("- Unexercised routes: " + (", ".join(f"`{r}` ({reason})" for r, reason in window["routes_unexercised"].items()) or "none"))
            lines.append("- Unexercised calendar elements: " + (", ".join(f"{e} ({reason})" for e, reason in window["calendar_elements_unexercised"].items()) or "none"))
            if "checks" in audit:
                lines.append(f"- Signals {audit['signals']}; census {audit['census']}; open-position rows {audit['open_positions_rows']}; "
                             f"capacity reconcile {audit['capacity_reconcile']}")
                for ledger_id, summary in audit["advisory_ledger_summaries"].items():
                    if ledger_id.endswith(".base"):
                        lines.append(f"- advisory `{ledger_id}`: admitted {summary['admitted']}, closed {summary['closed']}, "
                                     f"open {summary['unresolved_open_exposure']}, gross {summary['gross_pnl']:.2f}, "
                                     f"cost {summary['total_cost']:.2f}, net {summary['net_pnl']:.2f} {unit}")
                lines.append(f"- Surface 4 `full_policy`: `{NOT_DEFINED}`")
            lines.append("")
    (ROOT / f"docs/section3_smoke_replay_{RUN_DATE}{suffix}.md").write_text("\n".join(lines), encoding="utf-8")
    dump(OUT / "summary.json", dict(models={str(a["queue_id"]): a["status"] for a in audits},
        streams={k: dict(replay_identical=v["replay_identical"], problems=v["problems"], canonical_hash=v["canonical_hash"]) for k, v in streams.items()}))
    print({a["queue_id"]: a["status"] for a in audits})
    print("REPORT COMPLETE")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("setup", "run", "report"))
    parser.add_argument("--run", choices=("a", "b"), default="a")
    parser.add_argument("--stream", choices=tuple(OUTS), default="ES")
    parser.add_argument("--pass", dest="pass_", type=int, choices=(1, 2), default=1)
    args = parser.parse_args()
    if args.stream == "EURUSD" and args.pass_ == 2:
        PASS, OUTS["EURUSD"], AUDIT_SUFFIX["EURUSD"] = 2, FX_PASS2_ROOT, "_v3"
    STREAM, OUT = args.stream, OUTS[args.stream]
    {"setup": setup, "run": lambda: run(args.run), "report": report}[args.stage]()
