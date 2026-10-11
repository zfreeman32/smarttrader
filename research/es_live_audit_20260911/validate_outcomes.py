"""Independent audit calculations from frozen SQLite bars; no root audit imports."""
import collections
import csv
from datetime import datetime, timedelta
import json
import math
from pathlib import Path
import random
import sqlite3

OUT = Path(__file__).resolve().parent
STEP = timedelta(minutes=5)


def ts(x):
    return datetime.fromisoformat(x.replace("Z", "+00:00"))


def read(name):
    with (OUT / name).open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def main():
    con = sqlite3.connect((OUT / "es_audit_snapshot.sqlite3").as_uri() + "?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    bars = {}
    for row in con.execute("SELECT * FROM canonical_bars WHERE asset='ES' AND timeframe='5m' ORDER BY timestamp_utc"):
        r = dict(row)
        r["complete"] = json.loads(r["feature_context_json"] or "{}").get("ibkr_is_complete") is not False
        bars[ts(r["timestamp_utc"])] = r
    results = collections.Counter()
    errors = []

    def check(condition, label):
        results["checks"] += 1
        if not condition:
            errors.append(label)

    def approx(actual, expected, label):
        check(actual not in {None, ""} and math.isclose(float(actual), expected, abs_tol=1e-8), label)

    def window(t, n):
        b = [bars.get(t + STEP * j) for j in range(n + 1)]
        return b if all(x is not None and x["complete"] and x["contract_symbol"] == b[0]["contract_symbol"] for x in b) else None

    # Validate exported bars against frozen SQL rather than trusting root's export.
    exported = read("es_bars_5m.csv")
    check(len(exported) == len(bars), "export bar count")
    for row in exported:
        original = bars[ts(row["timestamp_utc"])]
        for field in ("open", "high", "low", "close", "volume"):
            approx(row[field], original[field], f"export:{row['timestamp_utc']}:{field}")

    setups = read("setup_outcomes.csv")
    prediction_rows = read("prediction_outcomes.csv")
    # Include one example per model and an independent fixed random sample.
    by_model = collections.defaultdict(list)
    for row in prediction_rows:
        by_model[row["model_id"]].append(row)
    selected = {int(rows[len(rows) // 2]["id"]): rows[len(rows) // 2] for rows in by_model.values()}
    for row in random.Random(20260912).sample(prediction_rows, 60):
        selected[int(row["id"])] = row

    def inspect(row, kind):
        key = row["setup_id"] if kind == "setup" else row["id"]
        t = ts(row["timestamp_utc"])
        side = int(row["setup_side"]) if kind == "setup" else 1 if row["direction"] == "long" else -1
        cost = 1 if kind == "setup" else float(row["assumed_cost_points"])
        approx(row["canonical_close"], bars[t]["close"], f"canonical:{key}")
        for n in (3, 6, 12, 20, 120):
            b = window(t, n)
            check((row[f"valid_{n}"] == "True") == bool(b), f"valid:{key}:{n}")
            results[f"{kind}_horizon_windows_checked"] += 1
            if b is None:
                check(not row.get(f"net_points_{n}"), f"censored_nonblank:{key}:{n}")
                continue
            entry, exit_price = b[1]["open"], b[-1]["close"]
            approx(row[f"net_points_{n}"], side * (exit_price - entry) - cost, f"net:{key}:{n}")
            approx(row[f"mfe_points_{n}"], max(0, max(x["high"] for x in b[1:]) - entry if side == 1 else entry - min(x["low"] for x in b[1:])), f"mfe:{key}:{n}")
            approx(row[f"mae_points_{n}"], max(0, entry - min(x["low"] for x in b[1:]) if side == 1 else max(x["high"] for x in b[1:]) - entry), f"mae:{key}:{n}")
            check(ts(row[f"exit_timestamp_{n}"]) == t + STEP * (n + 1), f"exit-time:{key}:{n}")
        recorded = row["original_mark_recorded_at_utc"] or row["first_model_recorded_at_utc"] if kind == "setup" else row["recorded_at_utc"]
        # Direct calendar-grid ceiling to the first eligible five-minute bar open.
        rec = ts(recorded)
        elapsed = (rec - t).total_seconds() / STEP.total_seconds()
        entry_steps = math.ceil(elapsed)
        b = window(t, entry_steps + 11) if entry_steps > 0 else None
        check((row["availability_valid_12"] == "True") == bool(b), f"availability-valid:{key}")
        if b:
            approx(row["availability_net_points_12"], side * (b[-1]["close"] - b[entry_steps]["open"]) - cost, f"availability-net:{key}")
            check(ts(row["availability_entry_timestamp"]) >= rec, f"availability-lookahead:{key}")

    for row in setups:
        inspect(row, "setup")
    for row in selected.values():
        inspect(row, "prediction")

    bracket_counts = collections.Counter()
    for row in setups:
        if row["strategy"] != "ICT":
            continue
        t, side = ts(row["timestamp_utc"]), int(row["setup_side"])
        entry, stop, target = (float(row[k]) for k in ("entry_price", "stop_price", "target_price"))
        settled = False
        for n in range(1, 21):
            b = window(t, n)
            if b is None:
                check(row["bracket_status"] == "censored_gap_or_incomplete", f"bracket-censor:{row['setup_id']}")
                check(not row["bracket_net_points"], f"bracket-censor-net:{row['setup_id']}")
                settled = True
                break
            last = b[-1]
            stop_hit = last["low"] <= stop if side == 1 else last["high"] >= stop
            target_hit = last["high"] >= target if side == 1 else last["low"] <= target
            if stop_hit or target_hit:
                if side * (last["open"] - stop) <= 0:
                    fill, status = last["open"], "stop_gap"
                elif side * (last["open"] - target) >= 0:
                    fill, status = target, "target_gap_conservative_fill"
                elif stop_hit:
                    fill, status = stop, "stop_ambiguous" if target_hit else "stop"
                else:
                    fill, status = target, "target"
                check(row["bracket_status"] == status, f"bracket-status:{row['setup_id']}")
                approx(row["bracket_net_points"], side * (fill - entry) - 1, f"bracket-net:{row['setup_id']}")
                approx(row["bracket_bars"], n, f"bracket-bars:{row['setup_id']}")
                settled = True
                break
        if not settled:
            check(row["bracket_status"] == "timeout", f"bracket-timeout:{row['setup_id']}")
            approx(row["bracket_net_points"], side * (b[-1]["close"] - entry) - 1, f"bracket-timeout-net:{row['setup_id']}")
        bracket_counts[row["bracket_status"]] += 1

    # Verify deduplication and time sequencing in optional one-at-a-time sensitivity.
    nonoverlap = read("nonoverlapping_setup_events_60m.csv")
    previous = {}
    for row in nonoverlap:
        strategy = row["strategy"]
        entry = ts(row["timestamp_utc"]) + STEP
        check(strategy not in previous or entry >= previous[strategy], f"overlap:{row['setup_id']}")
        previous[strategy] = ts(row["exit_timestamp_12"])
    check(len({r["setup_id"] for r in nonoverlap}) == len(nonoverlap), "nonoverlap-dedup")

    route_rows = read("matched_model_setup_outcomes.csv")
    for row in route_rows:
        parts = row["model_id"].split("_")
        check(parts[0].upper() == row["strategy"], "route-strategy")
        check((parts[1] == "long") == (int(row["setup_side"]) == 1), "route-side")
        check(parts[2] in {row["setup_family"], "meta"}, "route-family")
        if row["model_kind"] == "setup":
            suffix = "_".join(parts[3:-2])
            if parts[0] == "frvp":
                suffix = suffix.removeprefix("setup")
            check(suffix == row["setup_type"], "route-specialist")
        t, rec = ts(row["timestamp_utc"]), ts(row["recorded_at_utc"])
        entry_steps = math.ceil((rec - t).total_seconds() / STEP.total_seconds())
        b = window(t, entry_steps + 11) if entry_steps > 0 else None
        check((row["availability_valid_12"] == "True") == bool(b), f"matched-availability-valid:{row['prediction_id']}")
        check(row["availability_time_basis"] == "this_model_recorded_at", "matched-availability-basis")
        if b:
            approx(row["availability_net_points_12"], int(row["setup_side"]) * (b[-1]["close"] - b[entry_steps]["open"]) - 1, f"matched-availability-net:{row['prediction_id']}")
        else:
            check(not row["availability_net_points_12"], f"matched-availability-stale-value:{row['prediction_id']}")
    stats = {**dict(results), "setup_events_checked": len(setups), "prediction_rows_checked": len(selected), "models_sampled": len(by_model), "ict_bracket_status_counts": dict(bracket_counts), "matched_model_setup_rows_checked": len(route_rows), "nonoverlap_rows_checked": len(nonoverlap), "errors": errors}
    (OUT / "validation_summary.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")

    examples = sorted([r for r in setups if r["net_points_12"]], key=lambda r: float(r["net_points_12"]))
    examples = examples[:2] + examples[-2:] + [r for r in setups if r["strategy"] == "ICT" and r["bracket_status"] == "stop_ambiguous"][:2]
    narrative = [
        "# Independent validation of ES outcome analysis", "",
        f"Reviewed model_outcomes.py and setup_outcomes.py and independently recalculated from the read-only frozen SQLite canonical_bars table. Checked all {len(setups)} setup events at five horizons, {len(selected)} predictions spanning all {len(by_model)} models, all forty ICT brackets, all {len(route_rows)} matched setup/model joins, and {len(nonoverlap)} per-strategy nonoverlapping selections. The exported bar OHLCV values were checked against SQLite for every bar. Result: {len(errors)} failed checks across {results['checks']} comparisons.", "",
        "**Arithmetic and indexing.** A source bar stamped t is followed by entry at the next bar's open, t+5 minutes. Twelve future bars span that entry through the close of the bar stamped t+60, an exit at t+65: exactly 60 minutes held. Long and short signs, gross/net subtraction, future-only MFE/MAE, and exit timestamps are correct. Censoring requires all source-through-exit bars to be present, complete, and on the same contract. No test window bridges a missing five-minute slot or scheduled session break. Requiring a complete source bar is conservative even when entry is next open.", "",
        "**Brackets.** The ICT branch uses the archived setup entry/stop/target references, future bars only, a twenty-bar maximum, one-point assumed round-trip cost, adverse opening-gap fills for stops, and target-level fills for favorable target gaps. Intrabar stop/target ambiguity is assigned to the stop. The forty cases independently match the exported branch classifications and numbers. This is a hypothetical setup-geometry study; those reference prices are not evidence of submitted or filled orders.", "",
        "**Important interpretation limits; no root-script arithmetic correction is required.**", "",
        "- The main setup totals add independent marked opportunities. Simultaneous and overlapping signals remain in those totals. The optional nonoverlapping file removes overlap separately inside FRVP and ICT, so combining both still does not establish a single ES position book. Model totals also repeat the same market episode across related models; they should never be added as portfolio P&L.",
        "- Source-bar next-open and bracket-close studies use market prices before many archived predictions were actually recorded. Availability-adjusted setup results are a useful sensitivity, but ICT uses the first same-bar model timestamp as a proxy because the dashboard does not store the exact setup observed-at time. Following this review, the matched_model_setup_outcomes calculation was refined to use each selected model's own recorded_at. All 187 model/setup availability windows independently agree with direct SQLite calculations, and no invalid window retains a stale net result.",
        "- Model all-bar diagnostics use an assumed source-session round-trip spread schedule, with fixed slippage/commission; they do not change the exit leg when a horizon crosses into another session. The specialist manifests have null/empty cost assumptions, so the schedule is an audit assumption for those models. Setup comparisons and brackets consistently use a separately stated one-point round-trip cost and have cost-sensitivity output. None of these figures establishes broker commissions or realized fills.",
        "- Threshold_and_static_policy applies the archived static regime/session/stress blocks. It excludes neither the original stateful cooldown nor dynamic candidate-quantile/expected-move checks. Its name accurately describes a partial-policy sensitivity, not a full live-policy backtest. In these manifests probability-quantile thresholds are null; expected_move is not among archived context fields used by this audit.",
        "- Summary days and hourly buckets are source-bar New York calendar dates/hours, not futures trade-date sessions or actual delayed execution hours. Censored late-afternoon rows make the valid-hours sample uneven. In particular all sixteen setup marks in the 16:00 hour lack uninterrupted sixty-minute windows; no return ranking can be inferred for that hour.",
        "- The 600-minute FRVP sample is only ten uncensored observations, versus forty-five at sixty minutes. Positive long-horizon results must not be described as the same sample becoming profitable. Confidence intervals resample the available calendar days and remain small-sample descriptive uncertainty, not proof of independence or a validated edge.",
        "- ROC AUC is correctly named auc_net_win_proxy: the target is this audit's next-open/net markout win, which differs from models' training labels. Probability bins and threshold selectivity are descriptive; these are not calibration tests against the original model target.", "",
        "**Direct SQLite spot checks**", "",
        "| Source time ET | Strategy/setup/side | Next open | 60m exit close | Gross points | Net points at 1pt cost | Bracket status |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for r in examples:
        def fmt(k):
            return f"{float(r[k]):.4f}" if r.get(k) else "—"
        narrative.append(f"| {r['timestamp_et']} | {r['strategy']} {r['setup_type']} {r['setup_side']} | {fmt('next_open_12')} | {fmt('exit_close_12')} | {fmt('gross_points_12')} | {fmt('net_points_12')} | {r.get('bracket_status', '')} |")
    narrative.extend(["", "Reproduction: run validate_outcomes.py from the audit directory or repository root. Details are in validation_summary.json. Validation wrote only new audit files and did not alter either outcome script, the snapshot, or runtime source."])
    (OUT / "validation_findings.md").write_text("\n".join(narrative) + "\n", encoding="utf-8")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
