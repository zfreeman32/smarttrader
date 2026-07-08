from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from frvp.sessions.equity import build_equity_market_day_labels


DEFAULT_SOURCE_CSV = Path("data/futures_data/ES-5m.csv")
DEFAULT_SYMBOLOGY_PAYLOAD = Path("tmp/es_symbology_payload.json")
DEFAULT_DEFINITION_CSV = Path("tmp/es_definition_records.csv")
DEFAULT_SCHEDULE_JSON = Path("data/futures_data/es_roll_schedule.json")
DEFAULT_TAGGED_PARQUET = Path("data/futures_data/ES-5m-tagged.parquet")
DEFAULT_TAGGED_CSV = Path("data/futures_data/ES-5m-tagged.csv")
DEFAULT_REPORT_JSON = Path("data/futures_data/es_roll_reconstruction_report.json")


@dataclass(frozen=True)
class ReconstructionPaths:
    source_csv: Path = DEFAULT_SOURCE_CSV
    symbology_payload_json: Path = DEFAULT_SYMBOLOGY_PAYLOAD
    definition_csv: Path = DEFAULT_DEFINITION_CSV
    schedule_json: Path = DEFAULT_SCHEDULE_JSON
    tagged_parquet: Path = DEFAULT_TAGGED_PARQUET
    tagged_csv: Path = DEFAULT_TAGGED_CSV
    report_json: Path = DEFAULT_REPORT_JSON


def load_source_bars(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"ts_event", "open", "high", "low", "close", "volume", "symbol", "instrument_id"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Source CSV is missing required columns: {sorted(missing)}")

    working = df.copy()
    working["ts_event"] = pd.to_datetime(working["ts_event"], utc=True, errors="raise")
    for column in ("open", "high", "low", "close", "volume"):
        working[column] = pd.to_numeric(working[column], errors="raise")
    working["instrument_id"] = pd.to_numeric(working["instrument_id"], errors="raise").astype("Int64")
    working["symbol"] = working["symbol"].astype(str)
    working = working.sort_values("ts_event", kind="stable").reset_index(drop=True)
    return working


def dominant_bar_spacing(timestamps: pd.Series) -> pd.Timedelta:
    diffs = timestamps.diff().dropna()
    diffs = diffs[diffs > pd.Timedelta(0)]
    if diffs.empty:
        raise ValueError("Need at least two timestamps to infer bar spacing.")
    return pd.Timedelta(diffs.mode().iloc[0])


def build_observed_segments(source_bars: pd.DataFrame) -> pd.DataFrame:
    segment_id = source_bars["instrument_id"].ne(source_bars["instrument_id"].shift()).cumsum()
    spacing = dominant_bar_spacing(source_bars["ts_event"])
    segments = (
        source_bars.groupby(segment_id, sort=True)
        .agg(
            instrument_id=("instrument_id", "first"),
            start=("ts_event", "min"),
            end_inclusive=("ts_event", "max"),
            rows=("ts_event", "size"),
        )
        .reset_index(drop=True)
    )
    segments["end"] = segments["start"].shift(-1)
    if not segments.empty:
        segments.loc[len(segments) - 1, "end"] = segments.loc[len(segments) - 1, "end_inclusive"] + spacing
    return segments


def load_cached_symbology_payload(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_definition_records(path: Path) -> pd.DataFrame:
    definitions = pd.read_csv(path)
    required = {"instrument_id", "raw_symbol", "expiration"}
    missing = required - set(definitions.columns)
    if missing:
        raise KeyError(f"Definition CSV is missing required columns: {sorted(missing)}")

    definitions = definitions.copy()
    definitions["instrument_id"] = pd.to_numeric(definitions["instrument_id"], errors="raise").astype("Int64")
    definitions["raw_symbol"] = definitions["raw_symbol"].astype(str)
    definitions["expiration"] = pd.to_datetime(definitions["expiration"], utc=True, errors="raise")
    return definitions


def resolve_segment_raw_symbol(
    instrument_id: int,
    segment_start: pd.Timestamp,
    symbology_payload: dict[str, Any],
) -> str:
    raw_result = symbology_payload["resolve_instrument_id_to_raw_symbol"]["result"]
    instrument_key = str(instrument_id)
    if instrument_key not in raw_result:
        raise KeyError(f"instrument_id {instrument_id} is absent from the cached raw-symbol resolve payload.")

    start_date = segment_start.date().isoformat()
    matches = [
        item["s"]
        for item in raw_result[instrument_key]
        if item["d0"] <= start_date < item["d1"]
    ]
    if len(matches) != 1:
        raise ValueError(
            "Could not derive a unique raw_symbol from the cached symbology payload "
            f"for instrument_id={instrument_id} at segment start {start_date}: {matches}"
        )
    return str(matches[0])


def resolve_segment_expiration(
    instrument_id: int,
    raw_symbol: str,
    definitions: pd.DataFrame,
) -> pd.Timestamp:
    matches = definitions.loc[
        (definitions["instrument_id"].astype(int) == int(instrument_id))
        & (definitions["raw_symbol"] == raw_symbol),
        "expiration",
    ]
    unique_expirations = sorted(set(pd.Timestamp(value) for value in matches.tolist()))
    if len(unique_expirations) != 1:
        raise ValueError(
            "Could not derive a unique expiration from the cached definition records "
            f"for instrument_id={instrument_id}, raw_symbol={raw_symbol}: {unique_expirations}"
        )
    return unique_expirations[0]


def build_schedule_from_cache(
    source_bars: pd.DataFrame,
    observed_segments: pd.DataFrame,
    symbology_payload: dict[str, Any],
    definitions: pd.DataFrame,
) -> pd.DataFrame:
    continuous_segments = symbology_payload["resolve_continuous_to_instrument_id"]["result"]["ES.v.0"]
    if len(continuous_segments) != len(observed_segments):
        raise ValueError(
            f"Continuous resolve returned {len(continuous_segments)} segments but the CSV shows {len(observed_segments)}."
        )

    rows: list[dict[str, Any]] = []
    for segment, continuous_segment in zip(
        observed_segments.itertuples(index=False),
        continuous_segments,
        strict=True,
    ):
        instrument_id = int(segment.instrument_id)
        resolved_instrument_id = int(continuous_segment["s"])
        if resolved_instrument_id != instrument_id:
            raise ValueError(
                "Cached continuous resolve instrument_id does not match observed CSV segment: "
                f"{resolved_instrument_id} vs {instrument_id}"
            )
        start = pd.Timestamp(segment.start)
        raw_symbol = resolve_segment_raw_symbol(
            instrument_id=instrument_id,
            segment_start=start,
            symbology_payload=symbology_payload,
        )
        expiration = resolve_segment_expiration(
            instrument_id=instrument_id,
            raw_symbol=raw_symbol,
            definitions=definitions,
        )
        rows.append(
            {
                "instrument_id": instrument_id,
                "raw_symbol": raw_symbol,
                "expiration": expiration,
                "start": pd.Timestamp(f"{continuous_segment['d0']}T00:00:00Z"),
                "end": pd.Timestamp(f"{continuous_segment['d1']}T00:00:00Z"),
                "observed_start": start,
                "observed_end_exclusive": pd.Timestamp(segment.end),
                "end_inclusive": pd.Timestamp(segment.end_inclusive),
                "rows": int(segment.rows),
            }
        )

    schedule = pd.DataFrame(rows)
    schedule["roll_date"] = schedule["start"].shift(-1)
    return schedule


def _assert_continuous_payload_matches_observed_schedule(
    schedule: pd.DataFrame,
    symbology_payload: dict[str, Any],
) -> None:
    continuous_result = symbology_payload["resolve_continuous_to_instrument_id"]["result"]["ES.v.0"]
    if len(continuous_result) != len(schedule):
        raise ValueError(
            f"Continuous resolve returned {len(continuous_result)} segments but the CSV shows {len(schedule)}."
        )

    for payload_item, schedule_row in zip(continuous_result, schedule.itertuples(index=False), strict=True):
        observed_start_date = pd.Timestamp(schedule_row.observed_start).date().isoformat()
        observed_end_exclusive = pd.Timestamp(schedule_row.observed_end_exclusive)
        if observed_end_exclusive == observed_end_exclusive.normalize():
            observed_end_date = observed_end_exclusive.date().isoformat()
        else:
            observed_end_date = (
                observed_end_exclusive.normalize() + pd.Timedelta(days=1)
            ).date().isoformat()
        expected_instrument = str(int(schedule_row.instrument_id))
        if payload_item["d0"] != observed_start_date:
            raise ValueError(
                "Cached continuous resolve start date does not match observed CSV segment start: "
                f"{payload_item['d0']} vs {observed_start_date}"
            )
        if payload_item["s"] != expected_instrument:
            raise ValueError(
                "Cached continuous resolve instrument_id does not match observed CSV segment: "
                f"{payload_item['s']} vs {expected_instrument}"
            )
        if payload_item["d1"] != observed_end_date:
            raise ValueError(
                "Cached continuous resolve end date does not match observed CSV segment end: "
                f"{payload_item['d1']} vs {observed_end_date}"
            )


def write_schedule_json(schedule: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [
        {
            "instrument_id": int(row.instrument_id),
            "raw_symbol": str(row.raw_symbol),
            "expiration": pd.Timestamp(row.expiration).isoformat(),
            "start": pd.Timestamp(row.start).isoformat(),
            "end": pd.Timestamp(row.end).isoformat(),
        }
        for row in schedule.itertuples(index=False)
    ]
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def tag_bars(
    source_bars: pd.DataFrame,
    schedule: pd.DataFrame,
    *,
    roll_bracket_sessions: int,
) -> pd.DataFrame:
    tag_map = schedule.loc[:, ["instrument_id", "raw_symbol", "expiration"]].copy()
    tag_map = tag_map.rename(
        columns={
            "raw_symbol": "contract_symbol",
            "expiration": "contract_expiration",
        }
    )

    tagged = source_bars.merge(tag_map, on="instrument_id", how="left", validate="many_to_one")
    if tagged["contract_symbol"].isna().any() or tagged["contract_expiration"].isna().any():
        raise ValueError("At least one bar could not be enriched with contract metadata.")

    tagged["is_roll_boundary"] = tagged["contract_symbol"].ne(tagged["contract_symbol"].shift()).fillna(False)
    if not tagged.empty:
        tagged.loc[0, "is_roll_boundary"] = False
    segment_id = tagged["contract_symbol"].ne(tagged["contract_symbol"].shift()).cumsum()
    tagged["bars_since_roll"] = tagged.groupby(segment_id).cumcount().astype(int)

    market_day = build_equity_market_day_labels(
        tagged["ts_event"],
        source_timezone="UTC",
        canonical_timezone="UTC",
    )
    tagged["market_day_close"] = market_day
    session_codes, session_uniques = pd.factorize(market_day, sort=True)
    tagged["market_day_index"] = session_codes.astype(int)

    roll_session_indices = sorted(
        set(
            tagged.loc[tagged["is_roll_boundary"], "market_day_index"].astype(int).tolist()
        )
    )
    if roll_session_indices:
        distance_to_roll = [
            min(abs(session_index - roll_session) for roll_session in roll_session_indices)
            for session_index in tagged["market_day_index"].tolist()
        ]
        tagged["in_roll_bracket"] = pd.Series(distance_to_roll, index=tagged.index).le(roll_bracket_sessions)
    else:
        tagged["in_roll_bracket"] = False

    tagged["contract_expiration"] = pd.to_datetime(tagged["contract_expiration"], utc=True, errors="raise")
    return tagged


def write_tagged_bars(
    tagged: pd.DataFrame,
    *,
    parquet_path: Path,
    csv_path: Path,
) -> tuple[Path, str]:
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        tagged.to_parquet(parquet_path, index=False)
        return parquet_path, "parquet"
    except Exception:
        tagged.to_csv(csv_path, index=False)
        return csv_path, "csv"


def build_verification_report(
    source_bars: pd.DataFrame,
    schedule: pd.DataFrame,
    tagged: pd.DataFrame,
) -> dict[str, Any]:
    coverage = build_coverage_report(source_bars=source_bars, schedule=schedule, tagged=tagged)
    expiration = build_expiration_report(schedule=schedule)
    cadence = build_roll_cadence_report(schedule=schedule)
    seams = build_seam_report(tagged=tagged)
    contract_summary = build_contract_summary(schedule=schedule)

    return {
        "coverage": coverage,
        "expiration_monotonicity": expiration,
        "roll_cadence": cadence,
        "seam_report": seams,
        "contract_summary": contract_summary,
        "all_checks_passed": bool(
            coverage["all_bars_tagged"]
            and coverage["schedule_contiguous"]
            and coverage["schedule_covers_source_range"]
            and expiration["monotonic_increasing"]
            and cadence["plausible_full_year_roll_counts"]
            and cadence["plausible_days_before_expiration"]
            and seams["boundary_count_matches_schedule"]
        ),
    }


def build_coverage_report(
    *,
    source_bars: pd.DataFrame,
    schedule: pd.DataFrame,
    tagged: pd.DataFrame,
) -> dict[str, Any]:
    schedule_sorted = schedule.sort_values("start", kind="stable").reset_index(drop=True)
    contiguous = True
    gap_or_overlap_examples: list[dict[str, str]] = []
    for left, right in zip(
        schedule_sorted.itertuples(index=False),
        schedule_sorted.iloc[1:].itertuples(index=False),
        strict=False,
    ):
        if pd.Timestamp(left.end) != pd.Timestamp(right.start):
            contiguous = False
            gap_or_overlap_examples.append(
                {
                    "left_end": pd.Timestamp(left.end).isoformat(),
                    "right_start": pd.Timestamp(right.start).isoformat(),
                }
            )

    spacing = dominant_bar_spacing(source_bars["ts_event"])
    source_start = pd.Timestamp(source_bars["ts_event"].min())
    source_end_exclusive = pd.Timestamp(source_bars["ts_event"].max()) + spacing
    schedule_covers_source_range = (
        not schedule_sorted.empty
        and pd.Timestamp(schedule_sorted["start"].iloc[0]) <= source_start
        and pd.Timestamp(schedule_sorted["end"].iloc[-1]) >= source_end_exclusive
    )

    return {
        "all_bars_tagged": bool(tagged["contract_symbol"].notna().all() and tagged["contract_expiration"].notna().all()),
        "schedule_contiguous": bool(contiguous),
        "schedule_covers_source_range": bool(schedule_covers_source_range),
        "source_start": source_start.isoformat(),
        "source_end_exclusive": source_end_exclusive.isoformat(),
        "schedule_start": pd.Timestamp(schedule_sorted["start"].iloc[0]).isoformat(),
        "schedule_end": pd.Timestamp(schedule_sorted["end"].iloc[-1]).isoformat(),
        "gap_or_overlap_examples": gap_or_overlap_examples[:10],
    }


def build_expiration_report(schedule: pd.DataFrame) -> dict[str, Any]:
    expirations = pd.to_datetime(schedule["expiration"], utc=True, errors="raise")
    monotonic = bool(expirations.is_monotonic_increasing)
    deltas = expirations.diff().dropna().dt.total_seconds().div(86400.0)
    return {
        "monotonic_increasing": monotonic,
        "expiration_deltas_days": [round(float(value), 3) for value in deltas.tolist()],
    }


def build_roll_cadence_report(schedule: pd.DataFrame) -> dict[str, Any]:
    roll_rows = schedule.iloc[:-1].copy()
    if roll_rows.empty:
        return {
            "plausible_full_year_roll_counts": True,
            "full_year_roll_counts": {},
            "days_before_expiration": [],
            "plausible_days_before_expiration": True,
            "days_before_expiration_outliers": [],
        }

    roll_rows["roll_date"] = pd.to_datetime(roll_rows["end"], utc=True, errors="raise")
    roll_rows["expiration"] = pd.to_datetime(roll_rows["expiration"], utc=True, errors="raise")
    roll_rows["days_before_expiration"] = (
        roll_rows["expiration"] - roll_rows["roll_date"]
    ).dt.total_seconds().div(86400.0)

    counts_by_year = roll_rows.groupby(roll_rows["roll_date"].dt.year).size().to_dict()
    years = sorted(counts_by_year)
    full_year_counts = {
        int(year): int(count)
        for year, count in counts_by_year.items()
        if year not in {years[0], years[-1]}
    } if len(years) >= 2 else {}
    plausible_full_year_roll_counts = all(3 <= count <= 5 for count in full_year_counts.values())

    days_before_expiration = [round(float(value), 3) for value in roll_rows["days_before_expiration"].tolist()]
    outliers = [
        {
            "roll_date": pd.Timestamp(row.roll_date).isoformat(),
            "raw_symbol": str(row.raw_symbol),
            "days_before_expiration": round(float(row.days_before_expiration), 3),
        }
        for row in roll_rows.itertuples(index=False)
        if not (1.0 <= float(row.days_before_expiration) <= 14.0)
    ]

    return {
        "plausible_full_year_roll_counts": bool(plausible_full_year_roll_counts),
        "full_year_roll_counts": full_year_counts,
        "days_before_expiration": days_before_expiration,
        "plausible_days_before_expiration": len(outliers) == 0,
        "days_before_expiration_outliers": outliers,
    }


def build_seam_report(tagged: pd.DataFrame) -> dict[str, Any]:
    roll_indices = tagged.index[tagged["is_roll_boundary"]].tolist()
    seam_rows: list[dict[str, Any]] = []
    for index in roll_indices:
        previous_row = tagged.iloc[index - 1]
        current_row = tagged.iloc[index]
        step = float(current_row["open"] - previous_row["close"])
        seam_rows.append(
            {
                "roll_date": pd.Timestamp(current_row["ts_event"]).isoformat(),
                "from_symbol": str(previous_row["contract_symbol"]),
                "to_symbol": str(current_row["contract_symbol"]),
                "price_step": round(step, 6),
                "abs_price_step": round(abs(step), 6),
            }
        )

    abs_steps = np.array([row["abs_price_step"] for row in seam_rows], dtype=float)
    if abs_steps.size:
        median = float(np.median(abs_steps))
        mad = float(np.median(np.abs(abs_steps - median)))
        threshold = max(10.0, median + (5.0 * mad))
    else:
        median = 0.0
        mad = 0.0
        threshold = 10.0

    flagged = [row for row in seam_rows if row["abs_price_step"] > threshold]
    return {
        "boundary_count_matches_schedule": len(roll_indices) == max(len(tagged["contract_symbol"].drop_duplicates()) - 1, 0),
        "boundary_count": len(roll_indices),
        "median_abs_price_step": round(median, 6),
        "mad_abs_price_step": round(mad, 6),
        "flag_threshold_abs_price_step": round(threshold, 6),
        "flagged_boundaries": flagged,
        "seam_table": seam_rows,
    }


def build_contract_summary(schedule: pd.DataFrame) -> dict[str, Any]:
    bars_per_contract = {
        str(row.raw_symbol): int(row.rows)
        for row in schedule.itertuples(index=False)
    }
    spans = [
        {
            "contract_symbol": str(row.raw_symbol),
            "instrument_id": int(row.instrument_id),
            "start": pd.Timestamp(row.observed_start).isoformat(),
            "end": pd.Timestamp(row.end_inclusive).isoformat(),
            "expiration": pd.Timestamp(row.expiration).isoformat(),
            "rows": int(row.rows),
        }
        for row in schedule.itertuples(index=False)
    ]
    roll_dates = [
        {
            "from_symbol": str(current.raw_symbol),
            "to_symbol": str(next_row.raw_symbol),
            "roll_date": pd.Timestamp(next_row.start).isoformat(),
        }
        for current, next_row in zip(
            schedule.itertuples(index=False),
            schedule.iloc[1:].itertuples(index=False),
            strict=False,
        )
    ]
    return {
        "bars_per_contract": bars_per_contract,
        "date_span_per_contract": spans,
        "roll_dates": roll_dates,
    }


def write_report_json(report: dict[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(report), indent=2), encoding="utf-8")
    return path


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        if np.isnan(value):
            return None
        return float(value)
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if pd.isna(value):
        return None
    return str(value)


def reconstruct_boundaries(
    *,
    paths: ReconstructionPaths = ReconstructionPaths(),
    roll_bracket_sessions: int = 3,
) -> dict[str, Any]:
    source_bars = load_source_bars(paths.source_csv)
    observed_segments = build_observed_segments(source_bars)
    symbology_payload = load_cached_symbology_payload(paths.symbology_payload_json)
    definitions = load_definition_records(paths.definition_csv)

    schedule = build_schedule_from_cache(
        source_bars=source_bars,
        observed_segments=observed_segments,
        symbology_payload=symbology_payload,
        definitions=definitions,
    )
    _assert_continuous_payload_matches_observed_schedule(schedule, symbology_payload)

    tagged = tag_bars(
        source_bars=source_bars,
        schedule=schedule,
        roll_bracket_sessions=roll_bracket_sessions,
    )
    report = build_verification_report(source_bars=source_bars, schedule=schedule, tagged=tagged)

    schedule_path = write_schedule_json(schedule, paths.schedule_json)
    tagged_path, tagged_format = write_tagged_bars(
        tagged,
        parquet_path=paths.tagged_parquet,
        csv_path=paths.tagged_csv,
    )
    report_path = write_report_json(report, paths.report_json)

    return {
        "schedule_path": schedule_path,
        "tagged_path": tagged_path,
        "tagged_format": tagged_format,
        "report_path": report_path,
        "report": report,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Reconstruct ES continuous-contract boundaries from cached Databento metadata.")
    parser.add_argument("--source-csv", type=Path, default=DEFAULT_SOURCE_CSV)
    parser.add_argument("--symbology-payload", type=Path, default=DEFAULT_SYMBOLOGY_PAYLOAD)
    parser.add_argument("--definition-csv", type=Path, default=DEFAULT_DEFINITION_CSV)
    parser.add_argument("--schedule-json", type=Path, default=DEFAULT_SCHEDULE_JSON)
    parser.add_argument("--tagged-parquet", type=Path, default=DEFAULT_TAGGED_PARQUET)
    parser.add_argument("--tagged-csv", type=Path, default=DEFAULT_TAGGED_CSV)
    parser.add_argument("--report-json", type=Path, default=DEFAULT_REPORT_JSON)
    parser.add_argument("--roll-bracket-sessions", type=int, default=3)
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    paths = ReconstructionPaths(
        source_csv=args.source_csv,
        symbology_payload_json=args.symbology_payload,
        definition_csv=args.definition_csv,
        schedule_json=args.schedule_json,
        tagged_parquet=args.tagged_parquet,
        tagged_csv=args.tagged_csv,
        report_json=args.report_json,
    )
    result = reconstruct_boundaries(
        paths=paths,
        roll_bracket_sessions=int(args.roll_bracket_sessions),
    )

    print(f"Schedule: {result['schedule_path']}")
    print(f"Tagged dataset: {result['tagged_path']} ({result['tagged_format']})")
    print(f"Report: {result['report_path']}")
    print(f"All checks passed: {result['report']['all_checks_passed']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
