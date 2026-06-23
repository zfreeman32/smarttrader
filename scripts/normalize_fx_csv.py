from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path
from typing import Iterable


STANDARD_FIELDS = ("Date", "Time", "Open", "High", "Low", "Close", "Volume")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Normalize raw FX OHLCV CSV data into the standard "
            "Date,Time,Open,High,Low,Close,Volume format used by this repo."
        )
    )
    parser.add_argument("input", type=Path, help="Raw input CSV path.")
    parser.add_argument("output", type=Path, help="Normalized output CSV path.")
    parser.add_argument(
        "--delimiter",
        default="auto",
        choices=("auto", ",", ";"),
        help="Override the input delimiter. Defaults to automatic detection.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=0,
        help="Optional row cap for smoke tests. Use 0 for the full file.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing an existing output file.",
    )
    return parser.parse_args()


def sniff_delimiter(first_line: str) -> str:
    semicolons = first_line.count(";")
    commas = first_line.count(",")
    if semicolons > commas:
        return ";"
    return ","


def looks_like_header(fields: list[str]) -> bool:
    if len(fields) < len(STANDARD_FIELDS):
        return False
    normalized = [field.strip().lower() for field in fields[: len(STANDARD_FIELDS)]]
    return normalized == [field.lower() for field in STANDARD_FIELDS]


def normalize_date(value: str) -> str:
    raw = value.strip()
    if len(raw) == 8 and raw.isdigit():
        return raw

    for fmt in ("%d/%m/%Y", "%Y-%m-%d", "%Y/%m/%d"):
        try:
            return datetime.strptime(raw, fmt).strftime("%Y%m%d")
        except ValueError:
            continue

    raise ValueError(f"Unsupported date format: {value!r}")


def normalize_time(value: str) -> str:
    raw = value.strip()
    for fmt in ("%H:%M:%S", "%H:%M"):
        try:
            parsed = datetime.strptime(raw, fmt)
            return parsed.strftime("%H:%M:%S")
        except ValueError:
            continue

    raise ValueError(f"Unsupported time format: {value!r}")


def standardize_row(fields: list[str], row_number: int) -> list[str]:
    if len(fields) < len(STANDARD_FIELDS):
        raise ValueError(f"Row {row_number} has {len(fields)} fields, expected at least 7.")

    date_value, time_value, open_value, high_value, low_value, close_value, volume_value = (
        field.strip() for field in fields[: len(STANDARD_FIELDS)]
    )

    return [
        normalize_date(date_value),
        normalize_time(time_value),
        open_value,
        high_value,
        low_value,
        close_value,
        volume_value,
    ]


def iter_rows(reader: Iterable[list[str]]) -> Iterable[tuple[int, list[str]]]:
    for row_number, row in enumerate(reader, start=1):
        if not row:
            continue
        if all(not cell.strip() for cell in row):
            continue
        yield row_number, row


def normalize_file(
    input_path: Path,
    output_path: Path,
    *,
    delimiter: str = "auto",
    max_rows: int = 0,
) -> tuple[int, str, bool]:
    input_path = input_path.resolve()
    output_path = output_path.resolve()

    with input_path.open("r", encoding="utf-8-sig", newline="") as src:
        first_line = src.readline()
        if not first_line:
            raise ValueError(f"Input file is empty: {input_path}")

        detected_delimiter = sniff_delimiter(first_line) if delimiter == "auto" else delimiter
        src.seek(0)
        reader = csv.reader(src, delimiter=detected_delimiter)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8", newline="") as dst:
            writer = csv.writer(dst)
            writer.writerow(STANDARD_FIELDS)

            row_iter = iter_rows(reader)
            first_row_number, first_row = next(row_iter)
            header_present = looks_like_header(first_row)

            rows_written = 0
            if not header_present:
                writer.writerow(standardize_row(first_row, first_row_number))
                rows_written += 1

            for row_number, row in row_iter:
                writer.writerow(standardize_row(row, row_number))
                rows_written += 1
                if max_rows > 0 and rows_written >= max_rows:
                    break

    return rows_written, detected_delimiter, header_present


def main() -> int:
    args = parse_args()

    input_path = args.input.resolve()
    output_path = args.output.resolve()

    if not input_path.exists():
        raise FileNotFoundError(f"Input file does not exist: {input_path}")

    if input_path == output_path:
        raise ValueError("Input and output paths must differ for normalization.")

    if output_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"Output file already exists: {output_path}. Use --overwrite to replace it."
        )

    rows_written, detected_delimiter, header_present = normalize_file(
        input_path,
        output_path,
        delimiter=args.delimiter,
        max_rows=max(0, args.max_rows),
    )

    print(f"Normalized rows: {rows_written:,}")
    print(f"Input delimiter: {detected_delimiter!r}")
    print(f"Input header:    {'present' if header_present else 'absent'}")
    print(f"Output file:     {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
