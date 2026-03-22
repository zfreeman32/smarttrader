from __future__ import annotations

import argparse
import ast
import csv
import json
from collections import Counter
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, List, Sequence

from .strategy_registry import STRATEGY_REGISTRY, normalize_strategy_name


def _all_strategies_path() -> Path:
    return Path(__file__).resolve().parent / "all_strategies.py"


@dataclass(frozen=True)
class FunctionSnapshot:
    label: str
    name: str
    occurrence: int
    lineno: int
    module_path: str
    file_stem: str
    strategy_id: str
    normalized_name: str
    normalized_file_stem: str
    normalized_body: str


def collect_monolith_functions(path: Path | None = None) -> List[FunctionSnapshot]:
    path = path or _all_strategies_path()
    tree = ast.parse(path.read_text(encoding="utf-8", errors="ignore"))
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    name_counts = Counter(node.name for node in functions)
    seen_counts: Counter[str] = Counter()

    snapshots: List[FunctionSnapshot] = []
    for node in functions:
        seen_counts[node.name] += 1
        occurrence = seen_counts[node.name]
        label = node.name if name_counts[node.name] == 1 else f"{node.name}#{occurrence}"
        snapshots.append(
            FunctionSnapshot(
                label=label,
                name=node.name,
                occurrence=occurrence,
                lineno=node.lineno,
                module_path=str(path),
                file_stem=path.stem,
                strategy_id=label,
                normalized_name=normalize_strategy_name(node.name),
                normalized_file_stem=normalize_strategy_name(path.stem),
                normalized_body=_normalize_function_body(node),
            )
        )
    return snapshots


def collect_standalone_functions() -> List[FunctionSnapshot]:
    snapshots: List[FunctionSnapshot] = []
    for spec in STRATEGY_REGISTRY.describe():
        tree = ast.parse(spec.module_path.read_text(encoding="utf-8", errors="ignore"))
        functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == spec.function_name]
        if not functions:
            continue
        node = functions[0]
        snapshots.append(
            FunctionSnapshot(
                label=spec.strategy_id,
                name=spec.function_name,
                occurrence=1,
                lineno=node.lineno,
                module_path=str(spec.module_path),
                file_stem=spec.file_stem,
                strategy_id=spec.strategy_id,
                normalized_name=normalize_strategy_name(spec.function_name),
                normalized_file_stem=normalize_strategy_name(spec.file_stem),
                normalized_body=_normalize_function_body(node),
            )
        )
    return snapshots


def build_similarity_report() -> Dict[str, object]:
    monolith_functions = collect_monolith_functions()
    standalone_functions = collect_standalone_functions()

    matches = [
        _best_match(monolith=monolith, standalone_candidates=standalone_functions)
        for monolith in monolith_functions
    ]

    duplicate_names = Counter(item.name for item in monolith_functions)
    duplicate_names = {
        name: count
        for name, count in sorted(duplicate_names.items())
        if count > 1
    }

    summary = {
        "monolith_definitions": len(monolith_functions),
        "monolith_unique_names": len({item.name for item in monolith_functions}),
        "standalone_entries": len(standalone_functions),
        "exact_body_matches": sum(int(item["exact_body_match"]) for item in matches),
        "exact_function_name_matches": sum(int(item["exact_function_name_match"]) for item in matches),
        "exact_file_name_matches": sum(int(item["exact_file_name_match"]) for item in matches),
        "duplicate_monolith_names": duplicate_names,
        "confidence_counts": Counter(item["confidence"] for item in matches),
    }

    return {"summary": summary, "matches": matches}


def _best_match(
    *,
    monolith: FunctionSnapshot,
    standalone_candidates: Sequence[FunctionSnapshot],
) -> Dict[str, object]:
    best_record: Dict[str, object] | None = None
    best_sort_key: tuple[float, ...] | None = None

    for candidate in standalone_candidates:
        exact_body = monolith.normalized_body == candidate.normalized_body
        exact_function_name = monolith.normalized_name == candidate.normalized_name
        exact_file_name = monolith.normalized_name == candidate.normalized_file_stem
        body_similarity = _similarity(monolith.normalized_body, candidate.normalized_body)
        function_name_similarity = _similarity(monolith.normalized_name, candidate.normalized_name)
        file_name_similarity = _similarity(monolith.normalized_name, candidate.normalized_file_stem)
        name_similarity = max(function_name_similarity, file_name_similarity)

        sort_key = (
            float(exact_body),
            float(exact_function_name),
            float(exact_file_name),
            body_similarity,
            name_similarity,
        )
        if best_sort_key is not None and sort_key <= best_sort_key:
            continue

        best_sort_key = sort_key
        best_record = {
            "monolith_label": monolith.label,
            "monolith_name": monolith.name,
            "monolith_occurrence": monolith.occurrence,
            "monolith_line": monolith.lineno,
            "standalone_strategy_id": candidate.strategy_id,
            "standalone_function": candidate.name,
            "standalone_file": Path(candidate.module_path).name,
            "standalone_line": candidate.lineno,
            "exact_body_match": exact_body,
            "exact_function_name_match": exact_function_name,
            "exact_file_name_match": exact_file_name,
            "body_similarity": round(body_similarity, 6),
            "function_name_similarity": round(function_name_similarity, 6),
            "file_name_similarity": round(file_name_similarity, 6),
            "confidence": _confidence_label(
                exact_body=exact_body,
                body_similarity=body_similarity,
                name_similarity=name_similarity,
            ),
        }

    if best_record is None:
        return {
            "monolith_label": monolith.label,
            "monolith_name": monolith.name,
            "monolith_occurrence": monolith.occurrence,
            "monolith_line": monolith.lineno,
            "standalone_strategy_id": None,
            "standalone_function": None,
            "standalone_file": None,
            "standalone_line": None,
            "exact_body_match": False,
            "exact_function_name_match": False,
            "exact_file_name_match": False,
            "body_similarity": 0.0,
            "function_name_similarity": 0.0,
            "file_name_similarity": 0.0,
            "confidence": "none",
        }

    return best_record


def _confidence_label(*, exact_body: bool, body_similarity: float, name_similarity: float) -> str:
    if exact_body:
        return "exact"
    if body_similarity >= 0.8 and name_similarity >= 0.9:
        return "strong"
    if body_similarity >= 0.65 or name_similarity >= 0.9:
        return "possible"
    return "weak"


def _normalize_function_body(node: ast.FunctionDef) -> str:
    normalized = ast.FunctionDef(
        name="strategy",
        args=node.args,
        body=node.body,
        decorator_list=[],
        returns=node.returns,
        type_comment=node.type_comment,
        type_params=getattr(node, "type_params", []),
    )
    return ast.dump(normalized, annotate_fields=False, include_attributes=False)


def _similarity(left: str, right: str) -> float:
    return SequenceMatcher(None, left, right).ratio()


def _save_report(report: Dict[str, object], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix.lower() == ".json":
        payload = dict(report)
        payload["summary"] = dict(payload["summary"])
        payload["summary"]["confidence_counts"] = dict(payload["summary"]["confidence_counts"])
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return

    fieldnames = list(report["matches"][0].keys()) if report["matches"] else []
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(report["matches"])


def _print_summary(report: Dict[str, object], limit: int) -> None:
    summary = report["summary"]
    print("Strategy similarity report")
    print(f"  Monolith definitions:          {summary['monolith_definitions']}")
    print(f"  Monolith unique names:         {summary['monolith_unique_names']}")
    print(f"  Standalone strategy entries:   {summary['standalone_entries']}")
    print(f"  Exact body matches:            {summary['exact_body_matches']}")
    print(f"  Exact function-name matches:   {summary['exact_function_name_matches']}")
    print(f"  Exact file-name matches:       {summary['exact_file_name_matches']}")

    duplicate_names = summary["duplicate_monolith_names"]
    if duplicate_names:
        print("  Duplicate monolith definitions:")
        for name, count in duplicate_names.items():
            print(f"    - {name}: {count}")

    confidence_counts = dict(summary["confidence_counts"])
    if confidence_counts:
        print("  Match confidence:")
        for label, count in sorted(confidence_counts.items()):
            print(f"    - {label}: {count}")

    weak_or_non_exact = [
        item for item in report["matches"]
        if item["confidence"] != "exact"
    ]
    if weak_or_non_exact:
        print(f"\nTop {min(limit, len(weak_or_non_exact))} non-exact matches:")
        ordered = sorted(
            weak_or_non_exact,
            key=lambda item: (
                item["confidence"],
                item["body_similarity"],
                item["function_name_similarity"],
                item["file_name_similarity"],
            ),
        )
        for item in ordered[:limit]:
            print(
                "  - "
                f"{item['monolith_label']} -> {item['standalone_strategy_id']} "
                f"[confidence={item['confidence']}, body={item['body_similarity']:.3f}, "
                f"fn={item['function_name_similarity']:.3f}, file={item['file_name_similarity']:.3f}]"
            )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare all_strategies.py to standalone strategy scripts.")
    parser.add_argument(
        "--output",
        default=None,
        help="Optional output path (.json or .csv) for the full match report.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="How many non-exact matches to print in the summary.",
    )
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    report = build_similarity_report()
    _print_summary(report, limit=max(args.limit, 0))

    if args.output:
        output_path = Path(args.output)
        _save_report(report, output_path)
        print(f"\nSaved report: {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
