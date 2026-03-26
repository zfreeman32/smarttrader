from __future__ import annotations

import pickle
import shutil
import subprocess
import sys
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype

from ..config import FeatureBuilderConfig
from ..progress import report_progress
from ..registry import register_feature_set
from ..strategy_registry import (
    STRATEGY_REGISTRY,
    StrategyRegistry,
    prepare_strategy_input,
    slugify_strategy_name,
)


@register_feature_set(
    name="strategy_signals",
    category="strategy",
    description="Standalone strategy-module outputs encoded as dataset features",
    required_columns=("open", "high", "low", "close"),
)
def build_strategy_signals(
    df: pd.DataFrame,
    config: FeatureBuilderConfig,
) -> pd.DataFrame:
    if not config.strategy_ids:
        return pd.DataFrame(index=df.index)

    started = perf_counter()
    strategy_input = prepare_strategy_input(df)
    input_columns = set(strategy_input.columns)
    requested_strategy_ids = list(dict.fromkeys(config.strategy_ids))
    worker_count = min(max(1, config.transform_workers), len(requested_strategy_ids))
    report_progress(
        stage="strategy_batch",
        action="start",
        name="strategy_signals",
        requested=len(requested_strategy_ids),
        workers=worker_count,
        input_rows=int(len(strategy_input)),
        input_columns=int(len(strategy_input.columns)),
        timeout_seconds=config.strategy_timeout_seconds,
    )
    strategy_results = _execute_strategy_jobs(
        requested_strategy_ids=requested_strategy_ids,
        strategy_input=strategy_input,
        target_index=df.index,
        input_columns=input_columns,
        skip_failed_strategies=config.skip_failed_strategies,
        max_workers=worker_count,
        timeout_seconds=config.strategy_timeout_seconds,
        strategies_dir=STRATEGY_REGISTRY._strategies_dir,
    )
    built_frames = []
    built_strategy_ids: list[str] = []
    skipped_strategies: list[dict[str, str]] = []
    slowest_attempts = sorted(
        (
            {
                "requested_strategy": result.requested_name,
                "resolved_strategy": result.resolved_name or "",
                "status": "skipped" if result.failure is not None else "built",
                "elapsed_seconds": round(result.elapsed_seconds, 6),
                "output_columns": result.output_columns,
            }
            for result in strategy_results
        ),
        key=lambda item: item["elapsed_seconds"],
        reverse=True,
    )[:10]

    for resolved_name, normalized_output, failure in strategy_results:
        if failure is not None:
            skipped_strategies.append(failure)
            continue
        if resolved_name is None:
            continue
        built_strategy_ids.append(resolved_name)
        if not normalized_output.empty:
            built_frames.append(normalized_output)

    if built_frames:
        combined = pd.concat(built_frames, axis=1)
        result = combined.loc[:, ~combined.columns.duplicated()]
    else:
        result = pd.DataFrame(index=df.index)

    wall_clock_seconds = perf_counter() - started
    result.attrs["feature_build_report"] = {
        "requested": len(requested_strategy_ids),
        "built": len(built_strategy_ids),
        "skipped": len(skipped_strategies),
        "output_columns": int(result.shape[1]),
        "built_strategy_ids": built_strategy_ids,
        "skipped_strategies": skipped_strategies,
        "wall_clock_seconds": round(wall_clock_seconds, 6),
        "slowest_strategy_attempts": slowest_attempts,
    }
    report_progress(
        stage="strategy_batch",
        action="complete",
        name="strategy_signals",
        elapsed_seconds=wall_clock_seconds,
        requested=len(requested_strategy_ids),
        built=len(built_strategy_ids),
        skipped=len(skipped_strategies),
        output_columns=int(result.shape[1]),
    )
    return result


@dataclass(frozen=True)
class StrategyJobResult:
    requested_name: str
    resolved_name: str | None
    normalized_output: pd.DataFrame
    failure: dict[str, str] | None
    elapsed_seconds: float
    output_columns: int

    def __iter__(self):
        yield self.resolved_name
        yield self.normalized_output
        yield self.failure


def _execute_strategy_jobs(
    *,
    requested_strategy_ids: list[str],
    strategy_input: pd.DataFrame,
    target_index: pd.Index,
    input_columns: set[str],
    skip_failed_strategies: bool,
    max_workers: int,
    timeout_seconds: float | None,
    strategies_dir: Path,
) -> list[StrategyJobResult]:
    normalized_timeout = _normalize_timeout_seconds(timeout_seconds)
    if normalized_timeout is not None:
        return _execute_strategy_jobs_with_timeout(
            requested_strategy_ids=requested_strategy_ids,
            strategy_input=strategy_input,
            target_index=target_index,
            input_columns=input_columns,
            skip_failed_strategies=skip_failed_strategies,
            max_workers=max_workers,
            timeout_seconds=normalized_timeout,
            strategies_dir=strategies_dir,
        )

    total_jobs = len(requested_strategy_ids)
    if max_workers <= 1 or len(requested_strategy_ids) <= 1:
        results: list[StrategyJobResult] = []
        built_count = 0
        skipped_count = 0
        for index, strategy_name in enumerate(requested_strategy_ids, start=1):
            result = _run_strategy_job(
                requested_name=strategy_name,
                strategy_input=strategy_input,
                target_index=target_index,
                input_columns=input_columns,
                skip_failed_strategies=skip_failed_strategies,
                index=index,
                total=total_jobs,
            )
            _raise_strategy_failure_if_needed(result, skip_failed_strategies)
            if result.failure is None:
                built_count += 1
            else:
                skipped_count += 1
            _report_strategy_result(
                result=result,
                completed=index,
                total=total_jobs,
                built_count=built_count,
                skipped_count=skipped_count,
            )
            results.append(result)
        return results

    worker_count = min(max(1, max_workers), total_jobs)
    ordered_results: list[StrategyJobResult | None] = [None] * total_jobs
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = {
            executor.submit(
                _run_strategy_job,
                requested_name=strategy_name,
                strategy_input=strategy_input,
                target_index=target_index,
                input_columns=input_columns,
                skip_failed_strategies=skip_failed_strategies,
                index=idx + 1,
                total=total_jobs,
            ): idx
            for idx, strategy_name in enumerate(requested_strategy_ids)
        }
        completed = 0
        built_count = 0
        skipped_count = 0
        for future in as_completed(futures):
            idx = futures[future]
            result = future.result()
            _raise_strategy_failure_if_needed(result, skip_failed_strategies)
            completed += 1
            if result.failure is None:
                built_count += 1
            else:
                skipped_count += 1
            _report_strategy_result(
                result=result,
                completed=completed,
                total=total_jobs,
                built_count=built_count,
                skipped_count=skipped_count,
            )
            ordered_results[idx] = result

    return [result for result in ordered_results if result is not None]


def _run_strategy_job(
    *,
    requested_name: str,
    strategy_input: pd.DataFrame,
    target_index: pd.Index,
    input_columns: set[str],
    skip_failed_strategies: bool,
    index: int,
    total: int,
) -> StrategyJobResult:
    started = perf_counter()
    report_progress(
        stage="strategy",
        action="start",
        name=requested_name,
        index=index,
        total=total,
    )
    result = _run_strategy_job_core(
        requested_name=requested_name,
        strategy_input=strategy_input,
        target_index=target_index,
        input_columns=input_columns,
        registry=STRATEGY_REGISTRY,
        started=started,
    )
    _raise_strategy_failure_if_needed(result, skip_failed_strategies)
    return result


def _run_strategy_job_core(
    *,
    requested_name: str,
    strategy_input: pd.DataFrame,
    target_index: pd.Index,
    input_columns: set[str],
    registry: StrategyRegistry,
    started: float | None = None,
) -> StrategyJobResult:
    resolved_name: str | None = None
    started = perf_counter() if started is None else started
    try:
        resolved_name = registry.resolve(requested_name)
        raw_output = registry.build(resolved_name, strategy_input, copy_input=True)
        normalized_output = _normalize_strategy_output(
            strategy_id=resolved_name,
            output=raw_output,
            source_index=strategy_input.index,
            target_index=target_index,
            input_columns=input_columns,
        )
    except Exception as exc:
        elapsed = perf_counter() - started
        return StrategyJobResult(
            requested_name=requested_name,
            resolved_name=resolved_name,
            normalized_output=pd.DataFrame(index=target_index),
            failure=_serialize_strategy_failure(
                requested_name=requested_name,
                resolved_name=resolved_name,
                exc=exc,
            ),
            elapsed_seconds=elapsed,
            output_columns=0,
        )

    elapsed = perf_counter() - started
    return StrategyJobResult(
        requested_name=requested_name,
        resolved_name=resolved_name,
        normalized_output=normalized_output,
        failure=None,
        elapsed_seconds=elapsed,
        output_columns=int(normalized_output.shape[1]),
    )


def _execute_strategy_jobs_with_timeout(
    *,
    requested_strategy_ids: list[str],
    strategy_input: pd.DataFrame,
    target_index: pd.Index,
    input_columns: set[str],
    skip_failed_strategies: bool,
    max_workers: int,
    timeout_seconds: float,
    strategies_dir: Path,
) -> list[StrategyJobResult]:
    total_jobs = len(requested_strategy_ids)
    worker_count = min(max(1, max_workers), total_jobs)
    temp_dir = Path(_strategy_timeout_workspace()) / f"run_{uuid.uuid4().hex}"
    temp_dir.mkdir(parents=True, exist_ok=False)
    try:
        context_path = temp_dir / "strategy_context.pkl"
        _write_strategy_timeout_context(
            context_path=context_path,
            strategy_input=strategy_input,
            target_index=target_index,
            input_columns=input_columns,
        )

        if worker_count <= 1 or total_jobs <= 1:
            results: list[StrategyJobResult] = []
            built_count = 0
            skipped_count = 0
            for index, strategy_name in enumerate(requested_strategy_ids, start=1):
                result = _run_strategy_job_with_timeout(
                    requested_name=strategy_name,
                    context_path=context_path,
                    target_index=target_index,
                    skip_failed_strategies=skip_failed_strategies,
                    timeout_seconds=timeout_seconds,
                    strategies_dir=strategies_dir,
                    index=index,
                    total=total_jobs,
                    result_dir=temp_dir,
                )
                if result.failure is None:
                    built_count += 1
                else:
                    skipped_count += 1
                _report_strategy_result(
                    result=result,
                    completed=index,
                    total=total_jobs,
                    built_count=built_count,
                    skipped_count=skipped_count,
                )
                results.append(result)
            return results

        ordered_results: list[StrategyJobResult | None] = [None] * total_jobs
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = {
                executor.submit(
                    _run_strategy_job_with_timeout,
                    requested_name=strategy_name,
                    context_path=context_path,
                    target_index=target_index,
                    skip_failed_strategies=skip_failed_strategies,
                    timeout_seconds=timeout_seconds,
                    strategies_dir=strategies_dir,
                    index=idx + 1,
                    total=total_jobs,
                    result_dir=temp_dir,
                ): idx
                for idx, strategy_name in enumerate(requested_strategy_ids)
            }
            completed = 0
            built_count = 0
            skipped_count = 0
            for future in as_completed(futures):
                idx = futures[future]
                result = future.result()
                completed += 1
                if result.failure is None:
                    built_count += 1
                else:
                    skipped_count += 1
                _report_strategy_result(
                    result=result,
                    completed=completed,
                    total=total_jobs,
                    built_count=built_count,
                    skipped_count=skipped_count,
                )
                ordered_results[idx] = result
        return [result for result in ordered_results if result is not None]
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def _strategy_timeout_workspace() -> str:
    workspace = Path(__file__).resolve().parents[2] / "data" / ".strategy_timeout_cache"
    workspace.mkdir(parents=True, exist_ok=True)
    return str(workspace)


def _write_strategy_timeout_context(
    *,
    context_path: Path,
    strategy_input: pd.DataFrame,
    target_index: pd.Index,
    input_columns: set[str],
) -> None:
    with context_path.open("wb") as handle:
        pickle.dump(
            {
                "strategy_input": strategy_input,
                "target_index": target_index,
                "input_columns": tuple(sorted(input_columns)),
            },
            handle,
        )


def _run_strategy_job_with_timeout(
    *,
    requested_name: str,
    context_path: Path,
    target_index: pd.Index,
    skip_failed_strategies: bool,
    timeout_seconds: float,
    strategies_dir: Path,
    index: int,
    total: int,
    result_dir: Path,
) -> StrategyJobResult:
    report_progress(
        stage="strategy",
        action="start",
        name=requested_name,
        index=index,
        total=total,
    )

    result = _run_strategy_subprocess(
        requested_name=requested_name,
        context_path=context_path,
        target_index=target_index,
        timeout_seconds=timeout_seconds,
        strategies_dir=strategies_dir,
        result_dir=result_dir,
    )
    _raise_strategy_failure_if_needed(result, skip_failed_strategies)
    return result


def _run_strategy_subprocess(
    *,
    requested_name: str,
    context_path: Path,
    target_index: pd.Index,
    timeout_seconds: float,
    strategies_dir: Path,
    result_dir: Path,
) -> StrategyJobResult:
    started = perf_counter()
    output_path = result_dir / f"{slugify_strategy_name(requested_name)}_{abs(hash(requested_name))}.pkl"
    if output_path.exists():
        output_path.unlink()

    command = [
        sys.executable,
        "-m",
        "features.strategy_subprocess_runner",
        "--context",
        str(context_path),
        "--strategy-name",
        requested_name,
        "--output",
        str(output_path),
        "--strategies-dir",
        str(strategies_dir),
    ]

    try:
        completed = subprocess.run(
            command,
            cwd=Path(__file__).resolve().parents[2],
            timeout=timeout_seconds,
            capture_output=True,
            text=True,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return _build_strategy_timeout_result(
            requested_name=requested_name,
            target_index=target_index,
            elapsed_seconds=perf_counter() - started,
            timeout_seconds=timeout_seconds,
        )

    elapsed = perf_counter() - started
    if completed.returncode != 0:
        return _build_strategy_subprocess_error_result(
            requested_name=requested_name,
            target_index=target_index,
            elapsed_seconds=elapsed,
            return_code=completed.returncode,
            stderr=(completed.stderr or completed.stdout or "").strip(),
        )
    if not output_path.exists():
        return _build_strategy_subprocess_error_result(
            requested_name=requested_name,
            target_index=target_index,
            elapsed_seconds=elapsed,
            return_code=completed.returncode,
            stderr="Strategy subprocess finished without writing a result file.",
        )

    try:
        with output_path.open("rb") as handle:
            result = pickle.load(handle)
    finally:
        output_path.unlink(missing_ok=True)
    return result


def _build_strategy_timeout_result(
    *,
    requested_name: str,
    target_index: pd.Index,
    elapsed_seconds: float,
    timeout_seconds: float,
) -> StrategyJobResult:
    return StrategyJobResult(
        requested_name=requested_name,
        resolved_name=None,
        normalized_output=pd.DataFrame(index=target_index),
        failure={
            "requested_strategy": requested_name,
            "resolved_strategy": "",
            "error_type": "TimeoutError",
            "error_message": f"Timed out after {timeout_seconds:.1f} seconds",
        },
        elapsed_seconds=elapsed_seconds,
        output_columns=0,
    )


def _build_strategy_process_exit_result(
    *,
    requested_name: str,
    target_index: pd.Index,
    elapsed_seconds: float,
    exit_code: int | None,
) -> StrategyJobResult:
    exit_detail = "unknown" if exit_code is None else str(exit_code)
    return StrategyJobResult(
        requested_name=requested_name,
        resolved_name=None,
        normalized_output=pd.DataFrame(index=target_index),
        failure={
            "requested_strategy": requested_name,
            "resolved_strategy": "",
            "error_type": "ProcessExitError",
            "error_message": f"Worker process exited unexpectedly with code {exit_detail}",
        },
        elapsed_seconds=elapsed_seconds,
        output_columns=0,
    )


def _build_strategy_subprocess_error_result(
    *,
    requested_name: str,
    target_index: pd.Index,
    elapsed_seconds: float,
    return_code: int,
    stderr: str,
) -> StrategyJobResult:
    message = stderr or f"Subprocess exited with code {return_code}"
    message = " ".join(message.split())
    return StrategyJobResult(
        requested_name=requested_name,
        resolved_name=None,
        normalized_output=pd.DataFrame(index=target_index),
        failure={
            "requested_strategy": requested_name,
            "resolved_strategy": "",
            "error_type": "SubprocessError",
            "error_message": message[:500],
        },
        elapsed_seconds=elapsed_seconds,
        output_columns=0,
    )


def _normalize_timeout_seconds(timeout_seconds: float | None) -> float | None:
    if timeout_seconds is None:
        return None
    try:
        numeric = float(timeout_seconds)
    except (TypeError, ValueError):
        return None
    if numeric <= 0:
        return None
    return numeric


def _raise_strategy_failure_if_needed(
    result: StrategyJobResult,
    skip_failed_strategies: bool,
) -> None:
    if result.failure is None or skip_failed_strategies:
        return
    error_type = result.failure.get("error_type", "StrategyExecutionError")
    error_message = result.failure.get("error_message", "Strategy execution failed.")
    raise RuntimeError(f"{error_type}: {error_message}")


def _report_strategy_result(
    *,
    result: StrategyJobResult,
    completed: int,
    total: int,
    built_count: int,
    skipped_count: int,
) -> None:
    action = "skipped" if result.failure is not None else "complete"
    message = None
    if result.failure is not None:
        message = result.failure.get("error_message")
    report_progress(
        stage="strategy",
        action=action,
        name=result.resolved_name or result.requested_name,
        index=completed,
        total=total,
        elapsed_seconds=result.elapsed_seconds,
        message=message,
        built=built_count,
        skipped=skipped_count,
        output_columns=result.output_columns,
    )


def _normalize_strategy_output(
    *,
    strategy_id: str,
    output: pd.DataFrame,
    source_index: pd.Index,
    target_index: pd.Index,
    input_columns: set[str],
) -> pd.DataFrame:
    output = output.copy()
    output = output.loc[:, [column for column in output.columns if column not in input_columns]]
    if output.empty:
        return pd.DataFrame(index=target_index)

    if len(output.index) == len(source_index) and not output.index.equals(source_index):
        output.index = source_index
    else:
        output = output.reindex(source_index)

    output.index = target_index

    encoded_frames = []
    strategy_prefix = f"strategy__{slugify_strategy_name(strategy_id)}"

    for column in output.columns:
        series = output[column]
        column_slug = slugify_strategy_name(str(column))
        prefixed_name = f"{strategy_prefix}__{column_slug}"

        if is_bool_dtype(series):
            encoded_frames.append(series.fillna(False).astype(int).rename(prefixed_name).to_frame())
            continue

        if is_numeric_dtype(series):
            encoded_frames.append(pd.to_numeric(series, errors="coerce").rename(prefixed_name).to_frame())
            continue

        categorical = series.astype("string").fillna("missing")
        dummies = pd.get_dummies(
            categorical,
            prefix=prefixed_name,
            prefix_sep="__",
            dtype=int,
        )
        encoded_frames.append(dummies)

    if not encoded_frames:
        return pd.DataFrame(index=target_index)

    normalized = pd.concat(encoded_frames, axis=1)
    return normalized.loc[:, ~normalized.columns.duplicated()]


def _serialize_strategy_failure(
    *,
    requested_name: str,
    resolved_name: str | None,
    exc: Exception,
) -> dict[str, str]:
    message = " ".join(str(exc).split())
    return {
        "requested_strategy": requested_name,
        "resolved_strategy": resolved_name or "",
        "error_type": type(exc).__name__,
        "error_message": message[:500],
    }
