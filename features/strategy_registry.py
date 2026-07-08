from __future__ import annotations

import ast
import contextlib
import hashlib
import importlib.util
import re
import sys
import warnings
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from types import ModuleType
from typing import Callable, Dict, List, Tuple

import pandas as pd

from .config import FeatureBuilderConfig
from .fx_calendar import normalize_datetime_series


StrategyBuilder = Callable[[pd.DataFrame], pd.DataFrame | pd.Series]


def _default_strategies_dir() -> Path:
    return Path(__file__).resolve().parent / "strategies"


class _LegacyCompatibleSeries(pd.Series):
    _metadata = ["_parent_frame", "_parent_column"]

    @property
    def _constructor(self):  # type: ignore[override]
        return _LegacyCompatibleSeries

    def __setitem__(self, key, value) -> None:
        super().__setitem__(key, value)
        parent_frame = getattr(self, "_parent_frame", None)
        parent_column = getattr(self, "_parent_column", None)
        if parent_frame is None or parent_column is None:
            return
        parent_frame[parent_column] = pd.Series(self.to_numpy(copy=True), index=parent_frame.index, name=parent_column)


class _LegacyCompatibleDataFrame(pd.DataFrame):
    @property
    def _constructor(self):  # type: ignore[override]
        return _LegacyCompatibleDataFrame

    @property
    def _constructor_sliced(self):  # type: ignore[override]
        return _LegacyCompatibleSeries

    def __getitem__(self, key):
        result = super().__getitem__(key)
        if isinstance(key, str) and isinstance(result, pd.Series):
            series = _LegacyCompatibleSeries(result)
            series._parent_frame = self
            series._parent_column = key
            return series
        return result


@contextlib.contextmanager
def _legacy_dataframe_context():
    original_dataframe = pd.DataFrame
    try:
        pd.DataFrame = _LegacyCompatibleDataFrame  # type: ignore[assignment]
        yield
    finally:
        pd.DataFrame = original_dataframe  # type: ignore[assignment]


def normalize_strategy_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def slugify_strategy_name(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return slug or "strategy"


def _run_strategy_with_legacy_pandas_compat(
    builder: StrategyBuilder,
    working: pd.DataFrame,
) -> pd.DataFrame | pd.Series:
    chained_assignment_warning = getattr(pd.errors, "ChainedAssignmentError", None)
    setting_with_copy_warning = getattr(pd.errors, "SettingWithCopyWarning", None)
    pandas_major_version_match = re.match(r"^(\d+)", getattr(pd, "__version__", "0"))
    pandas_major_version = int(pandas_major_version_match.group(1)) if pandas_major_version_match else 0
    copy_on_write_context = (
        contextlib.nullcontext()
        if pandas_major_version >= 3
        else pd.option_context("mode.copy_on_write", False)
    )

    with _legacy_dataframe_context():
        with copy_on_write_context:
            with warnings.catch_warnings():
                if chained_assignment_warning is not None:
                    warnings.simplefilter("ignore", chained_assignment_warning)
                if setting_with_copy_warning is not None:
                    warnings.simplefilter("ignore", setting_with_copy_warning)
                warnings.filterwarnings(
                    "ignore",
                    message=r".*ChainedAssignmentError.*",
                    category=FutureWarning,
                )
                return builder(working)


@dataclass(frozen=True)
class StrategySpec:
    strategy_id: str
    file_stem: str
    function_name: str
    module_path: Path
    function_count_in_module: int
    aliases: Tuple[str, ...]


class StrategyRegistry:
    """Lazy registry for standalone strategy scripts under features/strategies."""

    def __init__(self, strategies_dir: Path | None = None) -> None:
        self._strategies_dir = strategies_dir or _default_strategies_dir()
        self._specs = self._discover_specs()
        self._aliases = self._build_alias_index()

    def names(self) -> List[str]:
        return sorted(self._specs)

    def describe(self) -> List[StrategySpec]:
        return [self._specs[name] for name in self.names()]

    def get(self, name: str) -> StrategySpec:
        strategy_id = self.resolve(name)
        return self._specs[strategy_id]

    def resolve(self, name: str) -> str:
        if name in self._specs:
            return name

        key = normalize_strategy_name(name)
        matches = self._aliases.get(key, ())
        if not matches:
            available = ", ".join(self.names()[:20])
            suffix = "..." if len(self._specs) > 20 else ""
            raise KeyError(f"Unknown strategy '{name}'. Available: {available}{suffix}")
        if len(matches) > 1:
            options = ", ".join(matches)
            raise KeyError(f"Ambiguous strategy alias '{name}'. Use one of: {options}")
        return matches[0]

    def load(self, name: str) -> StrategyBuilder:
        spec = self.get(name)
        try:
            module = _load_strategy_module(spec.module_path)
        except Exception as exc:
            raise ImportError(
                f"Could not import strategy '{spec.strategy_id}' from '{spec.module_path.name}': {exc}"
            ) from exc
        builder = getattr(module, spec.function_name, None)
        if builder is None or not callable(builder):
            raise TypeError(
                f"Strategy '{spec.strategy_id}' does not expose a callable '{spec.function_name}'."
            )
        return builder

    def build(
        self,
        name: str,
        df: pd.DataFrame,
        *,
        copy_input: bool = True,
    ) -> pd.DataFrame:
        builder = self.load(name)
        working = df.copy(deep=True) if copy_input else df
        built = _run_strategy_with_legacy_pandas_compat(builder, working)
        if isinstance(built, pd.Series):
            built = built.to_frame()
        if not isinstance(built, pd.DataFrame):
            raise TypeError(f"Strategy '{name}' must return a pandas DataFrame or Series.")
        return built

    def _discover_specs(self) -> Dict[str, StrategySpec]:
        specs: Dict[str, StrategySpec] = {}

        for module_path in sorted(self._strategies_dir.glob("*.py")):
            try:
                tree = ast.parse(module_path.read_text(encoding="utf-8", errors="ignore"))
            except SyntaxError:
                continue

            functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
            if not functions:
                continue

            file_stem = module_path.stem
            function_count = len(functions)
            for function in functions:
                strategy_id = (
                    file_stem
                    if function_count == 1
                    else f"{file_stem}.{function.name}"
                )
                aliases = tuple(
                    dict.fromkeys(
                        [
                            strategy_id,
                            file_stem,
                            function.name,
                            slugify_strategy_name(file_stem),
                            slugify_strategy_name(function.name),
                        ]
                    )
                )
                specs[strategy_id] = StrategySpec(
                    strategy_id=strategy_id,
                    file_stem=file_stem,
                    function_name=function.name,
                    module_path=module_path,
                    function_count_in_module=function_count,
                    aliases=aliases,
                )

        return specs

    def _build_alias_index(self) -> Dict[str, Tuple[str, ...]]:
        alias_map: Dict[str, List[str]] = {}
        for spec in self._specs.values():
            for alias in spec.aliases:
                key = normalize_strategy_name(alias)
                alias_map.setdefault(key, [])
                if spec.strategy_id not in alias_map[key]:
                    alias_map[key].append(spec.strategy_id)
        return {key: tuple(sorted(values)) for key, values in alias_map.items()}


@lru_cache(maxsize=None)
def _load_strategy_module(module_path: Path) -> ModuleType:
    module_name = _module_name_for_path(module_path)
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from '{module_path}'.")

    module = importlib.util.module_from_spec(spec)
    parent = str(module_path.parent)
    added_to_path = False
    if parent not in sys.path:
        sys.path.insert(0, parent)
        added_to_path = True

    try:
        spec.loader.exec_module(module)
    finally:
        if added_to_path:
            try:
                sys.path.remove(parent)
            except ValueError:
                pass

    return module


def _module_name_for_path(module_path: Path) -> str:
    digest = hashlib.sha1(str(module_path).encode("utf-8")).hexdigest()[:12]
    return f"features_strategy_{slugify_strategy_name(module_path.stem)}_{digest}"


def prepare_strategy_input(
    df: pd.DataFrame,
    config: FeatureBuilderConfig | None = None,
) -> pd.DataFrame:
    """Adapt the normalized feature-builder frame to legacy strategy expectations."""
    # Avoid an eager deep copy here; each strategy gets its own defensive copy
    # right before execution.
    prepared = df.copy(deep=False)
    dt_index: pd.Series | None = None

    if "datetime" in prepared.columns:
        dt_index = normalize_datetime_series(
            prepared["datetime"],
            source_timezone=config.source_timezone if config is not None else "UTC",
            canonical_timezone=config.canonical_timezone if config is not None else "UTC",
        )
        if config is not None:
            dt_index = dt_index.dt.tz_convert(config.feature_clock_timezone)
        if dt_index.notna().any():
            prepared.index = pd.DatetimeIndex(dt_index)

    additions: dict[str, pd.Series] = {}
    alias_map = {
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "volume": "Volume",
    }
    for source, alias in alias_map.items():
        if source in prepared.columns and alias not in prepared.columns:
            additions[alias] = pd.Series(prepared[source].to_numpy(), index=prepared.index, name=alias)

    if dt_index is not None and dt_index.notna().any():
        display_index = dt_index.dt.tz_localize(None)
        if "Date" not in prepared.columns:
            additions["Date"] = pd.Series(display_index.to_numpy(), index=prepared.index, name="Date")
        if "Time" not in prepared.columns:
            additions["Time"] = pd.Series(
                display_index.dt.strftime("%H:%M:%S").to_numpy(),
                index=prepared.index,
                name="Time",
            )

    if additions:
        prepared = pd.concat([prepared, pd.DataFrame(additions)], axis=1)

    return prepared


STRATEGY_REGISTRY = StrategyRegistry()
