from __future__ import annotations

import os
from pathlib import Path
from typing import TypeVar

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ENV_PATH = REPO_ROOT / ".env"
_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})
_FALSE_VALUES = frozenset({"0", "false", "no", "off"})
_LOADED_ENV_KEYS: set[tuple[str, bool]] = set()

_T = TypeVar("_T")


def load_repo_env(
    env_path: str | Path | None = None,
    *,
    override: bool = False,
) -> Path | None:
    resolved = resolve_repo_path(env_path or DEFAULT_ENV_PATH)
    if not resolved.exists():
        return None

    cache_key = (str(resolved), bool(override))
    if cache_key in _LOADED_ENV_KEYS:
        return resolved

    try:
        from dotenv import load_dotenv
    except ImportError:
        return None

    load_dotenv(resolved, override=override)
    _LOADED_ENV_KEYS.add(cache_key)
    return resolved


def resolve_repo_path(path_value: str | Path | None, *, default: Path | None = None) -> Path | None:
    if path_value is None:
        return default
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def env_str(name: str, default: _T | None = None) -> str | _T | None:
    value = os.getenv(name)
    if value is None:
        return default
    stripped = value.strip()
    return stripped if stripped else default


def env_int(name: str, default: _T | None = None) -> int | _T | None:
    value = env_str(name)
    if value is None:
        return default
    return int(value)


def env_float(name: str, default: _T | None = None) -> float | _T | None:
    value = env_str(name)
    if value is None:
        return default
    return float(value)


def env_bool(name: str, default: bool = False) -> bool:
    value = env_str(name)
    if value is None:
        return bool(default)
    lowered = value.lower()
    if lowered in _TRUE_VALUES:
        return True
    if lowered in _FALSE_VALUES:
        return False
    raise ValueError(f"Environment variable {name} must be one of {_TRUE_VALUES | _FALSE_VALUES}, got {value!r}.")


def env_path(name: str, default: Path | None = None) -> Path | None:
    value = env_str(name)
    if value is None:
        return default
    return resolve_repo_path(value)
