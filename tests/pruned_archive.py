"""Data deliberately pruned on 2026-10-11 (docs/audits/data_prune_20261011.csv).

Evidence records keep their hashes for these paths, but the bytes no longer exist. Archive checks skip exactly
these paths and keep verifying everything else, so any other missing file still fails.
"""
import csv
import json
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PRUNE_LIST = ROOT / "docs/audits/data_prune_20261011.csv"
B1_REGISTRY_INVENTORY = ROOT / "docs/audits/b1_registry_inventory_complete_20260918.json"


@lru_cache(maxsize=1)
def _pruned() -> tuple[frozenset, tuple]:
    with PRUNE_LIST.open(encoding="utf-8", newline="") as handle:
        paths = [row["path"] for row in csv.DictReader(handle)]
    return frozenset(paths), tuple(path + "/" for path in paths)


def _relative(path) -> str:
    text = str(path).replace("\\", "/")
    root = ROOT.as_posix()
    return text[len(root) + 1:] if text.startswith(root + "/") else text


def is_pruned(path) -> bool:
    """True when ``path`` was removed by the 2026-10-11 prune and is really absent."""
    rel = _relative(path)
    exact, dirs = _pruned()
    return (rel in exact or rel.startswith(dirs)) and not (ROOT / rel).exists()


def pruned_registrations() -> int:
    """Registrations the frozen B1 inventory attributes to registry files removed by the prune."""
    rows = json.loads(B1_REGISTRY_INVENTORY.read_text(encoding="utf-8"))["rows"]
    return sum(1 for row in rows if is_pruned(row["registry"]))
