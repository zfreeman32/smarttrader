from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
import threading
from types import SimpleNamespace
import uuid

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ote_live.env import load_repo_env
from ote_live.scripts.run_live_collector import build_parser as build_live_collector_parser
from ote_live.scripts.run_live_stack import build_process_plan
from ote_live.storage import SQLiteLiveDataStore


def test_load_repo_env_reads_explicit_env_file(monkeypatch) -> None:
    tmp_root = ROOT / "tmp" / "ote_live_env_tests"
    tmp_root.mkdir(parents=True, exist_ok=True)
    env_path = tmp_root / f"{uuid.uuid4().hex}.env"
    env_path.write_text("FMP_API_KEY=test-key\nOTE_LIVE_ASSET=GBPUSD\n", encoding="utf-8")
    monkeypatch.delenv("FMP_API_KEY", raising=False)
    monkeypatch.delenv("OTE_LIVE_ASSET", raising=False)

    def _fake_load_dotenv(path, override=False):
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            if override or key not in os.environ:
                os.environ[key] = value
        return True

    monkeypatch.setitem(sys.modules, "dotenv", SimpleNamespace(load_dotenv=_fake_load_dotenv))

    loaded_path = load_repo_env(env_path)

    assert loaded_path == env_path
    assert os.environ["FMP_API_KEY"] == "test-key"
    assert os.environ["OTE_LIVE_ASSET"] == "GBPUSD"


def test_build_process_plan_includes_dashboard_url_and_runtime_commands() -> None:
    args = argparse.Namespace(
        python_exe="python",
        db_path="ote_live/runtime_data/live_market_data.sqlite3",
        asset="EURUSD",
        timeframe="5m",
        dashboard_host="127.0.0.1",
        dashboard_port=8050,
        dashboard_refresh_interval_ms=10000,
        dashboard_signal_limit=120,
        max_cycles=5,
        skip_policy_package=False,
        skip_manifest_export=False,
        collector_only=False,
        dashboard_only=False,
        open_browser=False,
        browser_wait_seconds=0.0,
        collector_arg=["--log-level", "DEBUG"],
        dashboard_arg=["--debug"],
        dry_run=True,
    )

    plan = build_process_plan(args)

    assert plan.dashboard_url == "http://127.0.0.1:8050"
    assert plan.package_command == ("python", "-m", "ote_live.scripts.package_candidate_policies")
    assert plan.export_command == ("python", "-m", "ote_live.scripts.export_live_runtime_manifests")
    assert plan.dashboard_command is not None
    assert "--debug" in plan.dashboard_command
    assert plan.collector_command is not None
    assert "--dashboard-url" in plan.collector_command
    assert "--max-cycles" in plan.collector_command
    assert "DEBUG" in plan.collector_command


def test_live_collector_parser_accepts_runtime_manifest_paths_from_env(monkeypatch) -> None:
    monkeypatch.setenv("OTE_LIVE_LONG_RUNTIME_MANIFEST_PATH", "ote_live/runtime_manifests/live_runtime_manifest_long.json")
    monkeypatch.setenv("OTE_LIVE_SHORT_RUNTIME_MANIFEST_PATH", "ote_live/runtime_manifests/live_runtime_manifest_short.json")
    monkeypatch.setenv("FMP_API_KEY", "test-key")

    parser = build_live_collector_parser()
    args = parser.parse_args([])

    assert Path(args.long_runtime_manifest_path).name == "live_runtime_manifest_long.json"
    assert Path(args.short_runtime_manifest_path).name == "live_runtime_manifest_short.json"


def test_sqlite_live_data_store_connection_can_be_read_from_another_thread() -> None:
    tmp_root = ROOT / "tmp" / "ote_live_env_tests"
    tmp_root.mkdir(parents=True, exist_ok=True)
    db_path = tmp_root / f"{uuid.uuid4().hex}.sqlite"
    result: dict[str, object] = {}

    with SQLiteLiveDataStore(db_path) as store:
        def _query() -> None:
            row = store.connection.execute("SELECT 1 AS value").fetchone()
            result["value"] = int(row["value"]) if row is not None else None

        thread = threading.Thread(target=_query)
        thread.start()
        thread.join(timeout=5)

    assert result["value"] == 1
