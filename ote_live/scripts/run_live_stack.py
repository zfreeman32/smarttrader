from __future__ import annotations

import argparse
import subprocess
import sys
import time
import webbrowser
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from ote_live.env import env_bool, env_int, env_path, env_str, load_repo_env
from ote_live.ingestion.runtime import DEFAULT_DB_PATH

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DASHBOARD_HOST = "127.0.0.1"
DEFAULT_DASHBOARD_PORT = 8050
DEFAULT_DASHBOARD_REFRESH_INTERVAL_MS = 10_000
DEFAULT_DASHBOARD_SIGNAL_LIMIT = 120


@dataclass(frozen=True)
class StackProcessPlan:
    package_command: tuple[str, ...] | None
    export_command: tuple[str, ...] | None
    dashboard_command: tuple[str, ...] | None
    collector_command: tuple[str, ...] | None
    dashboard_url: str


def build_parser() -> argparse.ArgumentParser:
    load_repo_env()
    parser = argparse.ArgumentParser(
        description=(
            "Package the current live runtime artifacts, then launch the live dashboard "
            "and collector together from one command."
        ),
    )
    parser.add_argument("--python-exe", default=sys.executable or "python")
    parser.add_argument("--db-path", default=str(env_path("OTE_LIVE_DB_PATH", DEFAULT_DB_PATH)))
    parser.add_argument("--asset", default=env_str("OTE_LIVE_ASSET", "EURUSD"))
    parser.add_argument("--timeframe", default=env_str("OTE_LIVE_DASHBOARD_TIMEFRAME", "5m"))
    parser.add_argument("--dashboard-host", default=env_str("OTE_LIVE_DASHBOARD_HOST", DEFAULT_DASHBOARD_HOST))
    parser.add_argument("--dashboard-port", type=int, default=env_int("OTE_LIVE_DASHBOARD_PORT", DEFAULT_DASHBOARD_PORT))
    parser.add_argument(
        "--dashboard-refresh-interval-ms",
        type=int,
        default=env_int("OTE_LIVE_DASHBOARD_REFRESH_INTERVAL_MS", DEFAULT_DASHBOARD_REFRESH_INTERVAL_MS),
    )
    parser.add_argument(
        "--dashboard-signal-limit",
        type=int,
        default=env_int("OTE_LIVE_DASHBOARD_SIGNAL_LIMIT", DEFAULT_DASHBOARD_SIGNAL_LIMIT),
    )
    parser.add_argument("--max-cycles", type=int, default=env_int("OTE_LIVE_MAX_CYCLES"))
    parser.add_argument("--skip-policy-package", action="store_true")
    parser.add_argument("--skip-manifest-export", action="store_true")
    parser.add_argument("--collector-only", action="store_true")
    parser.add_argument("--dashboard-only", action="store_true")
    browser_group = parser.add_mutually_exclusive_group()
    browser_group.add_argument("--open-browser", dest="open_browser", action="store_true")
    browser_group.add_argument("--no-open-browser", dest="open_browser", action="store_false")
    parser.set_defaults(open_browser=env_bool("OTE_LIVE_OPEN_BROWSER", False))
    parser.add_argument("--browser-wait-seconds", type=float, default=2.0)
    parser.add_argument("--collector-arg", action="append", default=[], help="Pass an extra arg through to run_live_collector.")
    parser.add_argument("--dashboard-arg", action="append", default=[], help="Pass an extra arg through to run_live_dashboard.")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def build_process_plan(args: argparse.Namespace) -> StackProcessPlan:
    dashboard_url = f"http://{args.dashboard_host}:{args.dashboard_port}"
    package_command = None
    export_command = None
    dashboard_command = None
    collector_command = None

    if not args.skip_policy_package:
        package_command = (
            str(args.python_exe),
            "-m",
            "ote_live.scripts.package_candidate_policies",
        )

    if not args.skip_manifest_export:
        export_command = (
            str(args.python_exe),
            "-m",
            "ote_live.scripts.export_live_runtime_manifests",
        )

    if not args.collector_only:
        dashboard_command = (
            str(args.python_exe),
            "-m",
            "ote_live.scripts.run_live_dashboard",
            "--db-path",
            str(args.db_path),
            "--asset",
            str(args.asset),
            "--timeframe",
            str(args.timeframe),
            "--host",
            str(args.dashboard_host),
            "--port",
            str(args.dashboard_port),
            "--refresh-interval-ms",
            str(args.dashboard_refresh_interval_ms),
            "--signal-limit",
            str(args.dashboard_signal_limit),
            *tuple(args.dashboard_arg),
        )

    if not args.dashboard_only:
        collector_command_parts = [
            str(args.python_exe),
            "-m",
            "ote_live.scripts.run_live_collector",
            "--db-path",
            str(args.db_path),
            "--asset",
            str(args.asset),
            "--dashboard-url",
            dashboard_url,
        ]
        if args.max_cycles is not None:
            collector_command_parts.extend(["--max-cycles", str(args.max_cycles)])
        collector_command_parts.extend(args.collector_arg)
        collector_command = tuple(collector_command_parts)

    return StackProcessPlan(
        package_command=package_command,
        export_command=export_command,
        dashboard_command=dashboard_command,
        collector_command=collector_command,
        dashboard_url=dashboard_url,
    )


def main() -> int:
    load_repo_env()
    parser = build_parser()
    args = parser.parse_args()
    if args.collector_only and args.dashboard_only:
        parser.error("--collector-only and --dashboard-only cannot be used together.")

    plan = build_process_plan(args)
    if args.dry_run:
        _print_plan(plan)
        return 0

    for command in (plan.package_command, plan.export_command):
        if command is None:
            continue
        _run_command(command)

    dashboard_process: subprocess.Popen[str] | None = None
    collector_process: subprocess.Popen[str] | None = None
    try:
        if plan.dashboard_command is not None:
            dashboard_process = _spawn_command(plan.dashboard_command)
            print(f"Dashboard starting on {plan.dashboard_url} (pid={dashboard_process.pid})")
            if args.open_browser:
                time.sleep(max(0.0, float(args.browser_wait_seconds)))
                webbrowser.open(plan.dashboard_url)

        if plan.collector_command is not None:
            collector_process = _spawn_command(plan.collector_command)
            print(f"Collector starting (pid={collector_process.pid})")

        if collector_process is not None:
            return collector_process.wait()
        if dashboard_process is not None:
            return dashboard_process.wait()
        return 0
    except KeyboardInterrupt:
        print("Stopping live stack...")
        return 130
    finally:
        for process in (collector_process, dashboard_process):
            _terminate_process(process)


def _run_command(command: Sequence[str]) -> None:
    print(f"Running: {_format_command(command)}")
    subprocess.run(
        tuple(command),
        cwd=REPO_ROOT,
        check=True,
        text=True,
    )


def _spawn_command(command: Sequence[str]) -> subprocess.Popen[str]:
    print(f"Launching: {_format_command(command)}")
    return subprocess.Popen(
        tuple(command),
        cwd=REPO_ROOT,
        text=True,
    )


def _terminate_process(process: subprocess.Popen[str] | None) -> None:
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def _print_plan(plan: StackProcessPlan) -> None:
    print(f"dashboard_url={plan.dashboard_url}")
    for label, command in (
        ("package", plan.package_command),
        ("export", plan.export_command),
        ("dashboard", plan.dashboard_command),
        ("collector", plan.collector_command),
    ):
        print(f"{label}={'-' if command is None else _format_command(command)}")


def _format_command(command: Sequence[str]) -> str:
    return " ".join(_quote_arg(part) for part in command)


def _quote_arg(value: str) -> str:
    if not value or any(ch.isspace() for ch in value):
        escaped = value.replace('"', '\\"')
        return f'"{escaped}"'
    return value


if __name__ == "__main__":
    raise SystemExit(main())
