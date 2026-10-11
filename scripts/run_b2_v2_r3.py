"""Retrain R3: train, then review, every v2 roster model in sequence; a failed model never stops the queue.

Per model (each step a separate process with its own log):
  1. train   ``scripts/run_b2_v2_source_history.py train ID`` (includes the source-history preflight). Skipped when
             the model's training manifest already says ``trained_pending_v2_review``. A model directory left by an
             interrupted run (any other status) is reported as FAILED and never deleted or overwritten.
  2. review  ``scripts/review_b2_v2_models.py ID --report <audit dir>/ID.json``. Skipped when the report exists.
Status after every step: ``<audit dir>/status.json``; log lines ``ID n train: OK|SKIP|FAILED`` / ``review: PASS|FAILED``.
Order: ES XGBoost first (fast), then EURUSD (26 XGBoost, then TCN 22, 21, 23, 25). Rerunning resumes.
Nothing is installed or activated.

Usage: python scripts/run_b2_v2_r3.py [--ids 2 3 ...]
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ROSTER = ROOT / "docs/audits/b2_v2_model_data_roster_20261006.csv"
OUTPUT = ROOT / "models/b2_v2_retrained_20261006"
AUDIT = ROOT / "docs/audits/b2_v2_review_20261007"
LOGS = ROOT / "logs/b2_v2_r3_20261007"
ORDER = (12, 13, 14, 15, 16, 18, 19, 2, 3, 5, 10, 26, 22, 21, 23, 25)


def now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def step(args: list[str], log: Path) -> tuple[bool, str]:
    with log.open("a", encoding="utf-8") as handle:
        handle.write(f"\n=== {now()} {' '.join(args)}\n")
        handle.flush()
        code = subprocess.run([sys.executable, "-B", "-u", *args], cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT).returncode
    if code == 0:
        return True, ""
    tail = [line.strip() for line in log.read_text(encoding="utf-8", errors="replace").splitlines() if line.strip()]
    return False, tail[-1][:300] if tail else f"exit {code}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ids", nargs="*", type=int, default=list(ORDER))
    args = parser.parse_args()
    targets = {int(r["model_id"]): r["target"] for r in csv.DictReader(ROSTER.open(encoding="utf-8"))}
    unknown = [i for i in args.ids if i not in targets]
    if unknown:
        parser.error(f"Not in the v2 roster: {unknown}")
    AUDIT.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    status_path = AUDIT / "status.json"
    status = json.loads(status_path.read_text()) if status_path.exists() else {}

    def record(model_id: int, **values) -> None:
        status.setdefault(str(model_id), {}).update(values, updated_at_utc=now())
        status_path.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")

    print(f"R3 start {now()}: {args.ids}", flush=True)
    for model_id in args.ids:
        out = OUTPUT / f"{model_id:02d}_{targets[model_id]}"
        manifest = out / "v2_training_manifest.json"
        trained = manifest.exists() and json.loads(manifest.read_text()).get("status") == "trained_pending_v2_review"
        if trained:
            print(f"ID {model_id} train: SKIP (already trained)", flush=True)
        elif out.exists():
            reason = "interrupted training output exists; inspect and remove it to retrain"
            record(model_id, train="FAILED", train_error=reason)
            print(f"ID {model_id} train: FAILED ({reason})", flush=True)
            continue
        else:
            record(model_id, train="RUNNING", train_started_at_utc=now())
            print(f"ID {model_id} train: start {now()}", flush=True)
            ok, error = step(["scripts/run_b2_v2_source_history.py", "train", str(model_id)], LOGS / f"{model_id:02d}_train.log")
            record(model_id, train="OK" if ok else "FAILED", train_error=error or None, train_finished_at_utc=now())
            print(f"ID {model_id} train: {'OK' if ok else 'FAILED (' + error + ')'} {now()}", flush=True)
            if not ok:
                continue
        report = AUDIT / f"{model_id}.json"
        if report.exists():
            record(model_id, review="PASS")
            print(f"ID {model_id} review: PASS (report exists)", flush=True)
            continue
        ok, error = step(["scripts/review_b2_v2_models.py", str(model_id), "--report", str(report)], LOGS / f"{model_id:02d}_review.log")
        record(model_id, review="PASS" if ok else "FAILED", review_error=error or None)
        print(f"ID {model_id} review: {'PASS' if ok else 'FAILED (' + error + ')'} {now()}", flush=True)
    passed = [i for i in args.ids if status.get(str(i), {}).get("review") == "PASS"]
    failed = [i for i in args.ids if i not in passed]
    print(f"R3 COMPLETE {now()}: {len(passed)}/{len(args.ids)} reviewed PASS; not passed: {failed}", flush=True)
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
