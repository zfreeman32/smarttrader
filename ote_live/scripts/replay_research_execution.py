"""Replay an explicitly supplied diagnostic A7 stream; never start collectors.

Use a separate output database. JSONL records are ordered by observation time:
{"kind":"opportunity", "payload": <ResearchOpportunity JSON>}
{"kind":"bar", "observed_at": <UTC ISO time>, "payload": <MarketBar JSON>}
{"kind":"finish", "observed_at": <UTC ISO time>}
Omitting finish preserves open exposure for a subsequent invocation.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from ote_live.contracts.market_data import MarketBar
from ote_live.contracts.research_execution import ResearchContract, ResearchOpportunity
from ote_live.storage.db import SQLiteLiveDataStore
from ote_live.storage.research_execution import ResearchExecutionLedger


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--started-at", required=True)
    args = parser.parse_args(argv)
    if len({path.resolve() for path in (args.contract, args.events, args.database, args.report)}) != 4:
        parser.error("Contract, events, database and report paths must be distinct")
    contract = ResearchContract.model_validate_json(args.contract.read_text(encoding="utf-8"))
    with SQLiteLiveDataStore(args.database) as store:
        ledger = ResearchExecutionLedger(store)
        ledger.register(args.run_id, contract, started_at=datetime.fromisoformat(args.started_at))
        with args.events.open(encoding="utf-8") as stream:
            for number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                record = json.loads(line)
                try:
                    if record["kind"] == "opportunity":
                        ledger.submit(args.run_id, ResearchOpportunity.model_validate(record["payload"]))
                    elif record["kind"] == "bar":
                        ledger.process_bar(args.run_id, MarketBar.model_validate(record["payload"]),
                                           observed_at=datetime.fromisoformat(record["observed_at"]))
                    elif record["kind"] == "finish":
                        ledger.finish(args.run_id, observed_at=datetime.fromisoformat(record["observed_at"]))
                    else:
                        raise ValueError("Unknown event kind")
                except (ValueError, KeyError) as exc:
                    raise ValueError(f"Invalid replay event at line {number}: {exc}") from exc
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(ledger.report(args.run_id), indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
