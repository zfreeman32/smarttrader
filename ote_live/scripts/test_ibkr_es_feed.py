from __future__ import annotations

import argparse
import json
import time

from ote_live.env import load_repo_env
from ote_live.ingestion.ibkr import IBAPIUnavailableError, IBKRConfig
from ote_live.ingestion.ibkr.service import IBKRMarketDataService


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Read-only IBKR ES market-data smoke test. This script never places orders."
        )
    )
    parser.add_argument(
        "--updates",
        type=int,
        default=10,
        help="Number of changed quote/bar snapshots to print before exiting.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=120.0,
        help="Maximum time to wait for printed updates.",
    )
    return parser


def main() -> int:
    load_repo_env()
    args = build_parser().parse_args()
    config = IBKRConfig.from_env()
    if not config.enabled:
        print(
            "IBKR_ENABLED=false. Set IBKR_ENABLED=true after TWS or IB Gateway "
            "is authenticated in paper mode."
        )
        return 2

    service = IBKRMarketDataService(config)
    try:
        service.start()
        startup_timeout = (
            config.handshake_timeout_seconds
            + config.contract_timeout_seconds
            + config.history_timeout_seconds
        )
        if not service.wait_until_ready(startup_timeout):
            print(json.dumps(service.get_status(), indent=2, default=str))
            return 1

        contract = service.get_active_contract()
        status = service.get_status()
        print("Selected contract:")
        print(json.dumps(None if contract is None else contract.identity, indent=2))
        print(f"Next roll: {status.get('next_roll_at')}")
        print("Streaming read-only quote and five-minute bar snapshots. Ctrl+C exits.")

        deadline = time.monotonic() + max(1.0, float(args.timeout_seconds))
        prior_signature = None
        printed = 0
        required_updates = max(1, int(args.updates))
        while printed < required_updates and time.monotonic() < deadline:
            current_status = service.get_status()
            if current_status.get("connection_state") == "error":
                print("IBKR feed entered an error state:")
                print(json.dumps(current_status, indent=2, default=str))
                return 1
            quote = service.get_quote_snapshot()
            bars = service.get_bars(limit=1)
            latest_bar = bars[-1] if bars else None
            signature = (
                None if quote is None else quote.last_update_timestamp_utc,
                None if latest_bar is None else latest_bar.received_at_utc,
                None if latest_bar is None else latest_bar.close,
            )
            if signature != prior_signature:
                print(
                    json.dumps(
                        {
                            "quote": None
                            if quote is None
                            else {
                                "bid": quote.bid,
                                "ask": quote.ask,
                                "last": quote.last,
                                "bid_size": quote.bid_size,
                                "ask_size": quote.ask_size,
                                "last_size": quote.last_size,
                                "volume": quote.reported_volume,
                                "market_data_type": quote.market_data_type,
                                "received_at_utc": quote.last_update_timestamp_utc,
                            },
                            "bar": None
                            if latest_bar is None
                            else {
                                "timestamp_utc": latest_bar.timestamp_utc,
                                "open": latest_bar.open,
                                "high": latest_bar.high,
                                "low": latest_bar.low,
                                "close": latest_bar.close,
                                "volume": latest_bar.volume,
                                "conId": latest_bar.conid,
                                "local_symbol": latest_bar.local_symbol,
                                "is_complete": latest_bar.is_complete,
                            },
                        },
                        default=str,
                    )
                )
                prior_signature = signature
                printed += 1
            time.sleep(0.5)
        if printed < required_updates:
            print(
                f"Timed out after receiving {printed} of "
                f"{required_updates} requested feed updates."
            )
            print(json.dumps(service.get_status(), indent=2, default=str))
            return 1
        return 0
    except IBAPIUnavailableError as exc:
        print(str(exc))
        return 2
    except KeyboardInterrupt:
        return 130
    finally:
        service.stop()


if __name__ == "__main__":
    raise SystemExit(main())
