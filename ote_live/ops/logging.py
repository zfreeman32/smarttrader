from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LOG_PATH = REPO_ROOT / "ote_live" / "runtime_data" / "logs" / "live_signal_service.log"


class JsonLogFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp_utc": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "process": record.process,
            "thread": record.threadName,
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, sort_keys=True)


class UtcConsoleFormatter(logging.Formatter):
    converter = staticmethod(lambda timestamp: datetime.fromtimestamp(timestamp, tz=UTC).timetuple())


def configure_live_logging(
    level: str,
    *,
    log_path: str | Path | None = DEFAULT_LOG_PATH,
    max_bytes: int = 5_000_000,
    backup_count: int = 10,
) -> None:
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, str(level).upper(), logging.INFO))
    # The official IB API logs every decoded protocol payload at INFO,
    # including complete contract chains and historical-bar batches.
    logging.getLogger("ibapi").setLevel(logging.WARNING)

    for handler in tuple(root_logger.handlers):
        root_logger.removeHandler(handler)
        handler.close()

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(
        UtcConsoleFormatter(
            fmt="%(asctime)s %(levelname)s %(name)s - %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%SZ",
        )
    )
    root_logger.addHandler(console_handler)

    if log_path is not None:
        resolved_log_path = Path(log_path)
        resolved_log_path.parent.mkdir(parents=True, exist_ok=True)
        rotating_handler = RotatingFileHandler(
            resolved_log_path,
            maxBytes=int(max_bytes),
            backupCount=int(backup_count),
            encoding="utf-8",
        )
        rotating_handler.setFormatter(JsonLogFormatter())
        root_logger.addHandler(rotating_handler)

    logging.captureWarnings(True)
