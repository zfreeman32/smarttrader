# Live Service Supervision Notes

## Recommended Windows setup

- Run the collector with `WinSW` or `NSSM` as the primary Windows service wrapper.
- Use Task Scheduler only as a backup launcher after reboot, not as the main supervisor.
- Point the service at `scripts/run_live_signal_service.ps1` so log, heartbeat, and DB paths stay consistent.
- The PowerShell launcher now supports bounded restart loops. Use `-NoRestartOnFailure` when an external Windows service manager should own restart policy.

## Runtime artifacts to watch

- Rotating log file: `ote_live/runtime_data/logs/live_signal_service.log`
- Heartbeat/health snapshot: `ote_live/runtime_data/health/live_signal_service_heartbeat.json`
- SQLite audit store: `ote_live/runtime_data/live_market_data.sqlite3`

## Recommended restart policy

- Restart on non-zero exit.
- Allow a short backoff, such as 10 to 30 seconds, between retries.
- Alert if the process restarts repeatedly within the same hour.
- The provided launcher defaults to a 15-second delay, `MaxRestartCount=10`, and `MaxRestartsPerHour=6`.

## Operator checks after restart

- Confirm the heartbeat JSON updates every collector cycle.
- Confirm `service_status` returns to `running`.
- Check for `ops.startup`, `ops.shutdown`, `collector.bootstrap`, and `ops.disk_monitor` events in the database.
- Confirm unresolved gaps do not increase unexpectedly after the restart.
