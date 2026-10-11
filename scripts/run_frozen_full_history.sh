#!/bin/sh
# R0 follow-up: full-history rebuild vs frozen for the ES configs. Marker: FULL HISTORY COMPLETE
PY=ote_venv/Scripts/python.exe
E="2026-06-16T13:30:00+00:00 2026-03-09T06:00:00+00:00 2025-11-03T07:00:00+00:00"
for d in b2_ict_prepared_20260921_v1 b2_ict_specialists_prepared_20260921_v1 b2_frvp_july_prepared_20260921_v1 b2_frvp_setups_prepared_20260921_v1; do
  echo "== $d $(date -u +%H:%M:%S)"; $PY scripts/audit_frozen_full_history.py artifacts/$d ES $E 2>&1 | grep -vi warn | tail -1 | cut -c1-600
done
echo "FULL HISTORY COMPLETE $(date -u +%H:%M:%S)"
