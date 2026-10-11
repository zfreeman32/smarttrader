#!/bin/sh
# R1 v2 performance split: which EURUSD strategies carry memory. Marker: STRATEGY MEMORY AUDIT COMPLETE
PY=ote_venv/Scripts/python.exe
for t in 2026-03-10T12:00:00+00:00 2025-12-23T15:00:00+00:00; do
  echo "== $t $(date -u +%H:%M:%S)"; $PY scripts/audit_strategy_memory.py artifacts/b2_ote_prepared_20260921_v1 EURUSD $t 2>&1 | grep -vi warn | tail -4 | cut -c1-900
done
echo "STRATEGY MEMORY AUDIT COMPLETE $(date -u +%H:%M:%S)"
