#!/bin/sh
# R1 v2 acceptance, EURUSD, all strategies blocked and in-process. Marker: V2 EURUSD ACCEPTANCE COMPLETE
PY=ote_venv/Scripts/python.exe
for t in 2026-03-10T12:00:00+00:00 2025-11-20T14:00:00+00:00; do
  echo "== $t $(date -u +%H:%M:%S)"
  $PY scripts/audit_feature_causality.py artifacts/b2_ote_prepared_20260921_v1 EURUSD $t feature_version=2 strategy_timeout_seconds=None longs=9000,16000,24000 2>&1 | grep -vi warn | tail -1 | cut -c1-700
done
echo "V2 EURUSD ACCEPTANCE COMPLETE $(date -u +%H:%M:%S)"
