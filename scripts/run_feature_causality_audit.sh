#!/bin/sh
# Retrain step R0: causal-prefix, bounded-memory and frozen-contract audit of the five frozen dataset configs.
# Marker: AUDIT R0 COMPLETE
PY=ote_venv/Scripts/python.exe
for d in b2_ict_prepared_20260921_v1 b2_ict_specialists_prepared_20260921_v1 b2_frvp_july_prepared_20260921_v1 b2_frvp_setups_prepared_20260921_v1; do
  for t in 2026-06-16T13:30:00+00:00 2026-03-09T06:00:00+00:00 2025-11-03T07:00:00+00:00; do
    echo "== $d $t $(date -u +%H:%M:%S)"; $PY scripts/audit_feature_causality.py artifacts/$d ES $t 2>&1 | grep -vi warn | tail -1 | cut -c1-200
  done
done
echo "== b2_ote_prepared_20260921_v1 $(date -u +%H:%M:%S)"
$PY scripts/audit_feature_causality.py artifacts/b2_ote_prepared_20260921_v1 EURUSD 2026-03-10T12:00:00+00:00 longs=9000 cuts=4 2>&1 | grep -vi warn | tail -1 | cut -c1-200
echo "AUDIT R0 COMPLETE $(date -u +%H:%M:%S)"
