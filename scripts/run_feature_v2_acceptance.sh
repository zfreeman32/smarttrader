#!/bin/sh
# Retrain R1 acceptance: R0 audit (causal prefix + bounded memory) on the v2 configs. Marker: V2 ACCEPTANCE COMPLETE
PY=ote_venv/Scripts/python.exe
A="scripts/audit_feature_causality.py"
run() { echo "== $1 $3 $(date -u +%H:%M:%S)"; $PY $A artifacts/$1 $2 $3 feature_version=2 $4 2>&1 | grep -vi warn | tail -1 | cut -c1-600; }
for t in 2026-06-16T13:30:00+00:00 2026-03-09T06:00:00+00:00 2025-11-03T07:00:00+00:00; do run b2_ict_prepared_20260921_v1 ES $t; done
run b2_ict_specialists_prepared_20260921_v1 ES 2026-03-09T06:00:00+00:00
for t in 2026-06-16T13:30:00+00:00 2026-03-09T06:00:00+00:00 2025-11-03T07:00:00+00:00; do run b2_frvp_july_prepared_20260921_v1 ES $t; done
run b2_frvp_setups_prepared_20260921_v1 ES 2025-11-03T07:00:00+00:00
run b2_ote_prepared_20260921_v1 EURUSD 2026-03-10T12:00:00+00:00 "longs=9000 cuts=3"
echo "V2 ACCEPTANCE COMPLETE $(date -u +%H:%M:%S)"
