#!/bin/sh
# Retrain R2a OTE rerun after the 2026-10-06 acceptance FAIL: five more dropped strategies (rocwb, volume_flow,
# london_breakout, end-of-month nifty: whole-block look-ahead; dollar_cost_averaging: window-dependent) and the frozen v2
# strategy one-hot vocabulary. Third end bar 2025-07-16 is new (held out). Marker: R2A OTE RERUN COMPLETE
PY=ote_venv/Scripts/python.exe
echo "== ote $(date -u +%H:%M:%S)"
$PY -u scripts/build_b2_v2_features.py --group ote --base-features-root artifacts/b2_ote_features_20260920_v1 \
  --output-root artifacts/b2_ote_features_v2_20261006b \
  --check-ends 2026-03-10T12:00:00+00:00 2025-11-20T14:00:00+00:00 2025-07-16T13:00:00+00:00 2>&1 | grep -viE "warn|^\s*$" | tail -8
echo "R2A OTE RERUN COMPLETE $(date -u +%H:%M:%S)"
