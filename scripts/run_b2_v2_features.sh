#!/bin/sh
# Retrain R2a: v2 feature tapes for the four B2 groups, sequential. Marker: R2A FEATURES COMPLETE
PY=ote_venv/Scripts/python.exe
ES="2026-06-16T13:30:00+00:00 2026-03-09T06:00:00+00:00 2025-11-03T07:00:00+00:00"
run() { echo "== $1 $(date -u +%H:%M:%S)"; $PY -u scripts/build_b2_v2_features.py --group $1 --base-features-root artifacts/$2 --output-root artifacts/$3 --check-ends $4 2>&1 | grep -viE "warn|^\s*$" | tail -8; }
run ict b2_ict_features_20260920_v2 b2_ict_features_v2_20261006 "$ES"
run frvp_july b2_frvp_july_features_20260921_v2 b2_frvp_july_features_v2_20261006 "$ES"
run frvp_setups b2_frvp_setups_features_20260921_v1 b2_frvp_setups_features_v2_20261006 "$ES"
run ote b2_ote_features_20260920_v1 b2_ote_features_v2_20261006 "2026-03-10T12:00:00+00:00 2025-11-20T14:00:00+00:00"
echo "R2A FEATURES COMPLETE $(date -u +%H:%M:%S)"
