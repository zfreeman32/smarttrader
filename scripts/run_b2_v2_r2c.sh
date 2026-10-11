#!/bin/sh
# Retrain R2c: v2 validation views (EURUSD root; ES roots already prepared, resumed), then per-model source history
# prepare for the 16 roster models and the check (preflight) for each. No model is trained. Marker: R2C COMPLETE
PY=ote_venv/Scripts/python.exe
export LOKY_MAX_CPU_COUNT=2
CHK=docs/audits/b2_v2_source_history_check_20261006
mkdir -p $CHK
echo "== views $(date -u +%H:%M:%S)"
$PY -B -u scripts/prepare_b2_v2_validation_views.py 2>&1 | grep -viE "warn|^Preparing|^Prepared|^\s*$" | tail -12
test -f configs/b2_v2_validation_views_lock_20261006.json || { echo "views: FAILED (no lock)"; echo "R2C COMPLETE (FAILED) $(date -u +%H:%M:%S)"; exit 1; }
for id in 2 3 5 10 12 13 14 15 16 18 19; do
  echo "== prepare $id $(date -u +%H:%M:%S)"
  $PY -B -u scripts/run_b2_v2_source_history.py prepare $id 2>&1 | grep -viE "warn" | tail -3
done
echo "== prepare ote $(date -u +%H:%M:%S)"
$PY -B -u scripts/run_b2_v2_source_history.py prepare ote 2>&1 | grep -viE "warn" | tail -8
for id in 2 3 5 10 12 13 14 15 16 18 19 21 22 23 25 26; do
  echo "== check $id $(date -u +%H:%M:%S)"
  $PY -B -u scripts/run_b2_v2_source_history.py check $id --report $CHK/$id.json 2>&1 | grep -viE "warn" | tail -4
  test -f $CHK/$id.json && echo "check $id: PASS" || echo "check $id: FAILED"
done
echo "R2C COMPLETE $(date -u +%H:%M:%S)"
