#!/bin/sh
# Retrain R2b: prepared roots on the v2 feature tapes (same labels/events and recipes as run_b2_retained_preparation.py
# JOBS; OTE labels/events from b2_ote_prepared_20260921_v1 lineage), sequential, then the FRVP recency root.
# Marker: R2B PREPARED COMPLETE
PY=ote_venv/Scripts/python.exe
export LOKY_MAX_CPU_COUNT=2
D=20261006
run() { echo "== $1 $(date -u +%H:%M:%S)"; $PY -B -u scripts/rebuild_b2_prepared.py --group $1 --features-root artifacts/$2 \
  --labels artifacts/$3 --events artifacts/$4 --output-root artifacts/b2_$1_prepared_v2_$D 2>&1 | grep -viE "warn|^\s*$" | grep -vE "^Merged rows" | tail -20
  test -f artifacts/b2_$1_prepared_v2_$D/manifest.json && echo "$1: DONE $(date -u +%H:%M:%S)" || echo "$1: FAILED (no manifest)"; }
L=b2_frvp_labels_20260920_v1
run frvp_july b2_frvp_july_features_v2_$D $L/frvp_es_primary_refresh_20260701/labels.csv.gz $L/frvp_es_primary_refresh_20260701/events.csv.gz
run ict b2_ict_features_v2_$D b2_ict_labels_20260921_v1/ict_es_labels.csv b2_ict_labels_20260921_v1/ict_es_events.csv
run ict_specialists b2_ict_features_v2_$D b2_ict_labels_20260921_v1/ict_es_labels.csv b2_ict_labels_20260921_v1/ict_es_events.csv
run frvp_setups b2_frvp_setups_features_v2_$D $L/frvp_es_primary_setup_targets_20260829/labels.csv.gz $L/frvp_es_primary_setup_targets_20260829/events.csv.gz
run ote b2_ote_features_v2_${D}b b2_ote_labels_20260920_v2/labels.csv.gz b2_ote_labels_20260920_v2/events.csv.gz
echo "== frvp_recency $(date -u +%H:%M:%S)"
$PY -B scripts/materialize_frvp_recency_prepared_root.py --base-prepared-root artifacts/b2_frvp_july_prepared_v2_$D/prepared \
  --output-prepared-root artifacts/b2_frvp_recency_prepared_v2_$D/prepared --target long_frvp_reversal \
  --source-dataset artifacts/b2_frvp_july_prepared_v2_$D/dataset.parquet --half-life-days 730 --floor 0.20 2>&1 | grep -viE "warn" | tail -5
echo "R2B PREPARED COMPLETE $(date -u +%H:%M:%S)"
