#!/bin/sh
# Prompt 4 classification diagnostics, one at a time (no matrix job running). Marker: DIAGNOSTICS COMPLETE
T=docs/audits/section4_tools_20261002
PY=ote_venv/Scripts/python.exe
for s in ES_H_rth_maint ES_H_spring_dst ES_H_fall_dst ES_H_holiday ES_H_roll; do
  echo "== plain $s $(date -u +%H:%M:%S)"; $PY $T/variant_recompute.py ES $s plain 2>&1 | grep -E "cells|Error|Traceback"
done
for s in EU_H_weekend_dst EU_H_holiday EU_R_session EU_R_weekend EU_R_outage_edge; do
  echo "== htflegacy $s $(date -u +%H:%M:%S)"; $PY $T/variant_recompute.py EURUSD $s htflegacy htf_bar_timestamp_semantics=legacy_right_closed 2>&1 | grep -E "cells|Error|Traceback"
done
echo "DIAGNOSTICS COMPLETE $(date -u +%H:%M:%S)"
