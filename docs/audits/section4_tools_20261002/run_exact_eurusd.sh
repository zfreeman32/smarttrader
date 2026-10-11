#!/bin/sh
# Exact app-window checks for the EURUSD residuals (one endpoint each, sequential). Marker: EXACT EURUSD COMPLETE
T=docs/audits/section4_tools_20261002/exact_window_check.py
PY=ote_venv/Scripts/python.exe
O=htf_bar_timestamp_semantics=legacy_right_closed
echo "== EU_H_session_dst 21 $(date -u +%H:%M:%S)"; $PY $T EU_H_session_dst 21 auto:strategy__dual_momentum_signals__trade_signal__neutral $O 2>&1 | grep -E "^20|Trace|Error"
echo "== EU_H_holiday 21 $(date -u +%H:%M:%S)"; $PY $T EU_H_holiday 21 auto:continuation_bear_retest_quality_zscore_100 $O 2>&1 | grep -E "^20|Trace|Error"
echo "== EU_R_weekend 22 $(date -u +%H:%M:%S)"; $PY $T EU_R_weekend 22 auto:strategy__rocwb_signals__rocwb_buy_signal $O 2>&1 | grep -E "^20|Trace|Error"
echo "EXACT EURUSD COMPLETE $(date -u +%H:%M:%S)"
