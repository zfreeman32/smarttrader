#!/bin/sh
# Second exact app-window pass for the last EURUSD residuals. Marker: EXACT EURUSD 2 COMPLETE
T=docs/audits/section4_tools_20261002/exact_window_check.py
PY=ote_venv/Scripts/python.exe
O=htf_bar_timestamp_semantics=legacy_right_closed
echo "== EU_R_weekend 22 svesc $(date -u +%H:%M:%S)"; $PY $T EU_R_weekend 22 auto:strategy__svesc_signals__svesc_sell_signal $O 2>&1 | grep -E "^20|Trace|Error"
mv artifacts/s4parity_20261002_v1/classification/exact_window_EU_R_weekend_22.json artifacts/s4parity_20261002_v1/classification/exact_window_EU_R_weekend_22svesc.json
echo "== EU_H_session_dst 26 dual long $(date -u +%H:%M:%S)"; $PY $T EU_H_session_dst 26 auto:strategy__dual_momentum_signals__trade_signal__long $O 2>&1 | grep -E "^20|Trace|Error"
echo "EXACT EURUSD 2 COMPLETE $(date -u +%H:%M:%S)"
