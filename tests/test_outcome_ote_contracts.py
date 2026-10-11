"""OTE evaluation is independent of retrospective labels and their horizons."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model_testing.outcome_atr import completed_structural_atr, structural_atr_at
from model_testing.outcome_artifacts import run_outcome_evaluation, replay_outcome_run
from model_testing.outcome_contracts import load_contracts, resolve_contract, file_hash
from model_testing.outcome_engine import EvaluationTape, evaluate_signal, simulate_positions
from model_testing.outcome_lineage import baseline_completion_audit
from test_outcome_contracts import signal, simulator
from pruned_archive import is_pruned, pruned_registrations


FAMILIES = [('reversal', 5, 2, 1, 120, 60), ('reversal', 1, 2, 1, 600, 60),
            ('continuation', 5, .7, .8, 36, 30), ('breakout', 5, .7, .9, 24, 30),
            ('breakout_sustained', 5, .7, .9, 24, 30)]


def ote_contract(family='breakout', timeframe=5):
    return resolve_contract(instrument='EURUSD', strategy='ote', model_family=family,
                            setup_family=family, timeframe_minutes=timeframe, side='long')


def ote_signal(family='breakout', timeframe=5, **changes):
    c = ote_contract(family, timeframe)
    minutes = 60 if family == 'reversal' else 30
    proof = dict(definition=c.risk.atr_definition,
                 producer_id=f'b1_structural_{"1h" if minutes == 60 else "30m"}_wilder14_v1',
                 timeframe_minutes=minutes, value=.002,
                 source_close_ts='2026-09-14T14:00Z', history_start_ts='2026-09-14T00:00Z',
                 available_at_ts='2026-09-14T14:00Z', coordinate_id=None, source_hash='d' * 64)
    values = dict(model_id=f'ote_{family}_synthetic', model_family=family, strategy='ote',
                  setup_family=family, instrument='EURUSD', timeframe_minutes=timeframe,
                  atr=.002, atr_definition=c.risk.atr_definition, atr_provenance=proof, coordinate_id=None)
    values.update(changes)
    return signal(**values)


def fx_tape(timeframe=5, n=150, *, history=False):
    start = '2026-09-14T00:00Z' if history else f'2026-09-14T14:{timeframe:02d}Z'
    times = pd.date_range(start, periods=n, freq=f'{timeframe}min')
    bars = pd.DataFrame(dict(open_ts=times, open=1.1, high=1.1002, low=1.0998, close=1.1))
    schedule = pd.DataFrame(dict(open_ts=times, session_id='synthetic_fx'))
    return EvaluationTape(bars, schedule, timeframe_minutes=timeframe, calendar_id='synthetic_fx_v1',
        calendar_source='explicit synthetic fixture', coverage_start=times[0] - pd.Timedelta(minutes=timeframe),
        coverage_end=times[-1] + pd.Timedelta(minutes=timeframe))


@pytest.mark.parametrize('family,timeframe,stop,target,bars,minutes', FAMILIES)
@pytest.mark.parametrize('side', ['long', 'short'])
def test_ote_barriers_and_duration_start_at_actual_entry(family, timeframe, stop, target, bars, minutes, side):
    c, s = ote_contract(family, timeframe), ote_signal(family, timeframe, side=side)
    result = evaluate_signal(s, fx_tape(timeframe, n=max(60, bars)), c)
    sign = 1 if side == 'long' else -1
    assert c.contract_version == 2 and c.risk.rule == 'atr'
    assert s.atr_provenance.timeframe_minutes == minutes
    assert result['paper_contract_outcome'] == 'TIMEOUT_FLAT'
    assert result['stop_price'] == pytest.approx(1.1 - sign * stop * .002)
    assert result['target_price'] == pytest.approx(1.1 + sign * target * .002)
    assert result['entry_ts'] > s.known_at_ts >= s.event_anchor_ts
    assert result['paper_exit_ts'] == result['entry_ts'] + pd.Timedelta(minutes=bars * timeframe)
    assert result['paper_exit_ts'] == result['timeout_ts']


@pytest.mark.parametrize('side', ['long', 'short'])
@pytest.mark.parametrize('case,expected', [('target', 'TARGET'), ('stop', 'STOP'), ('collision', 'STOP'),
                                         ('profit', 'TIMEOUT_WIN'), ('loss', 'TIMEOUT_LOSS')])
def test_ote_gross_outcomes_and_fx_cost_reconciliation(side, case, expected):
    s, c, t = ote_signal(side=side), ote_contract(), fx_tape()
    if case in ('target', 'collision'):
        t.bars.iloc[0, t.bars.columns.get_loc('high' if side == 'long' else 'low')] = 1.102 if side == 'long' else 1.098
    if case in ('stop', 'collision'):
        t.bars.iloc[0, t.bars.columns.get_loc('low' if side == 'long' else 'high')] = 1.098 if side == 'long' else 1.102
    if case in ('profit', 'loss'):
        sign = (1 if side == 'long' else -1) * (1 if case == 'profit' else -1)
        t.bars.iloc[c.risk.timeout_bars - 1, t.bars.columns.get_loc('close')] += sign * .0001
    r = evaluate_signal(s, t, c)
    assert r['paper_contract_outcome'] == expected
    if case == 'collision':
        assert r['barrier_resolution_method'] == 'adverse_first'
    p, = simulate_positions([s], [r], {c.contract_id: c}, simulator(accounting_unit='pips'))
    assert p['position_taken']
    assert p['gross_pnl'] == pytest.approx(r['paper_signed_price_movement'] / .0001)
    assert p['total_cost'] == p['spread_cost'] + p['slippage_cost'] + p['commission_cost']
    assert p['slippage_cost'] == .3 and p['commission_cost'] == .35
    assert p['net_pnl'] == p['gross_pnl'] - p['total_cost']
    if case == 'profit':
        assert p['gross_pnl'] > 0 > p['net_pnl']  # Paper winner can lose after costs.


@pytest.mark.parametrize('minutes', [30, 60])
def test_spot_structural_atr_publishes_only_completed_periods_and_price_units(minutes):
    t = fx_tape(history=True, n=360)
    t.bars.loc[pd.Timestamp('2026-09-14T14:25Z'), 'high'] = 1.1042
    def proof(tape, known):
        return structural_atr_at(completed_structural_atr(tape, timeframe_minutes=minutes, instrument='EURUSD'),
                                 known, coordinate_id=None)
    p = proof(t, '2026-09-14T14:05Z')
    assert p.source_close_ts == pd.Timestamp('2026-09-14T14:00Z')
    assert p.value == pytest.approx(.0004) and p.value / .0001 == pytest.approx(4)
    completed = proof(t, pd.Timestamp('2026-09-14T14:00Z') + pd.Timedelta(minutes=minutes))
    assert completed.value == pytest.approx(.0004 + .004 / 14)
    t.bars.loc[t.bars.index >= pd.Timestamp('2026-09-14T14:10Z'), 'high'] = 1.2
    assert proof(t, '2026-09-14T14:05Z') == p
    # The same ID-less tape is invalid for futures, never implicitly spot.
    assert structural_atr_at(completed_structural_atr(t, timeframe_minutes=minutes, instrument='ES'),
                             '2026-09-14T14:05Z', coordinate_id=None) is None


@pytest.mark.parametrize('defect', ['missing', 'provisional', 'corrupt', 'mixed_series'])
def test_30m_atr_never_fills_unknown_constituents(defect):
    t = fx_tape(history=True, n=360)
    ts = pd.Timestamp('2026-09-14T14:15Z')
    if defect == 'missing':
        t.bars = t.bars.drop(ts)
    elif defect == 'provisional':
        t.bars['is_complete'] = True
        t.bars.loc[ts, 'is_complete'] = False
    elif defect == 'corrupt':
        t.bars.loc[ts, 'high'] = float('nan')
    else:
        t.bars['coordinate_id'] = 'feed_a'
        t.bars.loc[ts, 'coordinate_id'] = 'feed_b'
    history = completed_structural_atr(t, timeframe_minutes=30, instrument='EURUSD')
    assert structural_atr_at(history, '2026-09-14T14:30Z', coordinate_id=None) is None


def test_30m_warmup_and_late_arrival_are_not_hourly_or_early():
    t = fx_tape(history=True, n=180)
    t.bars['observed_at_ts'] = t.bars.index + pd.Timedelta(minutes=5)
    t.bars.loc[pd.Timestamp('2026-09-14T06:55Z'), 'observed_at_ts'] = pd.Timestamp('2026-09-14T07:03Z')
    history = completed_structural_atr(t, timeframe_minutes=30, instrument='EURUSD')
    assert structural_atr_at(history, '2026-09-14T07:00Z', coordinate_id=None) is None
    p = structural_atr_at(history, '2026-09-14T07:03Z', coordinate_id=None)
    assert p.source_close_ts == pd.Timestamp('2026-09-14T07:00Z')
    assert p.timeframe_minutes == 30 and p.period == 14


@pytest.mark.parametrize('field', ['confirmation_ts', 'feature_max_source_close_ts', 'geometry_available_at',
                                 'atr_available_at', 'prediction_available_at', 'policy_available_at'])
def test_ote_rejects_future_inputs(field):
    s = ote_signal(**{field: '2026-09-14T14:05Z'})
    r = evaluate_signal(s, fx_tape(), ote_contract())
    assert r['paper_contract_outcome'] == 'INVALID' and r['entry_ts'] is None


def test_wrong_atr_family_cannot_normalize_or_resolve_paper():
    s = ote_signal().model_copy(update={'atr_provenance': ote_signal('reversal').atr_provenance})
    r = evaluate_signal(s, fx_tape(), ote_contract())
    assert r['paper_contract_outcome'] == 'INVALID'
    assert r['atr_status'] == 'structural_atr_identity_mismatch'
    assert r['forward_return_1_status'] == 'OK' and r['forward_return_1_atr'] is None


def test_ote_labels_and_sizes_cannot_change_paper_or_gross_response():
    s, c, t = ote_signal(), ote_contract(), fx_tape()
    first = evaluate_signal(s, t, c)
    changed = evaluate_signal(s.model_copy(update=dict(training_label_outcome=0, quantity=100, policy_eligible=False)), t, c)
    assert {k: v for k, v in first.items() if not k.startswith('training_label_')} == {
        k: v for k, v in changed.items() if not k.startswith('training_label_')}
    different_risk = c.model_copy(update={'risk': c.risk.model_copy(update={'target_atr': 99, 'timeout_bars': 3})})
    other = evaluate_signal(s, t, different_risk)
    assert {k: v for k, v in first.items() if k.startswith('forward_')} == {
        k: v for k, v in other.items() if k.startswith('forward_')}


def test_mixed_ote_atr_artifacts_derive_independently_and_replay(tmp_path):
    signals = [ote_signal(family, event_id=family, atr=999, atr_provenance=None)
               for family in ('reversal', 'continuation', 'breakout', 'breakout_sustained')]
    signals.append(ote_signal(event_id='pooled', model_family='meta'))
    before = [s.model_dump() for s in signals]
    run = tmp_path / 'run'
    result = run_outcome_evaluation(signals, fx_tape(history=True, n=360), output_dir=run,
                                    derive_structural_atr=True, simulator=simulator(accounting_unit='pips'))
    assert replay_outcome_run(run, tmp_path / 'replay') == result
    assert [s.model_dump() for s in signals] == before
    rows = pd.read_csv(run / 'evaluated_signals.csv')
    assert rows.atr_price.tolist() == pytest.approx([.0004] * 5)
    assert rows.atr_status.eq('verified_definition').all()
    assert rows.training_label_outcome.eq(1).all()
    assert rows.evaluation_contract_version.eq(2).all()
    pooled = rows.loc[rows.event_id.eq('pooled')].iloc[0]
    assert pooled.paper_contract_outcome == 'NOT_EVALUATED' and not pooled.position_taken
    assert pooled.forward_return_48_status == 'OK'
    evidence = [json.loads(line) for line in (run / 'signal_evidence.jsonl').read_text().splitlines()]
    assert [s['atr_provenance']['timeframe_minutes'] for s in evidence] == [60, 30, 30, 30, 30]
    assert result['engine_version'] == 'b1-v5' and not result['scored_collection_enabled']


def test_ote_release_changes_only_five_contracts_and_keeps_costs():
    old = load_contracts(ROOT / 'configs/outcome_contracts/b1_v2.json')
    active = load_contracts()
    changed = 0
    for key, c in old.items():
        if key in active:
            assert active[key] == c
        else:
            new = active[key.removesuffix('_v1') + '_v2']
            assert c.strategy == 'ote' and c.risk.rule == 'UNRESOLVED'
            assert new.costs == c.costs and new.sha256 != c.sha256
            assert new.contract_version == c.contract_version + 1
            changed += 1
    assert changed == 5 and len(active) == 45


def test_completion_gate_fails_archived_unresolved_baseline_and_passes_active():
    old = baseline_completion_audit(catalog_path=ROOT / 'configs/outcome_contracts/b1_v2.json')
    assert not old['complete'] and len(old['unresolved_included_bindings']) == 5
    current = baseline_completion_audit()
    assert current['complete'] and current['defined_risk_count'] == 37
    assert current['excluded_risk_placeholder_count'] == 8
    assert current['registrations'] + pruned_registrations() == 120 and not current['registry_binding_failures']
    assert not current['unresolved_included_bindings'] and not current['scored_collection_enabled']


def test_native_1m_touch_reports_its_actual_resolution_timeframe():
    s, c, t = ote_signal('reversal', 1), ote_contract('reversal', 1), fx_tape(1, n=650)
    t.bars.iloc[0, t.bars.columns.get_loc('high')] = 1.103
    r = evaluate_signal(s, t, c)
    assert r['paper_contract_outcome'] == 'TARGET'
    assert r['barrier_resolution_method'] == '1m_single_touch'
    assert r['entry_ts'] == pd.Timestamp('2026-09-14T14:01Z')
    assert r['paper_exit_ts'] == pd.Timestamp('2026-09-14T14:02Z')


@pytest.mark.parametrize('missing', [0, 3])
def test_spot_missing_entry_or_holding_bar_censors_without_search(missing):
    t = fx_tape()
    t.bars = t.bars.drop(t.bars.index[missing])
    r = evaluate_signal(ote_signal(), t, ote_contract())
    assert r['paper_contract_outcome'] == 'DATA_GAP'
    assert not r['paper_resolved']
    assert (r['entry_ts'] is None) == (missing == 0)


@pytest.mark.parametrize('incomplete', [0, 3])
def test_provisional_execution_path_cannot_resolve_a_trade(incomplete):
    t = fx_tape()
    t.bars['is_complete'] = True
    t.bars.iloc[incomplete, t.bars.columns.get_loc('is_complete')] = False
    r = evaluate_signal(ote_signal(), t, ote_contract())
    assert r['paper_contract_outcome'] == 'DATA_GAP'
    assert not r['paper_resolved']
    assert (r['entry_ts'] is None) == (incomplete == 0)


def test_ote_evidence_preserves_archives_label_sources_and_training_artifacts():
    evidence = json.loads((ROOT / 'docs/audits/b1_ote_resolution_20260918.json').read_text())
    for relative, digest in evidence['source_hashes'].items():
        if relative.startswith(('data/labeling/', 'configs/outcome_contracts/b1_v1', 'configs/outcome_contracts/b1_v2')):
            assert file_hash(ROOT / relative) == digest
    for relative, digest in evidence['registry_hashes'].items():
        if not is_pruned(relative):
            assert file_hash(ROOT / relative) == digest
    for artifact, digest in evidence['training_summary_snapshot_hashes'].items():
        if digest and not is_pruned(artifact):
            assert file_hash(ROOT / artifact / 'training_summary.json') == digest
    assert len(evidence['changed_contracts']) == 5
    assert not evidence['performance_used_to_select_contract']


def test_published_signal_schema_includes_both_structural_atr_proofs():
    from model_testing.outcome_engine import SignalEvidence
    schema = json.loads((ROOT / 'configs/outcome_contracts/signal_evidence.schema.json').read_text())
    assert schema == SignalEvidence.model_json_schema()
    assert schema['$defs']['Structural30mATREvidence']['properties']['timeframe_minutes']['const'] == 30
