"""Build versioned training packages for exactly the retained B2 targets."""
from __future__ import annotations
import argparse
import gc
import gzip
import json
import sys
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from preprocessing.config import PreprocessingConfig
from preprocessing.pipeline import FeaturePreprocessingPipeline
from scripts.rebuild_b2_frvp_labels import dump
from model_testing.artifact_lineage_audit import digest

GROUPS = {
    'frvp_july': ('artifacts/frvp_es_primary_refresh_20260701/phase04/prepared', ['long_frvp_continuation', 'short_frvp_continuation', 'long_frvp_meta', 'short_frvp_meta', 'long_frvp_reversal']),
    'frvp_setups': ('artifacts/frvp_es_primary_setup_targets_20260829/phase04/prepared', ['long_frvp_continuation_setup3', 'long_frvp_continuation_setup5', 'long_frvp_reversal_setup1', 'long_frvp_continuation_setup2', 'long_frvp_reversal_setup6']),
    'ict': ('artifacts/ict_es_primary_refresh_20260724_spacing_refit_final_confirm/phase04_prepared/prepared', ['long_ict_continuation', 'long_ict_reversal', 'short_ict_reversal', 'short_ict_continuation', 'long_ict_meta', 'short_ict_meta']),
    'ict_specialists': ('artifacts/ict_short_setup_targets_20260811_audit/phase04_prepared/prepared', ['short_ict_continuation_premium_discount_continuation', 'short_ict_reversal_ifvg_reversal', 'short_ict_reversal_sweep_reclaim']),
    'ote': ('data/prepared/eurusd_5min_ote_full', ['long_reversal', 'short_reversal', 'long_breakout', 'short_breakout', 'long_ote', 'short_ote']),
}


class ProgressPipeline(FeaturePreprocessingPipeline):
    def _prepare_target_dataset(self, **kwargs):
        print('Preparing', kwargs['spec'].name, flush=True)
        result = super()._prepare_target_dataset(**kwargs)
        print('Prepared', kwargs['spec'].name, flush=True)
        return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--group', choices=GROUPS, required=True)
    ap.add_argument('--features-root', required=True, type=Path)
    ap.add_argument('--labels', required=True, type=Path)
    ap.add_argument('--events', required=True, type=Path)
    ap.add_argument('--output-root', required=True, type=Path)
    args = ap.parse_args()
    output = args.output_root.resolve()
    if output.exists() or not output.is_relative_to(ROOT / 'artifacts'):
        ap.error('Use a new version under artifacts.')
    output.mkdir(parents=True)
    old_root, targets = GROUPS[args.group]
    old_summary = json.loads((ROOT / old_root / 'summary.json').read_text())
    cfg = PreprocessingConfig(**old_summary['config'])
    cfg.target_columns = ['label_' + t for t in targets]
    source = args.features_root / 'features.csv.gz'
    metadata = json.loads((args.features_root / 'features.metadata.json').read_text())
    labels = pd.read_csv(args.labels).rename(columns={'timestamp': 'datetime'})
    labels.datetime = pd.to_datetime(labels.datetime, utc=True)
    assert labels.datetime.is_unique and labels.datetime.is_monotonic_increasing
    events = pd.read_csv(args.events)
    if args.group == 'ict_specialists':
        from scripts.materialize_ict_setup_prepared_root import _selected_events, _materialize_labels, _embargo_map
        selected = _selected_events(events, 'short')
        selected = selected.loc[('short_' + selected.label_family + '_' + selected.setup_type).isin(targets)]
        labels, _, counts = _materialize_labels(base_dataset=labels[['datetime']], events=selected, direction='short')
        dump(output / 'specialist_counts.json', counts)
        cfg.target_split_embargo_bars = _embargo_map(events=selected, target_columns=cfg.target_columns, feature_metadata=metadata)
    elif args.group == 'ict':
        from ict.pipelines.es_primary_phase06 import _resolve_ict_leakage_control_contract
        contract = _resolve_ict_leakage_control_contract(labels_csv_path=args.labels, phase02_metadata=metadata, target_columns=tuple(cfg.target_columns))
        assert contract['available']
        cfg.target_split_embargo_bars = contract['target_split_embargo_bars']
        dump(output / 'embargo_contract.json', contract)
    else:
        # Preserve chronology while keeping labels whose future outcome
        # windows cross a split boundary away from the next split.
        starts = pd.to_numeric(events.get('swing_index'), errors='coerce')
        confirms = pd.to_numeric(events.get('confirm_index'), errors='coerce')
        held = pd.to_numeric(events.get('tb_bars_held'), errors='coerce').fillna(0)
        entries = pd.to_numeric(events.get('entry_index'), errors='coerce').fillna(confirms)
        exits = pd.concat([entries, confirms], axis=1).max(axis=1) + held
        if starts is None:
            raise ValueError('Cannot verify label-window embargo')
        span = pd.concat([exits - starts, confirms - starts], axis=1).max(axis=1)
        bounds_path = ROOT / 'docs/audits/b2_label_window_bounds_20260921.json'
        bounds = json.loads(bounds_path.read_text())[('ote' if args.group == 'ote' else 'frvp')]
        cfg.split_embargo_bars = max(int(cfg.split_embargo_bars), int(span.max()) + int(metadata['config'].get('swing_window', 3)), bounds['recommended_embargo_bars'])
        cfg.target_split_embargo_bars = {}
        dump(output / 'embargo_contract.json', {'bars': cfg.split_embargo_bars, 'bounds': bounds, 'bounds_sha256': digest(bounds_path)})
    label_columns = [c for c in labels if c == 'warmup_mask' or c.startswith(('label_', 'sample_weight_', 'exclude_', 'neg_ok_', 'concurrency_', 'entry_quality_'))]
    labels = labels.set_index('datetime')
    assert len(label_columns) > 0
    dataset = output / 'dataset.parquet'
    removed_helpers = set()
    print('Merging repaired features and labels', args.group, flush=True)
    rows = 0
    writer = None
    try:
        for chunk in pd.read_csv(source, chunksize=10000):
            stamps = pd.to_datetime(chunk.datetime, utc=True)
            assert stamps.isin(labels.index).all()
            current = labels.loc[stamps, label_columns]
            for c in label_columns:
                chunk[c] = current[c].to_numpy()
            helpers = [c for c in chunk if c.startswith('htf_confluence_')]
            removed_helpers.update(helpers)
            chunk = chunk.drop(columns=helpers)
            chunk['source_row_idx'] = np.arange(rows, rows + len(chunk), dtype=np.int64)
            # The intermediate is columnar to avoid repeatedly formatting a
            # multi-gigabyte text file. Final train/val/test files stay CSV.
            chunk = chunk.loc[:, list(dict.fromkeys(['datetime', 'source_row_idx', *metadata['feature_columns'], *label_columns]))]
            for c in metadata['feature_columns']:
                if pd.api.types.is_numeric_dtype(chunk[c]):
                    chunk[c] = chunk[c].astype(np.float32)
            for c in label_columns:
                if c.startswith(('label_quality_', 'sample_weight_', 'entry_quality_')):
                    chunk[c] = chunk[c].astype(np.float32)
            chunk['datetime'] = stamps.to_numpy()
            table = pa.Table.from_pandas(chunk, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(dataset, table.schema, compression='snappy')
            else:
                table = table.cast(writer.schema)
            writer.write_table(table)
            rows += len(chunk)
            if rows % 100000 == 0:
                print('Merged rows', rows, flush=True)
    finally:
        if writer is not None:
            writer.close()
    metadata['source_path'] = str(source)
    metadata['source_metadata_file'] = str(args.features_root / 'features.metadata.json')
    metadata['feature_columns'] = [c for c in metadata['feature_columns'] if not c.startswith('htf_confluence_')]
    metadata['prepared_contract'] = dict(label_source_path=str(args.labels), event_source_path=str(args.events), target_context_columns=['source_row_idx'])
    metadata['b2_prepared_repair'] = dict(group=args.group, labels_sha256=digest(args.labels), events_sha256=digest(args.events),
        features_sha256=digest(source), recipe_summary_sha256=digest(ROOT / old_root / 'summary.json'),
        removed_helper_predictors=sorted(removed_helpers), trading_model_training_run=False,
        preparation_note='Feature selection is refitted on training rows. No trading model is trained.')
    meta_path = output / 'dataset.metadata.json'
    dump(meta_path, metadata)
    dump(output / 'run_config.json', cfg.to_dict())
    del labels, events
    gc.collect()
    summary = ProgressPipeline(cfg).run(dataset, output / 'prepared', metadata_path=meta_path)
    missing = set(targets) - set(summary['targets'])
    if missing:
        raise ValueError(f'Missing prepared targets: {missing}')
    for target in targets:
        p = output / 'prepared' / target
        if not all((p / f'{split}.csv').exists() for split in ['train', 'val', 'test']):
            raise ValueError(f'Insufficient usable rows: {target}')
        names = json.loads((p / 'features.json').read_text())['features']
        assert not any(n.startswith('htf_confluence_') for n in names)
    dump(output / 'manifest.json', dict(status='prepared_complete', group=args.group, targets=targets, rows=rows,
        dataset_sha256=digest(dataset), metadata_sha256=digest(meta_path), trading_model_training_run=False,
        trusted_model_baseline=False))
    print('Completed', args.group, flush=True)


if __name__ == '__main__':
    main()
