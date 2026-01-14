#!/usr/bin/env python3
"""
ICT ML Pipeline Runner
======================
Orchestrates the complete ML pipeline for ICT signal filtering.

Pipeline Steps:
1. Feature Engineering: Raw OHLCV → Feature-enriched data
2. Data Preparation: Feature data → Train/val/test splits
3. Model Training: Prepared data → Trained model

Usage:
    python run_pipeline.py data/EURUSD_M1.csv --output-dir output
    
    # Or run individual steps:
    python run_pipeline.py data/EURUSD_M1.csv --step features
    python run_pipeline.py data/features.csv --step prepare
    python run_pipeline.py data/prepared --step train

Author: ICT ML System
Version: 1.0
"""

import argparse
import logging
from pathlib import Path
from datetime import datetime
import json
import sys

# Import pipeline modules
from ict_feature_engineering import FeatureEngineeringPipeline, FeatureConfig
from ict_data_preparation import DataPreparationPipeline, DataPrepConfig
from ict_model_training import ModelTrainingPipeline, TrainingConfig

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_full_pipeline(
    input_file: str,
    output_dir: str = 'output',
    broker_gmt: int = 0,
    model_type: str = 'xgboost',
    min_confluence: int = 1,
    require_killzone: bool = True,
    tune_hyperparams: bool = False
):
    """
    Run the complete ML pipeline from raw OHLCV to trained model.
    
    Args:
        input_file: Path to raw OHLCV CSV
        output_dir: Base output directory
        broker_gmt: Broker GMT offset for time calculations
        model_type: Type of model to train
        min_confluence: Minimum confluence score for signal filtering
        require_killzone: Only use signals in kill zones
        tune_hyperparams: Enable hyperparameter tuning
    """
    logger.info("="*70)
    logger.info("ICT ML PIPELINE - FULL RUN")
    logger.info("="*70)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # ==========================================================================
    # STEP 1: FEATURE ENGINEERING
    # ==========================================================================
    logger.info("\n" + "="*70)
    logger.info("STEP 1: FEATURE ENGINEERING")
    logger.info("="*70)
    
    feature_config = FeatureConfig(
        broker_gmt_offset=broker_gmt,
        swing_strength=5,
        atr_period=14,
        volume_lookback=20
    )
    
    feature_pipeline = FeatureEngineeringPipeline(feature_config)
    
    features_output = output_dir / 'features' / f'features_{timestamp}.csv'
    features_output.parent.mkdir(parents=True, exist_ok=True)
    
    feature_df = feature_pipeline.run(input_file, str(features_output))
    
    logger.info(f"Feature engineering complete: {len(feature_df)} rows, {len(feature_df.columns)} columns")
    
    # ==========================================================================
    # STEP 2: DATA PREPARATION
    # ==========================================================================
    logger.info("\n" + "="*70)
    logger.info("STEP 2: DATA PREPARATION")
    logger.info("="*70)
    
    prep_config = DataPrepConfig(
        target_type='binary',
        target_lookforward=50,
        stop_loss_atr=1.5,
        take_profit_rr=1.5,
        min_confluence=min_confluence,
        require_killzone=require_killzone,
        test_size=0.2,
        validation_size=0.1,
        temporal_split=True,
        scaler_type='robust',
        handle_imbalance=True,
        imbalance_method='smote',
        imbalance_ratio=0.4,
        output_dir=str(output_dir / 'prepared')
    )
    
    prep_pipeline = DataPreparationPipeline(prep_config)
    prep_results = prep_pipeline.run(str(features_output), target_col='target_hit_tp1')
    
    logger.info(f"Data preparation complete:")
    logger.info(f"  Train samples: {prep_results['metadata']['n_samples_train']}")
    logger.info(f"  Test samples: {prep_results['metadata']['n_samples_test']}")
    logger.info(f"  Features: {prep_results['metadata']['n_features']}")
    
    # ==========================================================================
    # STEP 3: MODEL TRAINING
    # ==========================================================================
    logger.info("\n" + "="*70)
    logger.info("STEP 3: MODEL TRAINING")
    logger.info("="*70)
    
    train_config = TrainingConfig(
        model_type=model_type,
        cv_folds=5,
        use_time_series_cv=True,
        tune_hyperparams=tune_hyperparams,
        calibrate_probabilities=True,
        run_profit_simulation=True,
        risk_reward_ratio=1.5,
        output_dir=str(output_dir / 'models'),
        save_model=True,
        save_reports=True,
        plot_results=True
    )
    
    train_pipeline = ModelTrainingPipeline(train_config)
    train_results = train_pipeline.run(str(output_dir / 'prepared'))
    
    # ==========================================================================
    # SUMMARY
    # ==========================================================================
    logger.info("\n" + "="*70)
    logger.info("PIPELINE COMPLETE")
    logger.info("="*70)
    
    # Save pipeline metadata
    pipeline_meta = {
        'timestamp': timestamp,
        'input_file': str(input_file),
        'output_dir': str(output_dir),
        'config': {
            'broker_gmt': broker_gmt,
            'model_type': model_type,
            'min_confluence': min_confluence,
            'require_killzone': require_killzone
        },
        'results': {
            'n_features': prep_results['metadata']['n_features'],
            'n_train_samples': prep_results['metadata']['n_samples_train'],
            'n_test_samples': prep_results['metadata']['n_samples_test'],
            'test_auc': train_results['evaluation']['basic_metrics']['auc'],
            'test_precision': train_results['evaluation']['basic_metrics']['precision'],
            'test_recall': train_results['evaluation']['basic_metrics']['recall']
        }
    }
    
    with open(output_dir / 'pipeline_results.json', 'w') as f:
        json.dump(pipeline_meta, f, indent=2, default=str)
    
    logger.info(f"\nAll outputs saved to: {output_dir}")
    logger.info(f"  Features: {output_dir / 'features'}")
    logger.info(f"  Prepared data: {output_dir / 'prepared'}")
    logger.info(f"  Models: {output_dir / 'models'}")
    
    return train_results


def run_feature_engineering(input_file: str, output_file: str, broker_gmt: int = 0):
    """Run only the feature engineering step"""
    
    logger.info("Running feature engineering...")
    
    config = FeatureConfig(broker_gmt_offset=broker_gmt)
    pipeline = FeatureEngineeringPipeline(config)
    
    return pipeline.run(input_file, output_file)


def run_data_preparation(
    input_file: str, 
    output_dir: str,
    min_confluence: int = 1,
    require_killzone: bool = True
):
    """Run only the data preparation step"""
    
    logger.info("Running data preparation...")
    
    config = DataPrepConfig(
        min_confluence=min_confluence,
        require_killzone=require_killzone,
        output_dir=output_dir
    )
    
    pipeline = DataPreparationPipeline(config)
    return pipeline.run(input_file)


def run_training(data_dir: str, output_dir: str, model_type: str = 'xgboost'):
    """Run only the model training step"""
    
    logger.info("Running model training...")
    
    config = TrainingConfig(
        model_type=model_type,
        output_dir=output_dir
    )
    
    pipeline = ModelTrainingPipeline(config)
    return pipeline.run(data_dir)


def main():
    parser = argparse.ArgumentParser(
        description='ICT ML Pipeline Runner',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full pipeline
  python run_pipeline.py data/EURUSD_M1.csv --output-dir output
  
  # Run only feature engineering
  python run_pipeline.py data/EURUSD_M1.csv --step features --output-dir output/features.csv
  
  # Run only data preparation
  python run_pipeline.py output/features.csv --step prepare --output-dir output/prepared
  
  # Run only training
  python run_pipeline.py output/prepared --step train --output-dir output/models
        """
    )
    
    parser.add_argument('input', help='Input file or directory')
    parser.add_argument('--output-dir', '-o', default='output',
                        help='Output directory (default: output)')
    parser.add_argument('--step', choices=['all', 'features', 'prepare', 'train'],
                        default='all', help='Pipeline step to run (default: all)')
    parser.add_argument('--broker-gmt', type=int, default=0,
                        help='Broker GMT offset (default: 0)')
    parser.add_argument('--model', default='xgboost',
                        choices=['xgboost', 'lightgbm', 'random_forest', 'ensemble'],
                        help='Model type (default: xgboost)')
    parser.add_argument('--min-confluence', type=int, default=1,
                        help='Minimum confluence score (default: 1)')
    parser.add_argument('--no-killzone-filter', action='store_true',
                        help='Disable killzone filter')
    parser.add_argument('--tune', action='store_true',
                        help='Enable hyperparameter tuning')
    
    args = parser.parse_args()
    
    try:
        if args.step == 'all':
            run_full_pipeline(
                args.input,
                args.output_dir,
                broker_gmt=args.broker_gmt,
                model_type=args.model,
                min_confluence=args.min_confluence,
                require_killzone=not args.no_killzone_filter,
                tune_hyperparams=args.tune
            )
        
        elif args.step == 'features':
            run_feature_engineering(
                args.input,
                args.output_dir,
                broker_gmt=args.broker_gmt
            )
        
        elif args.step == 'prepare':
            run_data_preparation(
                args.input,
                args.output_dir,
                min_confluence=args.min_confluence,
                require_killzone=not args.no_killzone_filter
            )
        
        elif args.step == 'train':
            run_training(
                args.input,
                args.output_dir,
                model_type=args.model
            )
        
        logger.info("\nPipeline completed successfully!")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise


if __name__ == "__main__":
    main()
