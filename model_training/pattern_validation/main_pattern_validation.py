"""
Pattern Validation System - Complete Integration Guide & Runner
===============================================================

This script demonstrates the complete end-to-end workflow:
1. Generate training data from ICT detector outputs
2. Train the CNN-LSTM validation model
3. Deploy for real-time pattern scoring

Usage:
    python main_pattern_validation.py --mode [generate|train|infer]
"""

import argparse
import os
import sys
import json
from datetime import datetime
import numpy as np
import pandas as pd

# Import modules
from data_generation import (
    DetectedPattern,
    PatternOutcomeLabeler,
    PatternDatasetGenerator
)
from model_architecture import PatternValidationModel
from training_pipeline import PatternValidationTrainer, run_full_training_pipeline  # ← FIXED
from inference_pipeline import PatternValidator, TradingSystemIntegration

class PatternValidationSystem:
    """Complete pattern validation system orchestrator"""
    
    def __init__(self, base_dir: str = './pattern_validation_system'):
        self.base_dir = base_dir
        self.data_dir = os.path.join(base_dir, 'data')
        self.models_dir = os.path.join(base_dir, 'models')
        self.outputs_dir = os.path.join(base_dir, 'outputs')
        self.logs_dir = os.path.join(base_dir, 'logs')
        
        # Create directories
        for d in [self.data_dir, self.models_dir, self.outputs_dir, self.logs_dir]:
            os.makedirs(d, exist_ok=True)
        
        self.config = self._load_or_create_config()
    
    def _load_or_create_config(self) -> dict:
        """Load system configuration"""
        config_path = os.path.join(self.base_dir, 'system_config.json')
        
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                return json.load(f)
        
        # Default configuration
        config = {
            'system_name': 'Pattern Validation System',
            'version': '1.0.0',
            'created': datetime.now().isoformat(),
            
            # Model configuration
            'model': {
                'chart_shape': [100, 5],
                'volume_shape': [100, 1],
                'pattern_features_dim': 13,
                'indicator_features_dim': 15,
                'cnn_filters': [64, 128, 256],
                'lstm_units': [128, 64],
                'dropout_rate': 0.4,
                'l2_reg': 0.001
            },
            
            # Training configuration
            'training': {
                'epochs': 100,
                'batch_size': 32,
                'learning_rate': 0.001,
                'warmup_epochs': 20,
                'warmup_data_pct': 0.2,
                'use_progressive': True,
                'use_cv': False,
                'cv_splits': 5
            },
            
            # Data configuration
            'data': {
                'lookback_candles': 100,
                'train_ratio': 0.7,
                'val_ratio': 0.15,
                'test_ratio': 0.15,
                'stop_buffer_atr': 0.5,
                'target_rr': 2.0,
                'max_bars_forward': 200
            },
            
            # Inference configuration
            'inference': {
                'quality_threshold': 0.75,
                'batch_inference': True,
                'min_quality_score': 0.75
            }
        }
        
        # Save config
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"Configuration saved to {config_path}")
        return config
    
    def generate_training_data(
        self,
        ict_detections_path: str,
        price_data_path: str,
        output_name: str = 'training_data'
    ):
        """
        Step 1: Generate training data from ICT detector outputs
        
        Args:
            ict_detections_path: Path to pickle file with DetectedPattern objects
            price_data_path: Path to CSV with OHLCV data
            output_name: Name for output dataset
        """
        print("\n" + "="*70)
        print("STEP 1: GENERATING TRAINING DATA")
        print("="*70)
        
        # Load ICT detections
        print(f"\nLoading ICT detections from {ict_detections_path}...")
        import pickle
        with open(ict_detections_path, 'rb') as f:
            detected_patterns = pickle.load(f)
        print(f"Loaded {len(detected_patterns)} detected patterns")
        
        # Load price data
        print(f"\nLoading price data from {price_data_path}...")
        price_data = pd.read_csv(price_data_path, index_col=0, parse_dates=True)
        print(f"Loaded {len(price_data)} bars of price data")
        
        # Initialize data generator
        outcome_labeler = PatternOutcomeLabeler(
            stop_buffer_atr=self.config['data']['stop_buffer_atr'],
            target_rr=self.config['data']['target_rr'],
            max_bars_forward=self.config['data']['max_bars_forward']
        )
        
        generator = PatternDatasetGenerator(
            lookback_candles=self.config['data']['lookback_candles'],
            outcome_labeler=outcome_labeler
        )
        
        # Generate dataset
        save_path = os.path.join(self.data_dir, output_name)
        training_data, metadata = generator.generate_training_data(
            detected_patterns=detected_patterns,
            price_data=price_data,
            save_path=save_path
        )
        
        print(f"\n✓ Training data generated and saved to {save_path}")
        
        # Generate report
        self._generate_data_report(metadata, save_path)
        
        return training_data, metadata
    
    def train_model(
        self,
        data_name: str = 'training_data',
        model_name: str = 'pattern_validator'
    ):
        """
        Step 2: Train the pattern validation model
        
        Args:
            data_name: Name of dataset to use
            model_name: Name for trained model
        """
        print("\n" + "="*70)
        print("STEP 2: TRAINING PATTERN VALIDATION MODEL")
        print("="*70)
        
        data_dir = os.path.join(self.data_dir, data_name)
        output_dir = os.path.join(self.outputs_dir, f'{model_name}_training')
        from model_architecture import PatternValidationModel
    
        model_builder = PatternValidationModel(
            chart_shape=(100, 5),
            volume_shape=(100, 1),
            pattern_features_dim=13,
            indicator_features_dim=15,
            cnn_filters=(64, 128, 256),
            lstm_units=(128, 64),
            dropout_rate=0.4
        )
        
        model_builder.build_model()
        model_builder.compile_model(learning_rate=0.001, loss='mse')
        # Run training pipeline
        trainer = run_full_training_pipeline(
            data_dir=data_dir,
            output_dir=output_dir,
            model_builder=model_builder,
            use_progressive=self.config['training']['use_progressive'],
            # use_cv=self.config['training']['use_cv']
        )
        
        # Copy best model to models directory
        import shutil
        best_model_src = os.path.join(output_dir, 'best_model.keras')
        best_model_dst = os.path.join(self.models_dir, f'{model_name}.keras')
        shutil.copy(best_model_src, best_model_dst)
        
        # Copy config
        config_src = os.path.join(output_dir, 'model_config.json')
        config_dst = os.path.join(self.models_dir, f'{model_name}_config.json')
        shutil.copy(config_src, config_dst)
        
        print(f"\n✓ Model trained and saved to {best_model_dst}")
        
        return trainer
    
    def deploy_validator(
        self,
        model_name: str = 'pattern_validator'
    ) -> PatternValidator:
        """
        Step 3: Deploy validator for inference
        
        Args:
            model_name: Name of trained model to deploy
        
        Returns:
            PatternValidator instance ready for inference
        """
        print("\n" + "="*70)
        print("STEP 3: DEPLOYING PATTERN VALIDATOR")
        print("="*70)
        
        model_path = os.path.join(self.models_dir, f'{model_name}.keras')
        config_path = os.path.join(self.models_dir, f'{model_name}_config.json')
        
        validator = PatternValidator(
            model_path=model_path,
            config_path=config_path,
            quality_threshold=self.config['inference']['quality_threshold'],
            batch_inference=self.config['inference']['batch_inference']
        )
        
        print("\n✓ Validator deployed and ready for inference")
        
        # Performance check
        stats = validator.get_performance_stats()
        if stats:
            print(f"\nPerformance stats:")
            print(f"  Average inference time: {stats['avg_inference_time_ms']:.2f} ms")
        
        return validator
    
    def _generate_data_report(self, metadata: pd.DataFrame, save_path: str):
        """Generate data quality report"""
        report_path = os.path.join(save_path, 'data_report.txt')
        
        with open(report_path, 'w') as f:
            f.write("="*70 + "\n")
            f.write("TRAINING DATA QUALITY REPORT\n")
            f.write("="*70 + "\n\n")
            
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Patterns: {len(metadata)}\n\n")
            
            f.write("Pattern Distribution:\n")
            f.write("-" * 40 + "\n")
            f.write(metadata['pattern_type'].value_counts().to_string())
            f.write("\n\n")
            
            f.write("Quality Distribution:\n")
            f.write("-" * 40 + "\n")
            f.write(metadata['quality_class'].value_counts().to_string())
            f.write("\n\n")
            
            f.write("Direction Balance:\n")
            f.write("-" * 40 + "\n")
            f.write(metadata['direction'].value_counts().to_string())
            f.write("\n\n")
            
            f.write("Success Metrics:\n")
            f.write("-" * 40 + "\n")
            f.write(f"Reached 2R: {metadata['reached_2R'].sum()} ({metadata['reached_2R'].mean()*100:.1f}%)\n")
            f.write(f"Hit Stop: {metadata['hit_stop'].sum()} ({metadata['hit_stop'].mean()*100:.1f}%)\n")
            f.write(f"Average MFE: {metadata['mfe_pips'].mean():.2f} pips\n")
            f.write(f"Average MAE: {metadata['mae_pips'].mean():.2f} pips\n")
            f.write("\n")
            
            f.write("Quality Score Statistics:\n")
            f.write("-" * 40 + "\n")
            f.write(metadata['quality_score'].describe().to_string())
            f.write("\n")
        
        print(f"Data report saved to {report_path}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Pattern Validation System - Training & Deployment'
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        required=True,
        choices=['generate', 'train', 'deploy', 'full'],
        help='Operation mode: generate data, train model, deploy, or full pipeline'
    )
    
    parser.add_argument(
        '--base-dir',
        type=str,
        default='./pattern_validation_system',
        help='Base directory for system files'
    )
    
    parser.add_argument(
        '--ict-detections',
        type=str,
        help='Path to ICT detections pickle file (for generate mode)'
    )
    
    parser.add_argument(
        '--price-data',
        type=str,
        help='Path to price data CSV (for generate mode)'
    )
    
    parser.add_argument(
        '--data-name',
        type=str,
        default='training_data',
        help='Name for dataset'
    )
    
    parser.add_argument(
        '--model-name',
        type=str,
        default='pattern_validator',
        help='Name for model'
    )
    
    args = parser.parse_args()
    
    # Initialize system
    system = PatternValidationSystem(base_dir=args.base_dir)
    
    # Execute based on mode
    if args.mode == 'generate':
        if not args.ict_detections or not args.price_data:
            print("Error: --ict-detections and --price-data required for generate mode")
            sys.exit(1)
        
        system.generate_training_data(
            ict_detections_path=args.ict_detections,
            price_data_path=args.price_data,
            output_name=args.data_name
        )
    
    elif args.mode == 'train':
        system.train_model(
            data_name=args.data_name,
            model_name=args.model_name
        )
    
    elif args.mode == 'deploy':
        validator = system.deploy_validator(model_name=args.model_name)
        
        # Run performance benchmark
        print("\nRunning performance benchmark...")
        dummy_patterns = [
            {
                'pattern_id': f'test_{i}',
                'pattern_type': 'FVG',
                'direction': 'bullish',
                'chart_window': np.random.randn(100, 5),
                'volume_profile': np.random.randn(100),
                'pattern_features': {
                    'pattern_size_normalized': 1.0,
                    'volume_ratio': 2.0,
                    'volume_confirmed': True,
                    'session': 'London',
                    'rsi': 60.0,
                    'adx': 30.0,
                    'trend_alignment': True,
                    'distance_to_structure': 0.5
                },
                'indicator_values': np.random.randn(15)
            }
            for i in range(100)
        ]
        
        scores = validator.validate_batch(dummy_patterns)
        stats = validator.get_performance_stats()
        
        print(f"\nBenchmark Results (100 patterns):")
        print(f"  Avg time/pattern: {stats['avg_inference_time_ms']:.2f} ms")
        print(f"  P95 latency: {stats['p95_inference_time_ms']:.2f} ms")
        print(f"  P99 latency: {stats['p99_inference_time_ms']:.2f} ms")
    
    elif args.mode == 'full':
        # Run complete pipeline
        if not args.ict_detections or not args.price_data:
            print("Error: --ict-detections and --price-data required for full mode")
            sys.exit(1)
        
        print("\n" + "="*70)
        print("RUNNING COMPLETE PATTERN VALIDATION PIPELINE")
        print("="*70)
    
        
        # Step 2: Train model
        system.train_model(
            data_name=args.data_name,
            model_name=args.model_name
        )
        
        # Step 3: Deploy
        validator = system.deploy_validator(model_name=args.model_name)
        
        print("\n" + "="*70)
        print("✓ COMPLETE PIPELINE FINISHED SUCCESSFULLY")
        print("="*70)
        print(f"\nValidator ready at: {system.models_dir}/{args.model_name}.keras")
        print(f"Use the validator in your trading system to score patterns in real-time.")
    
    print("\n✓ Operation completed successfully!")


if __name__ == "__main__":
    main()