"""
Multi-Timeframe Confluence Model - Real Data Training Script
============================================================

This script trains the confluence model on your actual EURUSD data and generates
comprehensive reports, visualizations, and analysis.

Usage:
    python train_confluence_real_data.py
"""

import numpy as np
import pandas as pd
import os
import pickle
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import our modules
from mtf_confluence_trainer import (
    MultiTimeframeDataPreparator,
    MultiTimeframeConfluenceModel,
    ConfluenceModelTrainer
)
from mtf_confluence_analyzer import (
    ConfluenceAnalyzer as ConfluenceAnalyzer,
    format_analysis_report
)
from mtf_confluence_visualizations import save_all_visualizations

# Import sklearn metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


class RealDataTrainingPipeline:
    """Complete training pipeline for real EURUSD data"""
    
    def __init__(self, data_path: str, output_dir: str = './mtf_confluence_output/'):
        """
        Initialize training pipeline
        
        Args:
            data_path: Path to your EURUSD CSV file
            output_dir: Where to save all outputs
        """
        self.data_path = data_path
        self.output_dir = output_dir
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'models'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'plots'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'reports'), exist_ok=True)
        
        print(f"Output directory: {self.output_dir}")
        
        self.df = None
        self.preparator = None
        self.model = None
        self.trainer = None
        self.history = None
        
    def load_and_prepare_data(self):
        """Load and prepare your CSV data"""
        
        print("\n" + "="*80)
        print("STEP 1: Loading Data")
        print("="*80)
        
        print(f"\nLoading data from: {self.data_path}")
        
        # Load CSV
        self.df = pd.read_csv(self.data_path)
        
        print(f"✓ Loaded {len(self.df):,} rows")
        print(f"✓ Columns: {len(self.df.columns)}")
        
        # Check for datetime column
        if 'datetime' in self.df.columns:
            self.df['datetime'] = pd.to_datetime(self.df['datetime'])
            self.df = self.df.set_index('datetime')
            print("✓ Set 'datetime' as index")
        elif 'Date' in self.df.columns and 'Time' in self.df.columns:
            self.df['datetime'] = pd.to_datetime(self.df['Date'] + ' ' + self.df['Time'])
            self.df = self.df.set_index('datetime')
            self.df = self.df.drop(['Date', 'Time'], axis=1)
            print("✓ Combined Date + Time into datetime index")
        else:
            print("⚠ Warning: No datetime column found, using row index")
        
        # Verify required OHLCV columns
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_cols = [col for col in required_cols if col not in self.df.columns]
        
        if missing_cols:
            # Try lowercase
            required_cols_lower = ['open', 'high', 'low', 'close', 'volume']
            if all(col in self.df.columns for col in required_cols_lower):
                print("✓ Found OHLCV columns (lowercase)")
            else:
                raise ValueError(f"Missing required columns: {missing_cols}")
        else:
            # Rename to lowercase for consistency
            self.df = self.df.rename(columns={
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume'
            })
            print("✓ Renamed columns to lowercase")
        
        # Keep only OHLCV for confluence model
        self.df = self.df[['open', 'high', 'low', 'close', 'volume']]
        
        # Remove any NaN values
        initial_len = len(self.df)
        self.df = self.df.dropna()
        if len(self.df) < initial_len:
            print(f"✓ Removed {initial_len - len(self.df)} rows with NaN values")
        
        # Sort by datetime
        self.df = self.df.sort_index()
        
        print(f"\nData Summary:")
        print(f"  Date range: {self.df.index[0]} to {self.df.index[-1]}")
        print(f"  Total bars: {len(self.df):,}")
        print(f"  Duration: {(self.df.index[-1] - self.df.index[0]).days} days")
        print(f"\n  Price range: {self.df['close'].min():.5f} - {self.df['close'].max():.5f}")
        print(f"  Avg volume: {self.df['volume'].mean():.0f}")
        
        # Save data summary
        summary = {
            'data_path': self.data_path,
            'total_rows': len(self.df),
            'date_range': f"{self.df.index[0]} to {self.df.index[-1]}",
            'duration_days': (self.df.index[-1] - self.df.index[0]).days,
            'price_range': f"{self.df['close'].min():.5f} - {self.df['close'].max():.5f}",
            'avg_volume': float(self.df['volume'].mean())
        }
        
        with open(os.path.join(self.output_dir, 'reports', 'data_summary.txt'), 'w') as f:
            f.write("DATA SUMMARY\n")
            f.write("="*80 + "\n\n")
            for key, value in summary.items():
                f.write(f"{key}: {value}\n")
        
        return self.df
    
    def prepare_training_data(self, validation_split: float = 0.2):
        """Prepare multi-timeframe training data"""
        
        print("\n" + "="*80)
        print("STEP 2: Preparing Multi-Timeframe Data")
        print("="*80)
        
        self.preparator = MultiTimeframeDataPreparator()
        
        print("\nThis will create features for all timeframes...")
        print("Please wait, this may take a few minutes...")
        
        self.X_train, self.X_val, self.y_train_conf, self.y_train_bias, \
        self.y_val_conf, self.y_val_bias = self.preparator.prepare_training_data(
            self.df, 
            validation_split=validation_split
        )
        
        print(f"\n✓ Training samples: {len(self.y_train_conf):,}")
        print(f"✓ Validation samples: {len(self.y_val_conf):,}")
        
        print(f"\nInput shapes per timeframe:")
        for tf in sorted(self.X_train.keys()):
            print(f"  {tf}: {self.X_train[tf].shape}")
        
        # Save data preparation info
        prep_info = {
            'validation_split': validation_split,
            'train_samples': len(self.y_train_conf),
            'val_samples': len(self.y_val_conf),
            'timeframes': list(self.X_train.keys()),
            'features_per_timeframe': self.X_train['1m'].shape[-1],
            'lookback_periods': self.preparator.lookback_periods
        }
        
        with open(os.path.join(self.output_dir, 'reports', 'preparation_info.txt'), 'w') as f:
            f.write("DATA PREPARATION INFO\n")
            f.write("="*80 + "\n\n")
            for key, value in prep_info.items():
                f.write(f"{key}: {value}\n")
        
        return self.X_train, self.X_val
    
    def build_and_compile_model(self, 
                                d_model: int = 128,
                                num_heads: int = 4,
                                ff_dim: int = 256,
                                num_transformer_blocks: int = 2,
                                dropout_rate: float = 0.2,
                                learning_rate: float = 0.001):
        """Build and compile the model"""
        
        print("\n" + "="*80)
        print("STEP 3: Building Model")
        print("="*80)
        
        input_shapes = {tf: self.X_train[tf].shape[1:] for tf in self.X_train.keys()}
        
        self.model = MultiTimeframeConfluenceModel(
            input_shapes=input_shapes,
            d_model=d_model,
            num_heads=num_heads,
            ff_dim=ff_dim,
            num_transformer_blocks=num_transformer_blocks,
            dropout_rate=dropout_rate
        )
        
        self.model.build_model()
        print("✓ Model built")
        
        self.model.compile_model(learning_rate=learning_rate)
        print("✓ Model compiled")
        
        # Print model summary
        print("\nModel Architecture:")
        # self.model.get_model_summary()
        
        total_params = self.model.model.count_params()
        print(f"\n✓ Total parameters: {total_params:,}")
        
        # Save model config
        model_config = {
            'd_model': d_model,
            'num_heads': num_heads,
            'ff_dim': ff_dim,
            'num_transformer_blocks': num_transformer_blocks,
            'dropout_rate': dropout_rate,
            'learning_rate': learning_rate,
            'total_parameters': int(total_params)
        }
        
        with open(os.path.join(self.output_dir, 'reports', 'model_config.txt'), 'w') as f:
            f.write("MODEL CONFIGURATION\n")
            f.write("="*80 + "\n\n")
            for key, value in model_config.items():
                f.write(f"{key}: {value}\n")
        
        return self.model
    
    def train_model(self, 
                   epochs: int = 100,
                   batch_size: int = 64,
                   patience: int = 15):
        """Train the model"""
        
        print("\n" + "="*80)
        print("STEP 4: Training Model")
        print("="*80)
        
        model_path = os.path.join(self.output_dir, 'models', 'confluence_model_best.keras')
        
        self.trainer = ConfluenceModelTrainer(
            model=self.model,
            model_save_path=model_path
        )
        
        print(f"\nTraining for up to {epochs} epochs...")
        print(f"Batch size: {batch_size}")
        print(f"Early stopping patience: {patience}")
        print("\nThis may take 30-60 minutes depending on your hardware...")
        
        start_time = datetime.now()
        
        self.history = self.trainer.train(
            X_train=self.X_train,
            y_train_confluence=self.y_train_conf,
            y_train_bias=self.y_train_bias,
            X_val=self.X_val,
            y_val_confluence=self.y_val_conf,
            y_val_bias=self.y_val_bias,
            epochs=epochs,
            batch_size=batch_size,
            patience=patience
        )
        
        training_time = (datetime.now() - start_time).total_seconds()
        
        print(f"\n✓ Training completed in {training_time/60:.1f} minutes")
        print(f"✓ Best model saved to: {model_path}")
        
        # Save training history
        history_df = pd.DataFrame(self.history.history)
        history_df.to_csv(os.path.join(self.output_dir, 'reports', 'training_history.csv'), index=False)
        print(f"✓ Training history saved")
        
        # Save scalers
        scaler_path = os.path.join(self.output_dir, 'models', 'confluence_scalers.pkl')
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.preparator.scalers, f)
        print(f"✓ Scalers saved to: {scaler_path}")
        
        return self.history
    
    def evaluate_and_analyze(self):
        """Comprehensive evaluation and analysis"""
        
        print("\n" + "="*80)
        print("STEP 5: Evaluation & Analysis")
        print("="*80)
        
        # Make predictions on validation set
        X_val_list = [self.X_val[tf] for tf in sorted(self.X_val.keys())]
        predictions = self.model.model.predict(X_val_list, verbose=0)
        
        self.y_pred_conf = predictions[0].flatten()
        self.y_pred_bias = predictions[1].flatten()
        
        # Calculate metrics
        metrics = self.calculate_metrics()
        
        # Print metrics
        self.print_metrics(metrics)
        
        # Save metrics report
        self.save_metrics_report(metrics)
        
        # Generate visualizations
        self.generate_visualizations()
        
        # Generate detailed analysis report
        self.generate_analysis_report(metrics)
        
        return metrics
    
    def calculate_metrics(self):
        """Calculate all performance metrics"""
        
        metrics = {}
        
        # Confluence metrics
        metrics['confluence_mae'] = mean_absolute_error(self.y_val_conf, self.y_pred_conf)
        metrics['confluence_rmse'] = np.sqrt(mean_squared_error(self.y_val_conf, self.y_pred_conf))
        metrics['confluence_r2'] = r2_score(self.y_val_conf, self.y_pred_conf)
        
        # Bias metrics
        metrics['bias_mae'] = mean_absolute_error(self.y_val_bias, self.y_pred_bias)
        metrics['bias_rmse'] = np.sqrt(mean_squared_error(self.y_val_bias, self.y_pred_bias))
        metrics['bias_r2'] = r2_score(self.y_val_bias, self.y_pred_bias)
        
        # Directional accuracy
        correct_direction = (np.sign(self.y_val_bias) == np.sign(self.y_pred_bias))
        metrics['directional_accuracy'] = correct_direction.mean() * 100
        
        # Confluence bucket accuracy
        for threshold in [0.60, 0.70, 0.75, 0.80, 0.85]:
            high_conf = self.y_val_conf >= threshold
            if high_conf.sum() > 0:
                metrics[f'dir_acc_conf_{int(threshold*100)}'] = correct_direction[high_conf].mean() * 100
        
        # Trading signal statistics
        high_conf = self.y_pred_conf >= 0.70
        strong_bias = np.abs(self.y_pred_bias) >= 0.30
        tradeable = high_conf & strong_bias
        
        metrics['pct_high_confluence'] = high_conf.mean() * 100
        metrics['pct_strong_bias'] = strong_bias.mean() * 100
        metrics['pct_tradeable_signals'] = tradeable.mean() * 100
        
        # Signal quality distribution
        excellent = ((self.y_pred_conf >= 0.85) & (np.abs(self.y_pred_bias) >= 0.50)).sum()
        good = ((self.y_pred_conf >= 0.70) & (np.abs(self.y_pred_bias) >= 0.30)).sum()
        fair = ((self.y_pred_conf >= 0.60) & (np.abs(self.y_pred_bias) >= 0.20)).sum()
        
        metrics['excellent_signals'] = excellent
        metrics['good_signals'] = good
        metrics['fair_signals'] = fair
        
        # Training info
        metrics['total_epochs'] = len(self.history.history['loss'])
        metrics['best_epoch'] = np.argmin(self.history.history['val_loss']) + 1
        metrics['final_train_loss'] = self.history.history['loss'][-1]
        metrics['final_val_loss'] = self.history.history['val_loss'][-1]
        metrics['best_val_loss'] = min(self.history.history['val_loss'])
        
        return metrics
    
    def print_metrics(self, metrics):
        """Print metrics to console"""
        
        print("\n" + "="*80)
        print("PERFORMANCE METRICS")
        print("="*80)
        
        print("\n📊 CONFLUENCE SCORE:")
        print(f"  MAE:  {metrics['confluence_mae']:.4f} (target: <0.15) {'✓' if metrics['confluence_mae'] < 0.15 else '✗'}")
        print(f"  RMSE: {metrics['confluence_rmse']:.4f} (target: <0.20) {'✓' if metrics['confluence_rmse'] < 0.20 else '✗'}")
        print(f"  R²:   {metrics['confluence_r2']:.4f} (target: >0.40) {'✓' if metrics['confluence_r2'] > 0.40 else '✗'}")
        
        print("\n📊 DIRECTIONAL BIAS:")
        print(f"  MAE:  {metrics['bias_mae']:.4f} (target: <0.20) {'✓' if metrics['bias_mae'] < 0.20 else '✗'}")
        print(f"  RMSE: {metrics['bias_rmse']:.4f} (target: <0.30) {'✓' if metrics['bias_rmse'] < 0.30 else '✗'}")
        print(f"  R²:   {metrics['bias_r2']:.4f} (target: >0.40) {'✓' if metrics['bias_r2'] > 0.40 else '✗'}")
        
        print("\n🎯 DIRECTIONAL ACCURACY:")
        print(f"  Overall: {metrics['directional_accuracy']:.2f}% (target: >65%) {'✓' if metrics['directional_accuracy'] > 65 else '✗'}")
        
        print("\n  By Confluence Level:")
        for threshold in [60, 70, 75, 80, 85]:
            key = f'dir_acc_conf_{threshold}'
            if key in metrics:
                print(f"    ≥{threshold/100:.2f}: {metrics[key]:.2f}%")
        
        print("\n📈 TRADING SIGNALS:")
        print(f"  High Confluence (≥0.70): {metrics['pct_high_confluence']:.1f}%")
        print(f"  Strong Bias (|bias|≥0.30): {metrics['pct_strong_bias']:.1f}%")
        print(f"  Tradeable Signals: {metrics['pct_tradeable_signals']:.1f}%")
        
        print("\n⭐ SIGNAL QUALITY:")
        print(f"  EXCELLENT (≥0.85 conf, |bias|≥0.50): {metrics['excellent_signals']:,}")
        print(f"  GOOD (≥0.70 conf, |bias|≥0.30): {metrics['good_signals']:,}")
        print(f"  FAIR (≥0.60 conf, |bias|≥0.20): {metrics['fair_signals']:,}")
        
        print("\n🏋️ TRAINING INFO:")
        print(f"  Total Epochs: {metrics['total_epochs']}")
        print(f"  Best Epoch: {metrics['best_epoch']}")
        print(f"  Best Val Loss: {metrics['best_val_loss']:.4f}")
        print(f"  Final Val Loss: {metrics['final_val_loss']:.4f}")
        
        # Overall assessment
        print("\n" + "="*80)
        print("OVERALL ASSESSMENT")
        print("="*80)
        
        targets_met = 0
        total_targets = 4
        
        if metrics['confluence_mae'] < 0.15:
            targets_met += 1
        if metrics['bias_mae'] < 0.20:
            targets_met += 1
        if metrics['directional_accuracy'] > 65:
            targets_met += 1
        if metrics['confluence_r2'] > 0.40:
            targets_met += 1
        
        print(f"\n✓ Targets Met: {targets_met}/{total_targets}")
        
        if targets_met == total_targets:
            print("\n🎉 EXCELLENT! All targets met. Model is production-ready.")
        elif targets_met >= 3:
            print("\n✓ GOOD! Most targets met. Model is usable with monitoring.")
        elif targets_met >= 2:
            print("\n⚠ FAIR. Consider more training or data.")
        else:
            print("\n⚠ WARNING. Model needs improvement before production use.")
    
    def save_metrics_report(self, metrics):
        """Save metrics to text file"""
        
        report_path = os.path.join(self.output_dir, 'reports', 'performance_metrics.txt')
        
        with open(report_path, 'w') as f:
            f.write("MULTI-TIMEFRAME CONFLUENCE MODEL - PERFORMANCE REPORT\n")
            f.write("="*80 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*80 + "\n\n")
            
            f.write("CONFLUENCE SCORE METRICS\n")
            f.write("-"*80 + "\n")
            f.write(f"MAE:  {metrics['confluence_mae']:.6f} (target: <0.15)\n")
            f.write(f"RMSE: {metrics['confluence_rmse']:.6f} (target: <0.20)\n")
            f.write(f"R²:   {metrics['confluence_r2']:.6f} (target: >0.40)\n\n")
            
            f.write("DIRECTIONAL BIAS METRICS\n")
            f.write("-"*80 + "\n")
            f.write(f"MAE:  {metrics['bias_mae']:.6f} (target: <0.20)\n")
            f.write(f"RMSE: {metrics['bias_rmse']:.6f} (target: <0.30)\n")
            f.write(f"R²:   {metrics['bias_r2']:.6f} (target: >0.40)\n\n")
            
            f.write("DIRECTIONAL ACCURACY\n")
            f.write("-"*80 + "\n")
            f.write(f"Overall: {metrics['directional_accuracy']:.2f}% (target: >65%)\n\n")
            f.write("By Confluence Level:\n")
            for threshold in [60, 70, 75, 80, 85]:
                key = f'dir_acc_conf_{threshold}'
                if key in metrics:
                    f.write(f"  ≥{threshold/100:.2f}: {metrics[key]:.2f}%\n")
            
            f.write("\nTRADING SIGNAL STATISTICS\n")
            f.write("-"*80 + "\n")
            f.write(f"High Confluence (≥0.70): {metrics['pct_high_confluence']:.2f}%\n")
            f.write(f"Strong Bias (|bias|≥0.30): {metrics['pct_strong_bias']:.2f}%\n")
            f.write(f"Tradeable Signals: {metrics['pct_tradeable_signals']:.2f}%\n\n")
            
            f.write("SIGNAL QUALITY DISTRIBUTION\n")
            f.write("-"*80 + "\n")
            f.write(f"EXCELLENT: {metrics['excellent_signals']:,}\n")
            f.write(f"GOOD: {metrics['good_signals']:,}\n")
            f.write(f"FAIR: {metrics['fair_signals']:,}\n\n")
            
            f.write("TRAINING INFORMATION\n")
            f.write("-"*80 + "\n")
            f.write(f"Total Epochs: {metrics['total_epochs']}\n")
            f.write(f"Best Epoch: {metrics['best_epoch']}\n")
            f.write(f"Best Val Loss: {metrics['best_val_loss']:.6f}\n")
            f.write(f"Final Train Loss: {metrics['final_train_loss']:.6f}\n")
            f.write(f"Final Val Loss: {metrics['final_val_loss']:.6f}\n")
        
        print(f"\n✓ Metrics report saved to: {report_path}")
    
    def generate_visualizations(self):
        """Generate all plots"""
        
        print("\n" + "="*80)
        print("STEP 6: Generating Visualizations")
        print("="*80)
        
        plot_dir = os.path.join(self.output_dir, 'plots')
        
        # Import matplotlib here to avoid display issues
        import matplotlib.pyplot as plt
        from mtf_confluence_visualizations import (
            plot_training_history,
            plot_prediction_analysis,
            plot_confluence_distribution,
            create_performance_dashboard
        )
        
        # Training history
        print("\nGenerating training history plot...")
        fig1 = plot_training_history(self.history)
        fig1.savefig(os.path.join(plot_dir, '1_training_history.png'), dpi=150, bbox_inches='tight')
        plt.close(fig1)
        print("✓ Saved: 1_training_history.png")
        
        # Prediction analysis
        print("Generating prediction analysis plot...")
        fig2 = plot_prediction_analysis(
            self.y_val_conf, self.y_pred_conf,
            self.y_val_bias, self.y_pred_bias
        )
        fig2.savefig(os.path.join(plot_dir, '2_prediction_analysis.png'), dpi=150, bbox_inches='tight')
        plt.close(fig2)
        print("✓ Saved: 2_prediction_analysis.png")
        
        # Confluence distribution
        print("Generating confluence distribution plot...")
        fig3 = plot_confluence_distribution(self.y_pred_conf, self.y_pred_bias)
        fig3.savefig(os.path.join(plot_dir, '3_confluence_distribution.png'), dpi=150, bbox_inches='tight')
        plt.close(fig3)
        print("✓ Saved: 3_confluence_distribution.png")
        
        # Performance dashboard
        print("Generating performance dashboard...")
        fig4 = create_performance_dashboard(
            self.history,
            self.y_val_conf, self.y_pred_conf,
            self.y_val_bias, self.y_pred_bias
        )
        fig4.savefig(os.path.join(plot_dir, '4_performance_dashboard.png'), dpi=150, bbox_inches='tight')
        plt.close(fig4)
        print("✓ Saved: 4_performance_dashboard.png")
        
        print(f"\n✓ All plots saved to: {plot_dir}")
    
    def generate_analysis_report(self, metrics):
        """Generate comprehensive analysis report"""
        
        print("\n" + "="*80)
        print("STEP 7: Generating Final Report")
        print("="*80)
        
        report_path = os.path.join(self.output_dir, 'reports', 'FINAL_REPORT.txt')
        
        with open(report_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("MULTI-TIMEFRAME CONFLUENCE MODEL - FINAL TRAINING REPORT\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Data File: {self.data_path}\n")
            f.write(f"Output Directory: {self.output_dir}\n\n")
            
            f.write("="*80 + "\n")
            f.write("1. DATA SUMMARY\n")
            f.write("="*80 + "\n\n")
            f.write(f"Total Samples: {len(self.df):,}\n")
            f.write(f"Date Range: {self.df.index[0]} to {self.df.index[-1]}\n")
            f.write(f"Duration: {(self.df.index[-1] - self.df.index[0]).days} days\n\n")
            f.write(f"Training Samples: {len(self.y_train_conf):,}\n")
            f.write(f"Validation Samples: {len(self.y_val_conf):,}\n\n")
            
            f.write("="*80 + "\n")
            f.write("2. MODEL ARCHITECTURE\n")
            f.write("="*80 + "\n\n")
            f.write(f"Type: Multi-Stream Transformer\n")
            f.write(f"Timeframes: {list(self.X_train.keys())}\n")
            f.write(f"Features per Timeframe: {self.X_train['1m'].shape[-1]}\n")
            f.write(f"Total Parameters: {self.model.model.count_params():,}\n\n")
            
            f.write("="*80 + "\n")
            f.write("3. TRAINING RESULTS\n")
            f.write("="*80 + "\n\n")
            f.write(f"Epochs Trained: {metrics['total_epochs']}\n")
            f.write(f"Best Epoch: {metrics['best_epoch']}\n")
            f.write(f"Best Validation Loss: {metrics['best_val_loss']:.6f}\n\n")
            
            f.write("="*80 + "\n")
            f.write("4. PERFORMANCE METRICS\n")
            f.write("="*80 + "\n\n")
            
            # Targets
            targets = [
                ('Confluence MAE', metrics['confluence_mae'], 0.15, '<'),
                ('Bias MAE', metrics['bias_mae'], 0.20, '<'),
                ('Directional Accuracy', metrics['directional_accuracy'], 65.0, '>'),
                ('Confluence R²', metrics['confluence_r2'], 0.40, '>')
            ]
            
            for name, value, target, comp in targets:
                if comp == '<':
                    met = value < target
                else:
                    met = value > target
                
                status = "✓ MET" if met else "✗ NOT MET"
                f.write(f"{name}: {value:.4f} (target: {comp}{target}) {status}\n")
            
            f.write("\n")
            f.write(f"Directional Accuracy by Confluence:\n")
            for threshold in [60, 70, 75, 80, 85]:
                key = f'dir_acc_conf_{threshold}'
                if key in metrics:
                    f.write(f"  ≥{threshold/100:.2f}: {metrics[key]:.2f}%\n")
            
            f.write("\n")
            f.write("="*80 + "\n")
            f.write("5. TRADING SIGNAL ANALYSIS\n")
            f.write("="*80 + "\n\n")
            f.write(f"High Confluence Signals (≥0.70): {metrics['pct_high_confluence']:.2f}%\n")
            f.write(f"Tradeable Signals: {metrics['pct_tradeable_signals']:.2f}%\n\n")
            
            f.write("Signal Quality Distribution:\n")
            f.write(f"  EXCELLENT (≥0.85 conf, |bias|≥0.50): {metrics['excellent_signals']:,}\n")
            f.write(f"  GOOD (≥0.70 conf, |bias|≥0.30): {metrics['good_signals']:,}\n")
            f.write(f"  FAIR (≥0.60 conf, |bias|≥0.20): {metrics['fair_signals']:,}\n\n")
            
            f.write("="*80 + "\n")
            f.write("6. FILES GENERATED\n")
            f.write("="*80 + "\n\n")
            f.write("Models:\n")
            f.write("  - confluence_model_best.keras (trained model)\n")
            f.write("  - confluence_scalers.pkl (feature scalers)\n\n")
            f.write("Reports:\n")
            f.write("  - performance_metrics.txt (detailed metrics)\n")
            f.write("  - training_history.csv (epoch-by-epoch history)\n")
            f.write("  - FINAL_REPORT.txt (this file)\n\n")
            f.write("Plots:\n")
            f.write("  - 1_training_history.png\n")
            f.write("  - 2_prediction_analysis.png\n")
            f.write("  - 3_confluence_distribution.png\n")
            f.write("  - 4_performance_dashboard.png\n\n")
            
            f.write("="*80 + "\n")
            f.write("7. NEXT STEPS\n")
            f.write("="*80 + "\n\n")
            f.write("1. Review all plots in the 'plots' directory\n")
            f.write("2. Test the model with real-time inference\n")
            f.write("3. Integrate with your trading system\n")
            f.write("4. Paper trade for 2+ weeks\n")
            f.write("5. Monitor performance metrics\n")
            f.write("6. Retrain quarterly or if performance degrades\n\n")
            
            f.write("="*80 + "\n")
            f.write("8. USAGE EXAMPLE\n")
            f.write("="*80 + "\n\n")
            f.write("from mtf_confluence_analyzer import ConfluenceAnalyzer\n\n")
            f.write("analyzer = ConfluenceAnalyzer(\n")
            f.write(f"    model_path='{os.path.join(self.output_dir, 'models', 'confluence_model_best.keras')}',\n")
            f.write(f"    scaler_path='{os.path.join(self.output_dir, 'models', 'confluence_scalers.pkl')}'\n")
            f.write(")\n\n")
            f.write("score, bias, details = analyzer.analyze(current_data)\n\n")
            f.write("if analyzer.should_trade(score, bias, min_confluence=0.70):\n")
            f.write("    if bias > 0.3:\n")
            f.write("        print('LONG SIGNAL')\n")
            f.write("    elif bias < -0.3:\n")
            f.write("        print('SHORT SIGNAL')\n\n")
            
            f.write("="*80 + "\n")
            f.write("END OF REPORT\n")
            f.write("="*80 + "\n")
        
        print(f"\n✓ Final report saved to: {report_path}")
    
    def run_complete_pipeline(self,
                             validation_split: float = 0.2,
                             d_model: int = 128,
                             num_heads: int = 4,
                             ff_dim: int = 256,
                             num_transformer_blocks: int = 2,
                             dropout_rate: float = 0.2,
                             learning_rate: float = 0.001,
                             epochs: int = 100,
                             batch_size: int = 64,
                             patience: int = 15):
        """Run the complete training pipeline"""
        
        print("\n" + "="*80)
        print("MULTI-TIMEFRAME CONFLUENCE MODEL - COMPLETE TRAINING PIPELINE")
        print("="*80)
        print(f"\nStarted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Data: {self.data_path}")
        print(f"Output: {self.output_dir}")
        
        try:
            # Step 1: Load data
            self.load_and_prepare_data()
            
            # Step 2: Prepare training data
            self.prepare_training_data(validation_split=validation_split)
            
            # Step 3: Build model
            self.build_and_compile_model(
                d_model=d_model,
                num_heads=num_heads,
                ff_dim=ff_dim,
                num_transformer_blocks=num_transformer_blocks,
                dropout_rate=dropout_rate,
                learning_rate=learning_rate
            )
            
            # Step 4: Train
            self.train_model(
                epochs=epochs,
                batch_size=batch_size,
                patience=patience
            )
            
            # Step 5-7: Evaluate and generate reports
            metrics = self.evaluate_and_analyze()
            
            print("\n" + "="*80)
            print("TRAINING COMPLETE!")
            print("="*80)
            print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"\nAll outputs saved to: {self.output_dir}")
            print("\nNext steps:")
            print("  1. Review plots in plots/ directory")
            print("  2. Check FINAL_REPORT.txt")
            print("  3. Test with inference script")
            print("  4. Integrate with trading system")
            
            return True
            
        except Exception as e:
            print(f"\n❌ ERROR: {str(e)}")
            import traceback
            traceback.print_exc()
            return False


def main():
    """Main entry point"""
    
    # CONFIGURATION
    # =============
    
    # Your data file path (UPDATE THIS!)
    DATA_PATH = r"C:\Users\zebfr\Documents\All_Files\TRADING\trade_bot\data\currency_data\sampled\EURUSD_1min_sampled_features.csv"
    
    # Output directory
    OUTPUT_DIR = "./mtf_confluence_output/"
    
    # Training parameters
    VALIDATION_SPLIT = 0.2  # 20% for validation
    
    # Model architecture
    D_MODEL = 128           # Transformer dimension
    NUM_HEADS = 4           # Attention heads
    FF_DIM = 256            # Feed-forward dimension
    NUM_BLOCKS = 2          # Transformer blocks
    DROPOUT = 0.2           # Dropout rate
    
    # Training
    LEARNING_RATE = 0.001
    EPOCHS = 100            # Maximum epochs
    BATCH_SIZE = 64
    PATIENCE = 15           # Early stopping patience
    
    # Initialize pipeline
    pipeline = RealDataTrainingPipeline(
        data_path=DATA_PATH,
        output_dir=OUTPUT_DIR
    )
    
    # Run complete training
    success = pipeline.run_complete_pipeline(
        validation_split=VALIDATION_SPLIT,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        ff_dim=FF_DIM,
        num_transformer_blocks=NUM_BLOCKS,
        dropout_rate=DROPOUT,
        learning_rate=LEARNING_RATE,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        patience=PATIENCE
    )
    
    if success:
        print("\n✅ Training pipeline completed successfully!")
        return 0
    else:
        print("\n❌ Training pipeline failed.")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())