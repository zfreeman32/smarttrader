"""
Pattern Validation Model Training Pipeline - FIXED
===================================================
Fixed the learning rate setting issue in progressive training.
"""

from datetime import datetime
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import numpy as np
from typing import Tuple, Dict, Optional
import json

# Assuming you have this import from your model architecture file
# from model_architecture import PatternValidationModel


class PatternValidationTrainer:
    """Training pipeline for pattern validation model"""
    
    def __init__(
        self,
        model_builder,  # PatternValidationModel instance
        data_dir: str,
        output_dir: str = './training_outputs'
    ):
        """
        Initialize trainer
        
        Args:
            model_builder: PatternValidationModel instance
            data_dir: Directory containing training data
            output_dir: Directory for saving outputs
        """
        self.model_builder = model_builder
        self.data_dir = data_dir
        self.output_dir = output_dir
        
        os.makedirs(output_dir, exist_ok=True)
        
        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None
        
        self.training_history = {}
        self.evaluation_results = {}
    
    def load_data(self) -> Dict:
        """Load training data from disk"""
        print("Loading training data...")
        
        X_chart = np.load(os.path.join(self.data_dir, 'X_chart.npy'))
        X_volume = np.load(os.path.join(self.data_dir, 'X_volume.npy'))
        X_pattern = np.load(os.path.join(self.data_dir, 'X_pattern.npy'))
        X_indicators = np.load(os.path.join(self.data_dir, 'X_indicators.npy'), allow_pickle=True)
    
        print(f"Original indicators dtype: {X_indicators.dtype}")
        print(f"Original indicators shape: {X_indicators.shape}")
        
        # Convert to float, handling any non-numeric values
        try:
            # If it's object dtype, convert each element
            if X_indicators.dtype == object:
                print("Converting object dtype to float32...")
                X_indicators_float = []
                for row in X_indicators:
                    # Convert each row to float, replacing NaN/None with 0
                    float_row = []
                    for val in row:
                        try:
                            f = float(val)
                            # Replace NaN/inf with 0
                            if not np.isfinite(f):
                                f = 0.0
                            float_row.append(f)
                        except (ValueError, TypeError):
                            float_row.append(0.0)
                    X_indicators_float.append(float_row)
                X_indicators = np.array(X_indicators_float, dtype=np.float32)
            else:
                # If already numeric, just ensure it's float32
                X_indicators = X_indicators.astype(np.float32)
                # Replace NaN/inf with 0
                X_indicators = np.nan_to_num(X_indicators, nan=0.0, posinf=0.0, neginf=0.0)
        except Exception as e:
            print(f"Error converting indicators: {e}")
            print("Using zeros as fallback...")
            X_indicators = np.zeros((len(X_chart), 15), dtype=np.float32)
        
        print(f"Final indicators dtype: {X_indicators.dtype}")
        print(f"Final indicators shape: {X_indicators.shape}")
        y = np.load(os.path.join(self.data_dir, 'y.npy'))
        
        print(f"Loaded {len(y)} training examples")
        
        return {
            'X_chart': X_chart,
            'X_volume': X_volume,
            'X_pattern': X_pattern,
            'X_indicators': X_indicators,
            'y': y
        }
    
    def split_data(
        self,
        data: Dict,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15
    ):
        """Split data into train/val/test"""
        n_samples = len(data['y'])
        
        train_end = int(n_samples * train_ratio)
        val_end = int(n_samples * (train_ratio + val_ratio))
        
        print(f"\nSplitting data: Train={train_end}, Val={val_end-train_end}, Test={n_samples-val_end}")
        
        self.X_train = {
            'chart': data['X_chart'][:train_end],
            'volume': data['X_volume'][:train_end],
            'pattern': data['X_pattern'][:train_end],
            'indicators': data['X_indicators'][:train_end]
        }
        
        self.X_val = {
            'chart': data['X_chart'][train_end:val_end],
            'volume': data['X_volume'][train_end:val_end],
            'pattern': data['X_pattern'][train_end:val_end],
            'indicators': data['X_indicators'][train_end:val_end]
        }
        
        self.X_test = {
            'chart': data['X_chart'][val_end:],
            'volume': data['X_volume'][val_end:],
            'pattern': data['X_pattern'][val_end:],
            'indicators': data['X_indicators'][val_end:]
        }
        
        self.y_train = data['y'][:train_end]
        self.y_val = data['y'][train_end:val_end]
        self.y_test = data['y'][val_end:]
    
    def train_progressive(
        self,
        epochs: int = 100,
        batch_size: int = 32,
        warmup_epochs: int = 20,
        warmup_data_pct: float = 0.2
    ):
        """Progressive training: warmup then full dataset"""
        print("\n" + "="*60)
        print("PROGRESSIVE TRAINING")
        print("="*60)
        
        # Phase 1: Warmup
        print(f"\nPhase 1: Warmup ({warmup_data_pct*100:.0f}% data)")
        warmup_size = int(len(self.y_train) * warmup_data_pct)
        
        X_warmup = {k: v[:warmup_size] for k, v in self.X_train.items()}
        y_warmup = self.y_train[:warmup_size]
        
        history_warmup = self.model_builder.model.fit(
            [X_warmup['chart'], X_warmup['volume'], X_warmup['pattern'], X_warmup['indicators']],
            y_warmup,
            validation_data=(
                [self.X_val['chart'], self.X_val['volume'], self.X_val['pattern'], self.X_val['indicators']],
                self.y_val
            ),
            epochs=warmup_epochs,
            batch_size=batch_size,
            callbacks=self.model_builder.get_callbacks(
                model_save_path=os.path.join(self.output_dir, 'warmup_model.keras'),
                patience=10
            ),
            verbose=1
        )
        
        # Phase 2: Fine-tune
        print(f"\nPhase 2: Fine-tuning (full dataset)")
        
        # FIX: Use the proper method to set learning rate
        # Option 1: Using assign() method (recommended for TF 2.x)
        self.model_builder.model.optimizer.learning_rate.assign(0.0001)
        
        # Alternative Option 2: Recompile with new learning rate
        # self.model_builder.model.compile(
        #     optimizer=keras.optimizers.Adam(learning_rate=0.0001),
        #     loss='mse',
        #     metrics=['mae']
        # )
        
        history_finetune = self.model_builder.model.fit(
            [self.X_train['chart'], self.X_train['volume'], self.X_train['pattern'], self.X_train['indicators']],
            self.y_train,
            validation_data=(
                [self.X_val['chart'], self.X_val['volume'], self.X_val['pattern'], self.X_val['indicators']],
                self.y_val
            ),
            epochs=epochs - warmup_epochs,
            batch_size=batch_size,
            callbacks=self.model_builder.get_callbacks(
                model_save_path=os.path.join(self.output_dir, 'best_model.keras'),
                patience=15
            ),
            verbose=1
        )
        
        self.training_history = {
            'warmup': history_warmup.history,
            'finetune': history_finetune.history
        }
    
    def evaluate(self) -> Dict:
        """Evaluate model on test set"""
        print("\n" + "="*60)
        print("MODEL EVALUATION")
        print("="*60)
        
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        
        y_pred_test = self.model_builder.model.predict(
            [self.X_test['chart'], self.X_test['volume'], self.X_test['pattern'], self.X_test['indicators']],
            verbose=0
        ).flatten()
        
        results = {
            'test': {
                'mse': mean_squared_error(self.y_test, y_pred_test),
                'mae': mean_absolute_error(self.y_test, y_pred_test),
                'rmse': np.sqrt(mean_squared_error(self.y_test, y_pred_test)),
                'r2': r2_score(self.y_test, y_pred_test),
                'correlation': np.corrcoef(self.y_test, y_pred_test)[0, 1]
            }
        }
        
        print(f"\nTEST SET:")
        print(f"  MSE: {results['test']['mse']:.4f}")
        print(f"  MAE: {results['test']['mae']:.4f}")
        print(f"  RMSE: {results['test']['rmse']:.4f}")
        print(f"  R²: {results['test']['r2']:.4f}")
        print(f"  Correlation: {results['test']['correlation']:.4f}")
        
        self.evaluation_results = results
        return results
    
    def plot_training_history(self):
        """Plot training curves"""
        import matplotlib.pyplot as plt
        
        if 'warmup' in self.training_history:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
            
            ax1.plot(self.training_history['warmup']['loss'], label='Train')
            ax1.plot(self.training_history['warmup']['val_loss'], label='Val')
            ax1.set_title('Warmup Phase')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.legend()
            ax1.grid(alpha=0.3)
            
            ax2.plot(self.training_history['finetune']['loss'], label='Train')
            ax2.plot(self.training_history['finetune']['val_loss'], label='Val')
            ax2.set_title('Fine-tuning Phase')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Loss')
            ax2.legend()
            ax2.grid(alpha=0.3)
        
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'training_history.png'), dpi=300)
            print(f"\nTraining history saved to {self.output_dir}/training_history.png")
            plt.close()
    
    def plot_predictions(self):
        """Plot predictions vs actual"""
        import matplotlib.pyplot as plt
        
        y_pred_test = self.model_builder.model.predict(
            [self.X_test['chart'], self.X_test['volume'], self.X_test['pattern'], self.X_test['indicators']],
            verbose=0
        ).flatten()
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Scatter plot
        ax1.scatter(self.y_test, y_pred_test, alpha=0.5)
        ax1.plot([self.y_test.min(), self.y_test.max()], 
                 [self.y_test.min(), self.y_test.max()], 'r--', lw=2)
        ax1.set_xlabel('Actual')
        ax1.set_ylabel('Predicted')
        ax1.set_title('Predictions vs Actual')
        ax1.grid(alpha=0.3)
        
        # Residuals
        residuals = self.y_test - y_pred_test
        ax2.scatter(y_pred_test, residuals, alpha=0.5)
        ax2.axhline(y=0, color='r', linestyle='--', lw=2)
        ax2.set_xlabel('Predicted')
        ax2.set_ylabel('Residuals')
        ax2.set_title('Residual Plot')
        ax2.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'predictions.png'), dpi=300)
        print(f"Predictions plot saved to {self.output_dir}/predictions.png")
        plt.close()
    
    def plot_error_distribution(self):
        """Plot error distribution"""
        import matplotlib.pyplot as plt
        
        y_pred_test = self.model_builder.model.predict(
            [self.X_test['chart'], self.X_test['volume'], self.X_test['pattern'], self.X_test['indicators']],
            verbose=0
        ).flatten()
        
        errors = self.y_test - y_pred_test
        
        plt.figure(figsize=(10, 6))
        plt.hist(errors, bins=50, edgecolor='black', alpha=0.7)
        plt.axvline(x=0, color='r', linestyle='--', lw=2)
        plt.xlabel('Prediction Error')
        plt.ylabel('Frequency')
        plt.title('Error Distribution')
        plt.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'error_distribution.png'), dpi=300)
        print(f"Error distribution saved to {self.output_dir}/error_distribution.png")
        plt.close()
    
    def save_results(self):
        """Save results to disk"""
        import json
        
        metrics_file = os.path.join(self.output_dir, 'evaluation_metrics.json')
        with open(metrics_file, 'w') as f:
            json.dump(self.evaluation_results, f, indent=2)
        
        print(f"\nResults saved to {metrics_file}")


def run_full_training_pipeline(
    data_dir: str,
    model_builder,  # Pass the model builder instance
    output_dir: str = './pattern_validation_outputs',
    use_progressive: bool = True,
    epochs: int = 100,
    batch_size: int = 32,
    warmup_epochs: int = 20
):
    """
    Complete end-to-end training pipeline
    
    Args:
        data_dir: Directory with training data
        model_builder: PatternValidationModel instance (already built and compiled)
        output_dir: Directory for outputs
        use_progressive: Use progressive training
        epochs: Total epochs
        batch_size: Batch size
        warmup_epochs: Epochs for warmup phase
    """
    print("\n" + "="*60)
    print("PATTERN VALIDATION MODEL - TRAINING PIPELINE")
    print("="*60)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Initialize trainer
    trainer = PatternValidationTrainer(
        model_builder=model_builder,
        data_dir=data_dir,
        output_dir=output_dir
    )
    
    # Load and split data
    data = trainer.load_data()
    trainer.split_data(data)
    
    # Train
    if use_progressive:
        trainer.train_progressive(
            epochs=epochs, 
            batch_size=batch_size, 
            warmup_epochs=warmup_epochs
        )
    else:
        # Standard training
        history = model_builder.model.fit(
            [trainer.X_train['chart'], trainer.X_train['volume'], 
             trainer.X_train['pattern'], trainer.X_train['indicators']],
            trainer.y_train,
            validation_data=(
                [trainer.X_val['chart'], trainer.X_val['volume'], 
                 trainer.X_val['pattern'], trainer.X_val['indicators']],
                trainer.y_val
            ),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=model_builder.get_callbacks(
                model_save_path=os.path.join(output_dir, 'best_model.keras')
            ),
            verbose=1
        )
        trainer.training_history = history.history
    
    # Evaluate
    trainer.evaluate()
    
    # Plot results
    trainer.plot_training_history()
    trainer.plot_predictions()
    trainer.plot_error_distribution()
    
    # Save everything
    trainer.save_results()
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"All outputs saved to: {output_dir}")
    
    return trainer


# Example usage
if __name__ == "__main__":
    """
    Example of how to use the fixed training pipeline
    """
    
    # You would import and initialize your model builder here
    # from model_architecture import PatternValidationModel
    
    # model_builder = PatternValidationModel(
    #     chart_shape=(100, 5),
    #     volume_shape=(100, 1),
    #     pattern_features_dim=13,
    #     indicator_features_dim=15,
    #     cnn_filters=(64, 128, 256),
    #     lstm_units=(128, 64),
    #     dropout_rate=0.4
    # )
    # 
    # model_builder.build_model()
    # model_builder.compile_model(learning_rate=0.001, loss='mse')
    
    # Then run training
    # trainer = run_full_training_pipeline(
    #     data_dir='./training_data',
    #     model_builder=model_builder,
    #     output_dir='./outputs',
    #     use_progressive=True,
    #     epochs=100,
    #     batch_size=32,
    #     warmup_epochs=20
    # )
    
    print("Fixed training pipeline ready to use!")