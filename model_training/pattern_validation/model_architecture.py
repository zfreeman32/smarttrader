"""
Pattern Validation Model Architecture
======================================
CNN-LSTM Hybrid model for validating ICT pattern quality.

Architecture:
- CNN branch: Extracts spatial features from candlestick charts
- LSTM branch: Captures temporal dynamics from volume profiles
- Dense branch: Processes pattern metadata
- Merged layers: Combines all features for final prediction
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import numpy as np
from typing import Tuple, Dict, Optional
import json


class PatternValidationModel:
    """CNN-LSTM Hybrid for Pattern Quality Scoring"""
    
    def __init__(
        self,
        chart_shape: Tuple[int, int] = (100, 5),  # (timesteps, OHLCV)
        volume_shape: Tuple[int, int] = (100, 1),  # (timesteps, volume)
        pattern_features_dim: int = 13,
        indicator_features_dim: int = 15,
        cnn_filters: Tuple[int, int, int] = (64, 128, 256),
        lstm_units: Tuple[int, int] = (128, 64),
        dropout_rate: float = 0.4,
        l2_reg: float = 0.001
    ):
        """
        Initialize model architecture
        
        Args:
            chart_shape: Shape of candlestick chart input (timesteps, features)
            volume_shape: Shape of volume profile input
            pattern_features_dim: Number of pattern metadata features
            indicator_features_dim: Number of indicator features
            cnn_filters: Number of filters for each CNN layer
            lstm_units: Number of units for each LSTM layer
            dropout_rate: Dropout rate for regularization
            l2_reg: L2 regularization factor
        """
        self.chart_shape = chart_shape
        self.volume_shape = volume_shape
        self.pattern_features_dim = pattern_features_dim
        self.indicator_features_dim = indicator_features_dim
        self.cnn_filters = cnn_filters
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.l2_reg = l2_reg
        
        self.model = None
        self.history = None
    
    def build_model(self) -> Model:
        """Build the complete CNN-LSTM hybrid architecture"""
        
        # ==================== INPUT LAYERS ====================
        
        # Chart input (100 candles x 5 OHLCV)
        chart_input = layers.Input(
            shape=self.chart_shape,
            name='chart_input'
        )
        
        # Volume profile input (100 timesteps x 1)
        volume_input = layers.Input(
            shape=self.volume_shape,
            name='volume_input'
        )
        
        # Pattern metadata input (13 features)
        pattern_input = layers.Input(
            shape=(self.pattern_features_dim,),
            name='pattern_input'
        )
        
        # Technical indicators input (15 features)
        indicator_input = layers.Input(
            shape=(self.indicator_features_dim,),
            name='indicator_input'
        )
        
        # ==================== CNN BRANCH ====================
        # Process candlestick chart patterns
        
        x_cnn = chart_input
        
        # First CNN block
        x_cnn = layers.Conv1D(
            filters=self.cnn_filters[0],
            kernel_size=3,
            padding='same',
            kernel_regularizer=keras.regularizers.l2(self.l2_reg),
            name='conv1d_1'
        )(x_cnn)
        x_cnn = layers.BatchNormalization(name='bn_1')(x_cnn)
        x_cnn = layers.Activation('relu', name='relu_1')(x_cnn)
        x_cnn = layers.MaxPooling1D(pool_size=2, name='maxpool_1')(x_cnn)
        x_cnn = layers.Dropout(self.dropout_rate, name='dropout_1')(x_cnn)
        
        # Second CNN block
        x_cnn = layers.Conv1D(
            filters=self.cnn_filters[1],
            kernel_size=3,
            padding='same',
            kernel_regularizer=keras.regularizers.l2(self.l2_reg),
            name='conv1d_2'
        )(x_cnn)
        x_cnn = layers.BatchNormalization(name='bn_2')(x_cnn)
        x_cnn = layers.Activation('relu', name='relu_2')(x_cnn)
        x_cnn = layers.MaxPooling1D(pool_size=2, name='maxpool_2')(x_cnn)
        x_cnn = layers.Dropout(self.dropout_rate, name='dropout_2')(x_cnn)
        
        # Third CNN block
        x_cnn = layers.Conv1D(
            filters=self.cnn_filters[2],
            kernel_size=3,
            padding='same',
            kernel_regularizer=keras.regularizers.l2(self.l2_reg),
            name='conv1d_3'
        )(x_cnn)
        x_cnn = layers.BatchNormalization(name='bn_3')(x_cnn)
        x_cnn = layers.Activation('relu', name='relu_3')(x_cnn)
        x_cnn = layers.GlobalAveragePooling1D(name='global_avg_pool')(x_cnn)
        
        # ==================== LSTM BRANCH ====================
        # Process volume profile temporal patterns
        
        x_lstm = volume_input
        
        # First LSTM layer
        x_lstm = layers.LSTM(
            units=self.lstm_units[0],
            return_sequences=True,
            kernel_regularizer=keras.regularizers.l2(self.l2_reg),
            name='lstm_1'
        )(x_lstm)
        x_lstm = layers.Dropout(self.dropout_rate, name='lstm_dropout_1')(x_lstm)
        
        # Second LSTM layer
        x_lstm = layers.LSTM(
            units=self.lstm_units[1],
            return_sequences=False,
            kernel_regularizer=keras.regularizers.l2(self.l2_reg),
            name='lstm_2'
        )(x_lstm)
        x_lstm = layers.Dropout(self.dropout_rate, name='lstm_dropout_2')(x_lstm)
        
        # ==================== PATTERN METADATA BRANCH ====================
        
        x_pattern = pattern_input
        x_pattern = layers.Dense(
            64,
            activation='relu',
            kernel_regularizer=keras.regularizers.l2(self.l2_reg),
            name='pattern_dense_1'
        )(x_pattern)
        x_pattern = layers.Dropout(self.dropout_rate, name='pattern_dropout')(x_pattern)
        
        # ==================== INDICATOR BRANCH ====================
        
        x_indicator = indicator_input
        x_indicator = layers.Dense(
            32,
            activation='relu',
            kernel_regularizer=keras.regularizers.l2(self.l2_reg),
            name='indicator_dense_1'
        )(x_indicator)
        x_indicator = layers.Dropout(self.dropout_rate, name='indicator_dropout')(x_indicator)
        
        # ==================== MERGE ALL BRANCHES ====================
        
        merged = layers.Concatenate(name='merge_all')([
            x_cnn,
            x_lstm,
            x_pattern,
            x_indicator
        ])
        
        # ==================== FINAL CLASSIFICATION LAYERS ====================
        
        x = layers.Dense(
            128,
            activation='relu',
            kernel_regularizer=keras.regularizers.l2(self.l2_reg),
            name='dense_1'
        )(merged)
        x = layers.Dropout(self.dropout_rate, name='final_dropout_1')(x)
        
        x = layers.Dense(
            64,
            activation='relu',
            kernel_regularizer=keras.regularizers.l2(self.l2_reg),
            name='dense_2'
        )(x)
        x = layers.Dropout(self.dropout_rate * 0.5, name='final_dropout_2')(x)
        
        # Output: Pattern quality score [0, 1]
        output = layers.Dense(
            1,
            activation='sigmoid',
            name='quality_score_output'
        )(x)
        
        # ==================== CREATE MODEL ====================
        
        model = Model(
            inputs=[chart_input, volume_input, pattern_input, indicator_input],
            outputs=output,
            name='PatternValidationModel'
        )
        
        self.model = model
        return model
    
    def compile_model(
        self,
        learning_rate: float = 0.001,
        loss: str = 'mse',  # or 'focal_loss' for imbalanced data
        metrics: Optional[list] = None
    ):
        """Compile the model with optimizer and loss"""
        
        if metrics is None:
            metrics = ['mae', 'mse']
        
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        
        # Use custom focal loss if specified
        if loss == 'focal_loss':
            loss_fn = self._focal_loss(gamma=2.0, alpha=0.25)
        else:
            loss_fn = loss
        
        self.model.compile(
            optimizer=optimizer,
            loss=loss_fn,
            metrics=metrics
        )
    
    @staticmethod
    def _focal_loss(gamma=2.0, alpha=0.25):
        """Focal loss for handling class imbalance"""
        def focal_loss_fixed(y_true, y_pred):
            epsilon = tf.keras.backend.epsilon()
            y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
            
            # Calculate focal loss
            pt = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
            focal_weight = tf.pow(1 - pt, gamma)
            focal_loss = -alpha * focal_weight * tf.math.log(pt)
            
            return tf.reduce_mean(focal_loss)
        
        return focal_loss_fixed
    
    def get_callbacks(
        self,
        model_save_path: str = './best_pattern_validator.keras',
        patience: int = 15
    ) -> list:
        """Get training callbacks"""
        
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=patience,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            ),
            ModelCheckpoint(
                filepath=model_save_path,
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            )
        ]
        
        return callbacks
    
    def summary(self):
        """Print model architecture summary"""
        if self.model is None:
            print("Model not built yet. Call build_model() first.")
            return
        
        self.model.summary()
        
        # Print parameter counts per branch
        print("\n" + "="*60)
        print("PARAMETER BREAKDOWN BY BRANCH")
        print("="*60)
        
        cnn_params = sum([np.prod(v.shape) for v in self.model.get_layer('conv1d_1').trainable_weights])
        lstm_params = sum([np.prod(v.shape) for v in self.model.get_layer('lstm_1').trainable_weights])
        
        print(f"CNN Branch: ~{cnn_params:,} parameters")
        print(f"LSTM Branch: ~{lstm_params:,} parameters")
        print(f"Total Trainable: {self.model.count_params():,} parameters")
        print("="*60 + "\n")
    
    def save_config(self, path: str):
        """Save model configuration"""
        config = {
            'chart_shape': self.chart_shape,
            'volume_shape': self.volume_shape,
            'pattern_features_dim': self.pattern_features_dim,
            'indicator_features_dim': self.indicator_features_dim,
            'cnn_filters': self.cnn_filters,
            'lstm_units': self.lstm_units,
            'dropout_rate': self.dropout_rate,
            'l2_reg': self.l2_reg
        }
        
        with open(path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"Model config saved to {path}")
    
    @classmethod
    def load_from_config(cls, config_path: str):
        """Load model from configuration file"""
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        return cls(**config)


def build_alternative_architectures():
    """Build alternative model architectures for comparison"""
    
    # Architecture 1: Deeper CNN
    model_v1 = PatternValidationModel(
        cnn_filters=(64, 128, 256, 512),
        lstm_units=(128, 64),
        dropout_rate=0.5
    )
    
    # Architecture 2: Bidirectional LSTM
    # (Would require modification to use Bidirectional wrapper)
    
    # Architecture 3: Attention mechanism
    # (Would require custom attention layer)
    
    return model_v1


# Example usage
if __name__ == "__main__":
    
    # Build model
    model_builder = PatternValidationModel(
        chart_shape=(100, 5),
        volume_shape=(100, 1),
        pattern_features_dim=13,
        indicator_features_dim=15,
        cnn_filters=(64, 128, 256),
        lstm_units=(128, 64),
        dropout_rate=0.4
    )
    
    model = model_builder.build_model()
    model_builder.compile_model(learning_rate=0.001, loss='mse')
    
    # Print summary
    model_builder.summary()
    
    # Save config
    model_builder.save_config('./model_config.json')
    
    print("\nModel built successfully!")
    print(f"Total parameters: {model.count_params():,}")