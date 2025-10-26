"""
Multi-Timeframe Confluence Model Training Pipeline
===================================================

Builds a production-ready model that analyzes alignment across multiple timeframes
to generate confluence scores and directional bias for trading decisions.

Architecture: Multi-Stream Transformer with attention across timeframes
Output: Confluence score [0-1] and directional bias [-1 to +1]
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple, Optional
import json
import warnings
warnings.filterwarnings('ignore')


class MultiTimeframeDataPreparator:
    """Prepares aligned multi-timeframe data for confluence modeling"""
    
    def __init__(self, 
                 timeframes: Dict[str, int] = None,
                 lookback_periods: Dict[str, int] = None):
        """
        Args:
            timeframes: Dict of timeframe names to minutes (e.g., {'1m': 1, '5m': 5})
            lookback_periods: Number of bars to look back per timeframe
        """
        if timeframes is None:
            self.timeframes = {
                '1m': 1,
                '5m': 5,
                '15m': 15,
                '1h': 60,
                '4h': 240
            }
        else:
            self.timeframes = timeframes
            
        if lookback_periods is None:
            self.lookback_periods = {
                '1m': 240,
                '5m': 100,
                '15m': 50,
                '1h': 24,
                '4h': 20
            }
        else:
            self.lookback_periods = lookback_periods
            
        self.scalers = {}
        
    def calculate_features_per_timeframe(self, df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """
        Calculate confluence-relevant features for a single timeframe
        
        Features:
        - Trend direction (EMA crossovers, slope)
        - Momentum (RSI, STOCHRSI, MACD)
        - Volume confirmation
        - Volatility (ATR, Bollinger Bands)
        - Price position relative to key levels
        """
        features = df.copy()
        
        # Trend Features
        features['ema_9'] = features['close'].ewm(span=9, adjust=False).mean()
        features['ema_21'] = features['close'].ewm(span=21, adjust=False).mean()
        features['ema_50'] = features['close'].ewm(span=50, adjust=False).mean()
        
        # EMA crossover signals
        features['ema_9_21_cross'] = np.where(features['ema_9'] > features['ema_21'], 1, -1)
        features['ema_21_50_cross'] = np.where(features['ema_21'] > features['ema_50'], 1, -1)
        
        # Trend slope (rate of change)
        features['ema_50_slope'] = features['ema_50'].pct_change(5)
        features['price_slope'] = features['close'].pct_change(10)
        
        # Momentum Features
        # RSI
        delta = features['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        features['rsi'] = 100 - (100 / (1 + rs))
        features['rsi_signal'] = np.where(features['rsi'] > 50, 1, -1)
        
        # MACD
        exp1 = features['close'].ewm(span=12, adjust=False).mean()
        exp2 = features['close'].ewm(span=26, adjust=False).mean()
        features['macd'] = exp1 - exp2
        features['macd_signal'] = features['macd'].ewm(span=9, adjust=False).mean()
        features['macd_histogram'] = features['macd'] - features['macd_signal']
        features['macd_direction'] = np.where(features['macd'] > features['macd_signal'], 1, -1)
        
        # Stochastic RSI
        rsi_min = features['rsi'].rolling(window=14).min()
        rsi_max = features['rsi'].rolling(window=14).max()
        features['stoch_rsi'] = (features['rsi'] - rsi_min) / (rsi_max - rsi_min)
        features['stoch_rsi_signal'] = np.where(features['stoch_rsi'] > 0.5, 1, -1)
        
        # Volume Features
        features['volume_sma'] = features['volume'].rolling(window=20).mean()
        features['volume_ratio'] = features['volume'] / features['volume_sma']
        features['volume_trend'] = np.where(features['volume_ratio'] > 1.2, 1, 
                                            np.where(features['volume_ratio'] < 0.8, -1, 0))
        
        # Volume-weighted average price
        features['vwap'] = (features['volume'] * (features['high'] + features['low'] + features['close']) / 3).cumsum() / features['volume'].cumsum()
        features['price_vs_vwap'] = np.where(features['close'] > features['vwap'], 1, -1)
        
        # Volatility Features
        # ATR
        high_low = features['high'] - features['low']
        high_close = np.abs(features['high'] - features['close'].shift())
        low_close = np.abs(features['low'] - features['close'].shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        features['atr'] = true_range.rolling(window=14).mean()
        features['atr_pct'] = features['atr'] / features['close']
        
        # Bollinger Bands
        features['bb_middle'] = features['close'].rolling(window=20).mean()
        bb_std = features['close'].rolling(window=20).std()
        features['bb_upper'] = features['bb_middle'] + (bb_std * 2)
        features['bb_lower'] = features['bb_middle'] - (bb_std * 2)
        features['bb_width'] = (features['bb_upper'] - features['bb_lower']) / features['bb_middle']
        features['bb_position'] = (features['close'] - features['bb_lower']) / (features['bb_upper'] - features['bb_lower'])
        
        # Price position relative to high/low
        features['high_20'] = features['high'].rolling(window=20).max()
        features['low_20'] = features['low'].rolling(window=20).min()
        features['price_range_position'] = (features['close'] - features['low_20']) / (features['high_20'] - features['low_20'])
        
        # Distance from key levels
        features['distance_from_high'] = (features['high_20'] - features['close']) / features['close']
        features['distance_from_low'] = (features['close'] - features['low_20']) / features['close']
        
        # Composite signals
        features['bullish_signals'] = (
            (features['ema_9_21_cross'] == 1).astype(int) +
            (features['ema_21_50_cross'] == 1).astype(int) +
            (features['rsi_signal'] == 1).astype(int) +
            (features['macd_direction'] == 1).astype(int) +
            (features['stoch_rsi_signal'] == 1).astype(int) +
            (features['price_vs_vwap'] == 1).astype(int)
        )
        
        features['bearish_signals'] = (
            (features['ema_9_21_cross'] == -1).astype(int) +
            (features['ema_21_50_cross'] == -1).astype(int) +
            (features['rsi_signal'] == -1).astype(int) +
            (features['macd_direction'] == -1).astype(int) +
            (features['stoch_rsi_signal'] == -1).astype(int) +
            (features['price_vs_vwap'] == -1).astype(int)
        )
        
        features['signal_consensus'] = (features['bullish_signals'] - features['bearish_signals']) / 6
        
        # Forward fill NaN values
        features = features.fillna(method='ffill').fillna(0)
        
        return features
    
    def resample_to_timeframe(self, df: pd.DataFrame, target_minutes: int) -> pd.DataFrame:
        """Resample 1-minute data to target timeframe"""
        df = df.copy()
        df.index = pd.to_datetime(df.index)
        
        resampled = df.resample(f'{target_minutes}T').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()
        
        return resampled
    
    def create_aligned_dataset(self, df_1m: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Create aligned multi-timeframe dataset from 1-minute data
        
        Returns dict with keys: '1m', '5m', '15m', '1h', '4h'
        Each value is a 3D array: [samples, timesteps, features]
        """
        aligned_data = {}
        
        # Ensure datetime index
        df_1m = df_1m.copy()
        if not isinstance(df_1m.index, pd.DatetimeIndex):
            df_1m.index = pd.to_datetime(df_1m.index)
        
        print("Creating multi-timeframe dataset...")
        
        for tf_name, tf_minutes in self.timeframes.items():
            print(f"Processing {tf_name} timeframe...")
            
            # Resample if not 1m
            if tf_minutes == 1:
                df_tf = df_1m.copy()
            else:
                df_tf = self.resample_to_timeframe(df_1m, tf_minutes)
            
            # Calculate features
            df_tf = self.calculate_features_per_timeframe(df_tf, tf_name)
            
            # Select feature columns (exclude OHLCV)
            feature_cols = [col for col in df_tf.columns if col not in ['open', 'high', 'low', 'close', 'volume']]
            
            # Create sequences
            lookback = self.lookback_periods[tf_name]
            sequences = []
            
            for i in range(lookback, len(df_tf)):
                seq = df_tf[feature_cols].iloc[i-lookback:i].values
                sequences.append(seq)
            
            aligned_data[tf_name] = np.array(sequences)
            
            print(f"{tf_name}: {aligned_data[tf_name].shape}")
        
        return aligned_data
    
    def calculate_labels(self, df_1m: pd.DataFrame, forward_window: int = 20) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate confluence score and directional bias labels
        
        Returns:
            confluence_scores: [0-1] how well future direction is predicted
            directional_bias: [-1 to +1] future price direction
        """
        df = df_1m.copy()
        
        # Calculate future returns
        df['future_return'] = df['close'].pct_change(forward_window).shift(-forward_window)
        
        # Directional bias: normalized future return
        df['directional_bias'] = np.tanh(df['future_return'] * 100)  # Normalize to [-1, 1]
        
        # Confluence score: based on consistency of direction across multiple horizons
        df['return_5'] = df['close'].pct_change(5).shift(-5)
        df['return_10'] = df['close'].pct_change(10).shift(-10)
        df['return_15'] = df['close'].pct_change(15).shift(-15)
        df['return_20'] = df['close'].pct_change(20).shift(-20)
        
        # Calculate how many horizons agree on direction
        returns_matrix = df[['return_5', 'return_10', 'return_15', 'return_20']].values
        signs = np.sign(returns_matrix)
        
        # Confluence = proportion of horizons agreeing with majority
        majority_sign = np.sign(np.sum(signs, axis=1))
        agreement = np.sum(signs == majority_sign[:, np.newaxis], axis=1) / 4
        
        df['confluence_score'] = agreement
        
        # Clean up
        df = df.dropna()
        
        confluence_scores = df['confluence_score'].values
        directional_bias = df['directional_bias'].values
        
        return confluence_scores, directional_bias
    
    def prepare_training_data(self, 
                             df_1m: pd.DataFrame,
                             validation_split: float = 0.2) -> Tuple[Dict, Dict, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Complete data preparation pipeline
        
        Returns:
            X_train: Dict of timeframe data
            X_val: Dict of timeframe data
            y_train_confluence: Confluence scores
            y_train_bias: Directional bias
            y_val_confluence: Confluence scores
            y_val_bias: Directional bias
        """
        # Create aligned dataset
        aligned_data = self.create_aligned_dataset(df_1m)
        
        # Calculate labels
        confluence_scores, directional_bias = self.calculate_labels(df_1m)
        
        # Align lengths (labels are calculated from the end)
        min_samples = min([data.shape[0] for data in aligned_data.values()] + [len(confluence_scores)])
        
        for tf_name in aligned_data:
            aligned_data[tf_name] = aligned_data[tf_name][-min_samples:]
        
        confluence_scores = confluence_scores[-min_samples:]
        directional_bias = directional_bias[-min_samples:]
        
        # Split into train/validation
        split_idx = int(len(confluence_scores) * (1 - validation_split))
        
        X_train = {tf: data[:split_idx] for tf, data in aligned_data.items()}
        X_val = {tf: data[split_idx:] for tf, data in aligned_data.items()}
        
        y_train_confluence = confluence_scores[:split_idx]
        y_train_bias = directional_bias[:split_idx]
        
        y_val_confluence = confluence_scores[split_idx:]
        y_val_bias = directional_bias[split_idx:]
        
        # Fit scalers on training data and transform both
        for tf_name in self.timeframes.keys():
            scaler = StandardScaler()
            
            # Reshape to 2D for scaling
            original_shape = X_train[tf_name].shape
            X_train_reshaped = X_train[tf_name].reshape(-1, original_shape[-1])
            
            scaler.fit(X_train_reshaped)
            X_train[tf_name] = scaler.transform(X_train_reshaped).reshape(original_shape)
            
            # Transform validation
            val_shape = X_val[tf_name].shape
            X_val_reshaped = X_val[tf_name].reshape(-1, val_shape[-1])
            X_val[tf_name] = scaler.transform(X_val_reshaped).reshape(val_shape)
            
            self.scalers[tf_name] = scaler
        
        print(f"\nTraining samples: {len(y_train_confluence)}")
        print(f"Validation samples: {len(y_val_confluence)}")
        
        return X_train, X_val, y_train_confluence, y_train_bias, y_val_confluence, y_val_bias


class MultiTimeframeConfluenceModel:
    """Multi-Stream Transformer for timeframe confluence analysis"""
    
    def __init__(self, 
                 input_shapes: Dict[str, Tuple],
                 d_model: int = 128,
                 num_heads: int = 4,
                 ff_dim: int = 256,
                 num_transformer_blocks: int = 2,
                 dropout_rate: float = 0.2):
        """
        Args:
            input_shapes: Dict of {timeframe: (timesteps, features)}
            d_model: Transformer model dimension
            num_heads: Number of attention heads
            ff_dim: Feed-forward dimension
            num_transformer_blocks: Number of transformer blocks
            dropout_rate: Dropout rate
        """
        self.input_shapes = input_shapes
        self.d_model = d_model
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.num_transformer_blocks = num_transformer_blocks
        self.dropout_rate = dropout_rate
        
        self.model = None
        
    def transformer_encoder(self, inputs, d_model, num_heads, ff_dim, dropout_rate, name_prefix):
        """Single transformer encoder block"""
        # Multi-head attention
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=d_model // num_heads,
            dropout=dropout_rate,
            name=f"{name_prefix}_attention"
        )(inputs, inputs)
        
        attention_output = layers.Dropout(dropout_rate)(attention_output)
        attention_output = layers.LayerNormalization(epsilon=1e-6)(inputs + attention_output)
        
        # Feed-forward network
        ffn_output = layers.Dense(ff_dim, activation='relu', name=f"{name_prefix}_ffn1")(attention_output)
        ffn_output = layers.Dropout(dropout_rate)(ffn_output)
        ffn_output = layers.Dense(d_model, name=f"{name_prefix}_ffn2")(ffn_output)
        ffn_output = layers.Dropout(dropout_rate)(ffn_output)
        
        # Residual connection
        output = layers.LayerNormalization(epsilon=1e-6)(attention_output + ffn_output)
        
        return output
    
    def build_model(self):
        """Build multi-stream transformer architecture"""
        
        # Input layers for each timeframe
        inputs = {}
        stream_outputs = []
        
        for tf_name, shape in self.input_shapes.items():
            # Input layer
            input_layer = layers.Input(shape=shape, name=f"input_{tf_name}")
            inputs[tf_name] = input_layer
            
            # Project to d_model dimensions
            x = layers.Dense(self.d_model, name=f"{tf_name}_projection")(input_layer)
            x = layers.LayerNormalization(epsilon=1e-6)(x)
            
            # Apply transformer blocks
            for i in range(self.num_transformer_blocks):
                x = self.transformer_encoder(
                    x, 
                    self.d_model, 
                    self.num_heads, 
                    self.ff_dim, 
                    self.dropout_rate,
                    name_prefix=f"{tf_name}_block{i}"
                )
            
            # Global average pooling
            x = layers.GlobalAveragePooling1D(name=f"{tf_name}_pool")(x)
            stream_outputs.append(x)
        
        # Concatenate all timeframe streams
        concatenated = layers.Concatenate(name="concat_streams")(stream_outputs)
        
        # Cross-timeframe attention/fusion
        x = layers.Dense(self.d_model * 2, activation='relu', name="fusion_dense1")(concatenated)
        x = layers.Dropout(self.dropout_rate)(x)
        x = layers.Dense(self.d_model, activation='relu', name="fusion_dense2")(x)
        x = layers.Dropout(self.dropout_rate)(x)
        
        # Dual outputs
        # Confluence score [0-1]
        confluence_output = layers.Dense(64, activation='relu', name="confluence_dense")(x)
        confluence_output = layers.Dropout(self.dropout_rate)(confluence_output)
        confluence_output = layers.Dense(1, activation='sigmoid', name="confluence_score")(confluence_output)
        
        # Directional bias [-1 to +1]
        bias_output = layers.Dense(64, activation='relu', name="bias_dense")(x)
        bias_output = layers.Dropout(self.dropout_rate)(bias_output)
        bias_output = layers.Dense(1, activation='tanh', name="directional_bias")(bias_output)
        
        # Create model
        self.model = Model(
            inputs=list(inputs.values()),
            outputs=[confluence_output, bias_output],
            name="MultiTimeframeConfluenceModel"
        )
        
        return self.model
    
    def compile_model(self, learning_rate: float = 0.001):
        """Compile model with appropriate losses"""
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss={
                'confluence_score': 'mse',
                'directional_bias': 'mse'
            },
            loss_weights={
                'confluence_score': 1.0,
                'directional_bias': 1.0
            },
            metrics={
                'confluence_score': ['mae', 'mse'],
                'directional_bias': ['mae', 'mse']
            }
        )
        
        return self.model
    
    def get_model_summary(self):
        """Print model architecture"""
        if self.model is None:
            raise ValueError("Model not built yet. Call build_model() first.")
        return self.model.summary()


class ConfluenceModelTrainer:
    """Training pipeline for confluence model"""
    
    def __init__(self, model: MultiTimeframeConfluenceModel, model_save_path: str):
        self.model = model
        self.model_save_path = model_save_path
        self.history = None
        
    def train(self,
             X_train: Dict[str, np.ndarray],
             y_train_confluence: np.ndarray,
             y_train_bias: np.ndarray,
             X_val: Dict[str, np.ndarray],
             y_val_confluence: np.ndarray,
             y_val_bias: np.ndarray,
             epochs: int = 100,
             batch_size: int = 32,
             patience: int = 15):
        """
        Train the confluence model
        
        Args:
            X_train: Dict of timeframe training data
            y_train_confluence: Training confluence scores
            y_train_bias: Training directional bias
            X_val: Dict of timeframe validation data
            y_val_confluence: Validation confluence scores
            y_val_bias: Validation directional bias
            epochs: Maximum epochs
            batch_size: Batch size
            patience: Early stopping patience
        """
        
        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=patience,
                restore_best_weights=True,
                verbose=1
            ),
            ModelCheckpoint(
                filepath=self.model_save_path,
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=patience//3,
                min_lr=1e-6,
                verbose=1
            )
        ]
        
        # Prepare input data (model expects list of arrays)
        X_train_list = [X_train[tf] for tf in sorted(X_train.keys())]
        X_val_list = [X_val[tf] for tf in sorted(X_val.keys())]
        
        # Train
        print("\nStarting training...")
        self.history = self.model.model.fit(
            X_train_list,
            {
                'confluence_score': y_train_confluence,
                'directional_bias': y_train_bias
            },
            validation_data=(
                X_val_list,
                {
                    'confluence_score': y_val_confluence,
                    'directional_bias': y_val_bias
                }
            ),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        print(f"\nTraining complete. Best model saved to: {self.model_save_path}")
        
        return self.history
    
    def evaluate(self, X_test: Dict[str, np.ndarray], 
                y_test_confluence: np.ndarray,
                y_test_bias: np.ndarray):
        """Evaluate model on test set"""
        
        X_test_list = [X_test[tf] for tf in sorted(X_test.keys())]
        
        results = self.model.model.evaluate(
            X_test_list,
            {
                'confluence_score': y_test_confluence,
                'directional_bias': y_test_bias
            },
            verbose=1
        )
        
        return results


def main():
    """
    Main training pipeline
    
    Usage:
        # Assumes you have 1-minute EURUSD data in a CSV with columns:
        # timestamp, open, high, low, close, volume
        
        df = pd.read_csv('eurusd_1m_data.csv', index_col='timestamp', parse_dates=True)
        
        # Run training
        trainer = train_confluence_model(df)
    """
    
    print("=" * 80)
    print("Multi-Timeframe Confluence Model Training Pipeline")
    print("=" * 80)
    
    # Example with dummy data (replace with your actual data)
    print("\nNote: This example uses dummy data.")
    print("Replace with your actual EURUSD 1-minute data.")
    
    # Create dummy data for demonstration
    dates = pd.date_range(start='2023-01-01', end='2024-12-31', freq='1T')
    np.random.seed(42)
    
    df_1m = pd.DataFrame({
        'open': 1.08 + np.random.randn(len(dates)) * 0.001,
        'high': 1.08 + np.random.randn(len(dates)) * 0.001,
        'low': 1.08 + np.random.randn(len(dates)) * 0.001,
        'close': 1.08 + np.random.randn(len(dates)) * 0.001,
        'volume': np.random.randint(1000, 10000, len(dates))
    }, index=dates)
    
    df_1m['high'] = df_1m[['open', 'high', 'low', 'close']].max(axis=1)
    df_1m['low'] = df_1m[['open', 'high', 'low', 'close']].min(axis=1)
    
    print(f"\nData shape: {df_1m.shape}")
    print(f"Date range: {df_1m.index.min()} to {df_1m.index.max()}")
    
    # Step 1: Prepare data
    print("\n" + "=" * 80)
    print("Step 1: Data Preparation")
    print("=" * 80)
    
    preparator = MultiTimeframeDataPreparator()
    
    X_train, X_val, y_train_conf, y_train_bias, y_val_conf, y_val_bias = preparator.prepare_training_data(
        df_1m, 
        validation_split=0.2
    )
    
    # Step 2: Build model
    print("\n" + "=" * 80)
    print("Step 2: Model Architecture")
    print("=" * 80)
    
    input_shapes = {tf: X_train[tf].shape[1:] for tf in X_train.keys()}
    print("\nInput shapes:")
    for tf, shape in input_shapes.items():
        print(f"  {tf}: {shape}")
    
    model = MultiTimeframeConfluenceModel(
        input_shapes=input_shapes,
        d_model=128,
        num_heads=4,
        ff_dim=256,
        num_transformer_blocks=2,
        dropout_rate=0.2
    )
    
    model.build_model()
    model.compile_model(learning_rate=0.001)
    
    print("\nModel Summary:")
    model.get_model_summary()
    
    # Step 3: Train
    print("\n" + "=" * 80)
    print("Step 3: Training")
    print("=" * 80)
    
    trainer = ConfluenceModelTrainer(
        model=model,
        model_save_path='/home/claude/mtf_confluence_model_best.keras'
    )
    
    history = trainer.train(
        X_train=X_train,
        y_train_confluence=y_train_conf,
        y_train_bias=y_train_bias,
        X_val=X_val,
        y_val_confluence=y_val_conf,
        y_val_bias=y_val_bias,
        epochs=50,
        batch_size=64,
        patience=10
    )
    
    # Step 4: Evaluate
    print("\n" + "=" * 80)
    print("Step 4: Final Evaluation")
    print("=" * 80)
    
    results = trainer.evaluate(X_val, y_val_conf, y_val_bias)
    
    print("\nTraining complete!")
    print(f"Model saved to: {trainer.model_save_path}")
    
    # Save scalers
    import pickle
    scaler_path = '/home/claude/mtf_confluence_scalers.pkl'
    with open(scaler_path, 'wb') as f:
        pickle.dump(preparator.scalers, f)
    print(f"Scalers saved to: {scaler_path}")
    
    return trainer, preparator


if __name__ == "__main__":
    trainer, preparator = main()
