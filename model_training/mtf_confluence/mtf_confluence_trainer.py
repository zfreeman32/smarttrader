"""
Multi-Timeframe Confluence Model - Complete Implementation
==========================================================

Implements:
1. MultiTimeframeDataPreparator - Prepares multi-timeframe data
2. MultiTimeframeConfluenceModel - Transformer-based model architecture
3. ConfluenceModelTrainer - Training orchestration

Architecture: Multi-Stream Transformer with attention across timeframes
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint


class MultiTimeframeDataPreparator:
    """Prepares multi-timeframe data using pre-calculated features from CSV"""
    
    def __init__(self, 
                 timeframes: Dict[str, int] = None,
                 lookback_periods: Dict[str, int] = None):
        """
        Args:
            timeframes: Dict of timeframe names to minutes
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
        
        # Map existing column names to feature categories
        self.feature_mapping = self._create_feature_mapping()
    
    def _create_feature_mapping(self) -> Dict[str, List[str]]:
        """Map existing columns to feature categories"""
        mapping = {
            # Trend indicators
            'trend': [
                'EMA_10', 'EMA_20', 'EMA_50', 'EMA_200',
                'DEMA', 'TEMA', 'TRIMA',
                'MA_10', 'MA_50', 'MA_200',
                'KAMA', 'T3', 'WMA', 'SMA',
                'HT_TRENDLINE',
                'MAMA', 'FAMA'
            ],
            
            # Momentum indicators
            'momentum': [
                'RSI', 'STOCH_K', 'STOCH_D', 'STOCHF_K', 'STOCHF_D',
                'STOCHRSI_K', 'STOCHRSI_D',
                'MACD', 'MACDSIGNAL', 'MACDHIST',
                'CCI', 'CMO', 'ROC', 'ROCP', 'ROCR', 'ROCR100',
                'MOM', 'APO', 'PPO', 'TRIX',
                'WILLR', 'ULTOSC', 'BOP'
            ],
            
            # Volatility indicators
            'volatility': [
                'UPPERBAND', 'MIDDLEBAND', 'LOWERBAND',
                'ATR', 'NATR', 'TRANGE',
                'KELTNER_UPPER', 'KELTNER_LOWER',
                'rolling_std'
            ],
            
            # Volume indicators
            'volume': [
                'AD', 'ADOSC', 'OBV',
                'MFI', 'ADOSC',
                'EFI'
            ],
            
            # Trend strength
            'trend_strength': [
                'ADX', 'ADXR', 'DX',
                'PLUS_DI', 'MINUS_DI',
                'PLUS_DM', 'MINUS_DM',
                'AROON_UP', 'AROON_DOWN', 'AROONOSC'
            ],
            
            # Support/Resistance
            'support_resistance': [
                'SAR', 'SAREXT',
                'SUPER_TREND',
                'DONCHIAN_HIGH', 'DONCHIAN_LOW'
            ],
            
            # Price patterns
            'price_patterns': [
                'AVGPRICE', 'MEDPRICE', 'TYPPRICE', 'WCLPRICE',
                'VWAP', 'HMA'
            ],
            
            # Cycle indicators
            'cycle': [
                'HT_DCPERIOD', 'HT_DCPHASE',
                'HT_PHASOR_INPHASE', 'HT_PHASOR_QUADRATURE',
                'HT_SINE', 'HT_LEADSINE', 'HT_TRENDMODE'
            ],
            
            # Statistical
            'statistical': [
                'rolling_mean', 'z_score'
            ]
        }
        
        return mapping
    
    def select_best_features(self, df: pd.DataFrame, max_features: int = 50) -> List[str]:
        """Select the most important features from available columns"""
        # Priority features (most important for confluence)
        priority_features = [
            # Trend
            'EMA_10', 'EMA_20', 'EMA_50', 'EMA_200',
            'KAMA', 'DEMA',
            
            # Momentum
            'RSI', 'STOCHRSI_K', 'STOCHRSI_D',
            'MACD', 'MACDSIGNAL', 'MACDHIST',
            'CCI', 'CMO', 'WILLR',
            
            # Volatility
            'ATR', 'UPPERBAND', 'MIDDLEBAND', 'LOWERBAND',
            'KELTNER_UPPER', 'KELTNER_LOWER',
            
            # Volume
            'OBV', 'AD', 'MFI', 'ADOSC',
            
            # Trend Strength
            'ADX', 'ADXR',
            'PLUS_DI', 'MINUS_DI',
            'AROON_UP', 'AROON_DOWN',
            
            # Support/Resistance
            'SAR', 'SUPER_TREND',
            'DONCHIAN_HIGH', 'DONCHIAN_LOW',
            
            # Price patterns
            'VWAP', 'HMA', 'AVGPRICE',
            
            # Additional
            'rolling_mean', 'rolling_std', 'z_score',
            'BOP', 'TRIX', 'APO', 'PPO',
            'ROC', 'MOM'
        ]
        
        # Filter to only features that exist in the dataframe
        available_features = [f for f in priority_features if f in df.columns]
        
        # If no priority features found, use all numeric columns except OHLCV
        if len(available_features) == 0:
            print("⚠ No priority features found, using all available numeric columns...")
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            # Exclude basic OHLC if other features exist
            exclude = ['open', 'high', 'low', 'close', 'volume']
            available_features = [col for col in numeric_cols if col.lower() not in exclude]
            
            # If still empty, just use all numeric columns
            if len(available_features) == 0:
                available_features = numeric_cols
        
        # If we have more than max_features, take the first max_features
        selected = available_features[:max_features]
        
        print(f"Selected {len(selected)} features from {len(df.columns)} available columns")
        
        if len(selected) == 0:
            raise ValueError("No features available! Please check your data file has technical indicators.")
        
        return selected
    
    def prepare_timeframe_data(self, df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
        """Prepare data for a single timeframe using existing features"""
        # Create output dataframe
        result = pd.DataFrame(index=df.index)
        
        # Add selected features
        for col in feature_cols:
            if col in df.columns:
                result[col] = df[col]
        
        # Add some derived features for better confluence detection
        
        # EMA crossover signals
        if 'EMA_10' in df.columns and 'EMA_20' in df.columns:
            result['ema_10_20_cross'] = np.where(df['EMA_10'] > df['EMA_20'], 1, -1)
        
        if 'EMA_20' in df.columns and 'EMA_50' in df.columns:
            result['ema_20_50_cross'] = np.where(df['EMA_20'] > df['EMA_50'], 1, -1)
        
        if 'EMA_50' in df.columns and 'EMA_200' in df.columns:
            result['ema_50_200_cross'] = np.where(df['EMA_50'] > df['EMA_200'], 1, -1)
        
        # RSI zones
        if 'RSI' in df.columns:
            result['rsi_overbought'] = np.where(df['RSI'] > 70, 1, 0)
            result['rsi_oversold'] = np.where(df['RSI'] < 30, 1, 0)
        
        # MACD signal
        if 'MACD' in df.columns and 'MACDSIGNAL' in df.columns:
            result['macd_signal'] = np.where(df['MACD'] > df['MACDSIGNAL'], 1, -1)
        
        # ADX trend strength
        if 'ADX' in df.columns:
            result['adx_strong'] = np.where(df['ADX'] > 25, 1, 0)
        
        # Bollinger Band position
        if all(col in df.columns for col in ['UPPERBAND', 'MIDDLEBAND', 'LOWERBAND', 'close']):
            bb_position = (df['close'] - df['LOWERBAND']) / (df['UPPERBAND'] - df['LOWERBAND'] + 1e-10)
            result['bb_position'] = bb_position
        
        # Volume confirmation
        if 'volume' in df.columns:
            vol_sma = df['volume'].rolling(20).mean()
            result['volume_ratio'] = df['volume'] / (vol_sma + 1e-10)
        
        return result
    
    def resample_to_timeframe(self, df_1m: pd.DataFrame, minutes: int, feature_cols: List[str]) -> pd.DataFrame:
        """Resample 1-minute data to higher timeframe"""
        # Resample OHLCV
        ohlcv_cols = ['open', 'high', 'low', 'close', 'volume']
        available_ohlcv = [col for col in ohlcv_cols if col in df_1m.columns]
        
        if not available_ohlcv:
            # Try capital case
            ohlcv_cols_cap = ['Open', 'High', 'Low', 'Close', 'Volume']
            available_ohlcv = [col for col in ohlcv_cols_cap if col in df_1m.columns]
        
        # Resample
        agg_dict = {}
        for col in available_ohlcv:
            lower_col = col.lower()
            if lower_col == 'open':
                agg_dict[col] = 'first'
            elif lower_col == 'high':
                agg_dict[col] = 'max'
            elif lower_col == 'low':
                agg_dict[col] = 'min'
            elif lower_col == 'close':
                agg_dict[col] = 'last'
            elif lower_col == 'volume':
                agg_dict[col] = 'sum'
        
        # For features, use last value
        for col in feature_cols:
            if col in df_1m.columns:
                agg_dict[col] = 'last'
        
        df_resampled = df_1m.resample(f'{minutes}T').agg(agg_dict)
        df_resampled = df_resampled.dropna()
        
        return df_resampled
    
    def create_aligned_dataset(self, df_1m: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Create aligned multi-timeframe dataset from 1-minute data"""
        aligned_data = {}
        
        # Ensure datetime index
        df_1m = df_1m.copy()
        if not isinstance(df_1m.index, pd.DatetimeIndex):
            df_1m.index = pd.to_datetime(df_1m.index)
        
        print("Creating multi-timeframe dataset using existing features...")
        
        # Select features to use
        feature_cols = self.select_best_features(df_1m, max_features=50)
        
        for tf_name, tf_minutes in self.timeframes.items():
            print(f"Processing {tf_name} timeframe...")
            
            # Resample if not 1m
            if tf_minutes == 1:
                df_tf = df_1m.copy()
            else:
                df_tf = self.resample_to_timeframe(df_1m, tf_minutes, feature_cols)
            
            # Prepare features for this timeframe
            df_tf = self.prepare_timeframe_data(df_tf, feature_cols)
            
            # Get all numeric columns
            numeric_cols = df_tf.select_dtypes(include=[np.number]).columns.tolist()
            
            # Create sequences
            lookback = self.lookback_periods[tf_name]
            sequences = []
            
            for i in range(lookback, len(df_tf)):
                seq = df_tf[numeric_cols].iloc[i-lookback:i].values
                sequences.append(seq)
            
            aligned_data[tf_name] = np.array(sequences)
            
            print(f"{tf_name}: {aligned_data[tf_name].shape}")
        
        return aligned_data
    
    def calculate_labels(self, df_1m: pd.DataFrame, forward_window: int = 20) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate confluence score and directional bias labels"""
        df = df_1m.copy()
        
        # Get close price column
        if 'close' in df.columns:
            close_col = 'close'
        elif 'Close' in df.columns:
            close_col = 'Close'
        else:
            raise ValueError("No 'close' or 'Close' column found")
        
        # Calculate future returns
        df['future_return'] = df[close_col].pct_change(forward_window).shift(-forward_window)
        
        # Directional bias: normalized future return
        df['directional_bias'] = np.tanh(df['future_return'] * 100)
        
        # Confluence score: based on consistency across multiple horizons
        df['return_5'] = df[close_col].pct_change(5).shift(-5)
        df['return_10'] = df[close_col].pct_change(10).shift(-10)
        df['return_15'] = df[close_col].pct_change(15).shift(-15)
        df['return_20'] = df[close_col].pct_change(20).shift(-20)
        
        # Calculate agreement
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
                             validation_split: float = 0.2) -> Tuple:
        """Complete data preparation pipeline"""
        print("\n" + "="*80)
        print("DATA PREPARATION USING EXISTING FEATURES")
        print("="*80)
        
        # Create aligned dataset
        aligned_data = self.create_aligned_dataset(df_1m)
        
        # Calculate labels
        print("\nCalculating labels...")
        confluence_scores, directional_bias = self.calculate_labels(df_1m)
        
        # Align lengths
        min_samples = min([data.shape[0] for data in aligned_data.values()] + [len(confluence_scores)])
        
        for tf_name in aligned_data:
            aligned_data[tf_name] = aligned_data[tf_name][-min_samples:]
        
        confluence_scores = confluence_scores[-min_samples:]
        directional_bias = directional_bias[-min_samples:]
        
        # Split
        split_idx = int(len(confluence_scores) * (1 - validation_split))
        
        X_train = {tf: data[:split_idx] for tf, data in aligned_data.items()}
        X_val = {tf: data[split_idx:] for tf, data in aligned_data.items()}
        
        y_train_confluence = confluence_scores[:split_idx]
        y_train_bias = directional_bias[:split_idx]
        
        y_val_confluence = confluence_scores[split_idx:]
        y_val_bias = directional_bias[split_idx:]
        
        # Fit scalers on training data
        print("\nScaling features...")
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
        
        print(f"\n✓ Training samples: {len(y_train_confluence):,}")
        print(f"✓ Validation samples: {len(y_val_confluence):,}")
        
        return X_train, X_val, y_train_confluence, y_train_bias, y_val_confluence, y_val_bias


class MultiTimeframeConfluenceModel:
    """
    Transformer-based Multi-Timeframe Confluence Model
    
    Architecture:
    - Separate LSTM encoders for each timeframe
    - Transformer layers for cross-timeframe attention
    - Dual outputs: confluence score [0-1] and directional bias [-1, +1]
    """
    
    def __init__(self,
                 input_shapes: Dict[str, Tuple[int, int]],
                 d_model: int = 128,
                 num_heads: int = 4,
                 ff_dim: int = 256,
                 num_transformer_blocks: int = 2,
                 dropout_rate: float = 0.2):
        """
        Args:
            input_shapes: Dict of {timeframe: (timesteps, features)} for each timeframe
            d_model: Dimension of transformer model
            num_heads: Number of attention heads
            ff_dim: Feed-forward network dimension
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
        
    def build_transformer_block(self, 
                                x: tf.Tensor, 
                                num_heads: int, 
                                ff_dim: int, 
                                dropout_rate: float,
                                name: str = "transformer") -> tf.Tensor:
        """Build a single transformer block"""
        
        # Multi-head attention
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=self.d_model // num_heads,
            dropout=dropout_rate,
            name=f"{name}_mha"
        )(x, x)
        
        attention_output = layers.Dropout(dropout_rate)(attention_output)
        x = layers.Add()([x, attention_output])
        x = layers.LayerNormalization(epsilon=1e-6)(x)
        
        # Feed-forward network
        ff_output = layers.Dense(ff_dim, activation='relu', name=f"{name}_ff1")(x)
        ff_output = layers.Dropout(dropout_rate)(ff_output)
        ff_output = layers.Dense(self.d_model, name=f"{name}_ff2")(ff_output)
        ff_output = layers.Dropout(dropout_rate)(ff_output)
        
        x = layers.Add()([x, ff_output])
        x = layers.LayerNormalization(epsilon=1e-6)(x)
        
        return x
    
    def build_model(self) -> Model:
        """Build the complete multi-timeframe confluence model"""
        
        print("\n" + "="*80)
        print("BUILDING MULTI-TIMEFRAME CONFLUENCE MODEL")
        print("="*80)
        
        # Create input layers for each timeframe
        inputs = {}
        for tf_name, (timesteps, features) in self.input_shapes.items():
            inputs[tf_name] = layers.Input(
                shape=(timesteps, features),
                name=f"input_{tf_name}"
            )
            print(f"Input {tf_name}: {(timesteps, features)}")
        
        # Encode each timeframe with LSTM
        encoded_streams = []
        for tf_name, input_layer in inputs.items():
            # LSTM encoder
            x = layers.LSTM(
                self.d_model,
                return_sequences=True,
                dropout=self.dropout_rate,
                recurrent_dropout=self.dropout_rate,
                name=f"lstm_{tf_name}"
            )(input_layer)
            
            encoded_streams.append(x)
            print(f"LSTM {tf_name}: {self.d_model} units")
        
        # Concatenate all timeframe streams
        if len(encoded_streams) > 1:
            combined = layers.Concatenate(axis=1)(encoded_streams)
        else:
            combined = encoded_streams[0]
        
        print(f"\nCombined streams shape: {combined.shape}")
        
        # Apply transformer blocks for cross-timeframe attention
        x = combined
        for i in range(self.num_transformer_blocks):
            x = self.build_transformer_block(
                x,
                num_heads=self.num_heads,
                ff_dim=self.ff_dim,
                dropout_rate=self.dropout_rate,
                name=f"transformer_block_{i+1}"
            )
            print(f"Transformer Block {i+1}: {self.num_heads} heads, {self.ff_dim} ff_dim")
        
        # Global average pooling
        x = layers.GlobalAveragePooling1D()(x)
        
        # Dense layers before outputs
        x = layers.Dense(128, activation='relu', name='dense_1')(x)
        x = layers.Dropout(self.dropout_rate)(x)
        x = layers.Dense(64, activation='relu', name='dense_2')(x)
        x = layers.Dropout(self.dropout_rate)(x)
        
        # Dual outputs
        confluence_output = layers.Dense(
            1, 
            activation='sigmoid', 
            name='confluence_score'
        )(x)
        
        bias_output = layers.Dense(
            1, 
            activation='tanh', 
            name='directional_bias'
        )(x)
        
        print("\nOutputs:")
        print("  - Confluence Score: [0, 1]")
        print("  - Directional Bias: [-1, +1]")
        
        # Build model with inputs in sorted order (alphabetical by timeframe name)
        # This ensures consistency with training data preparation
        sorted_timeframes = sorted(inputs.keys())
        sorted_inputs = [inputs[tf] for tf in sorted_timeframes]
        
        self.model = Model(
            inputs=sorted_inputs,
            outputs=[confluence_output, bias_output],
            name='MultiTimeframeConfluenceModel'
        )
        
        print(f"\n✓ Model built with {self.model.count_params():,} parameters")
        
        return self.model
    
    def compile_model(self, learning_rate: float = 0.001):
        """Compile the model with optimizers and loss functions"""
        
        if self.model is None:
            raise ValueError("Model not built yet. Call build_model() first.")
        
        # Custom loss weights
        loss_weights = {
            'confluence_score': 1.0,
            'directional_bias': 0.8
        }
        
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss={
                'confluence_score': 'mse',
                'directional_bias': 'mse'
            },
            loss_weights=loss_weights,
            metrics={
                'confluence_score': ['mae', 'mse'],
                'directional_bias': ['mae', 'mse']
            }
        )
        
        print(f"\n✓ Model compiled with learning rate: {learning_rate}")
    
    def get_model_summary(self):
        """Print detailed model summary"""
        
        if self.model is None:
            raise ValueError("Model not built yet. Call build_model() first.")
        
        print("\n" + "="*80)
        print("MODEL SUMMARY")
        print("="*80)
        
        # Print Keras summary
        self.model.summary()
        
        # Print additional details
        print("\n" + "="*80)
        print("MODEL DETAILS")
        print("="*80)
        
        print(f"\nArchitecture Configuration:")
        print(f"  Transformer Dimension (d_model): {self.d_model}")
        print(f"  Attention Heads: {self.num_heads}")
        print(f"  Feed-Forward Dimension: {self.ff_dim}")
        print(f"  Transformer Blocks: {self.num_transformer_blocks}")
        print(f"  Dropout Rate: {self.dropout_rate}")
        
        print(f"\nInput Shapes:")
        for tf_name, shape in self.input_shapes.items():
            print(f"  {tf_name}: {shape}")
        
        print(f"\nOutput Shapes:")
        print(f"  Confluence Score: (batch_size, 1) - Range: [0, 1]")
        print(f"  Directional Bias: (batch_size, 1) - Range: [-1, +1]")
        
        print(f"\nTotal Parameters: {self.model.count_params():,}")
        
        # Count trainable vs non-trainable
        trainable_count = sum([tf.size(w).numpy() for w in self.model.trainable_weights])
        non_trainable_count = sum([tf.size(w).numpy() for w in self.model.non_trainable_weights])
        
        print(f"  Trainable: {trainable_count:,}")
        print(f"  Non-trainable: {non_trainable_count:,}")
        
        print("\n" + "="*80)
    
    def get_model(self) -> Model:
        """Get the Keras model"""
        return self.model


class ConfluenceModelTrainer:
    """Handles training of the Multi-Timeframe Confluence Model"""
    
    def __init__(self, 
                 model: MultiTimeframeConfluenceModel,
                 output_dir: str = './mtf_confluence_output/',
                 model_save_path: Optional[str] = None):
        """
        Args:
            model: MultiTimeframeConfluenceModel instance
            output_dir: Directory to save outputs
            model_save_path: Optional custom path to save the best model
        """
        self.model = model
        self.output_dir = output_dir
        self.model_save_path = model_save_path
        self.history = None
        
    def create_callbacks(self, 
                        patience: int = 15,
                        min_delta: float = 0.0001) -> List:
        """Create training callbacks"""
        
        import os
        os.makedirs(os.path.join(self.output_dir, 'models'), exist_ok=True)
        
        # Use custom save path if provided, otherwise use default
        if self.model_save_path:
            save_path = self.model_save_path
        else:
            save_path = os.path.join(self.output_dir, 'models', 'confluence_model_best.keras')
        
        callbacks = [
            # Early stopping
            EarlyStopping(
                monitor='val_loss',
                patience=patience,
                min_delta=min_delta,
                restore_best_weights=True,
                verbose=1
            ),
            
            # Learning rate reduction
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=patience // 2,
                min_lr=1e-7,
                verbose=1
            ),
            
            # Model checkpoint
            ModelCheckpoint(
                filepath=save_path,
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            )
        ]
        
        return callbacks
    
    def train(self,
             X_train: Dict[str, np.ndarray],
             y_train_confluence: np.ndarray,
             y_train_bias: np.ndarray,
             X_val: Dict[str, np.ndarray],
             y_val_confluence: np.ndarray,
             y_val_bias: np.ndarray,
             epochs: int = 100,
             batch_size: int = 64,
             patience: int = 15) -> keras.callbacks.History:
        """
        Train the model
        
        Args:
            X_train: Dict of training data for each timeframe
            y_train_confluence: Training confluence scores
            y_train_bias: Training directional bias
            X_val: Dict of validation data for each timeframe
            y_val_confluence: Validation confluence scores
            y_val_bias: Validation directional bias
            epochs: Maximum number of epochs
            batch_size: Batch size
            patience: Early stopping patience
            
        Returns:
            Training history
        """
        
        print("\n" + "="*80)
        print("TRAINING MULTI-TIMEFRAME CONFLUENCE MODEL")
        print("="*80)
        print(f"\nConfiguration:")
        print(f"  Epochs: {epochs}")
        print(f"  Batch size: {batch_size}")
        print(f"  Patience: {patience}")
        print(f"  Training samples: {len(y_train_confluence):,}")
        print(f"  Validation samples: {len(y_val_confluence):,}")
        
        # Prepare inputs (list format for multi-input model)
        X_train_list = [X_train[tf] for tf in sorted(X_train.keys())]
        X_val_list = [X_val[tf] for tf in sorted(X_val.keys())]
        
        # Prepare outputs (dict format for named outputs)
        y_train_dict = {
            'confluence_score': y_train_confluence,
            'directional_bias': y_train_bias
        }
        
        y_val_dict = {
            'confluence_score': y_val_confluence,
            'directional_bias': y_val_bias
        }
        
        # Create callbacks
        callbacks = self.create_callbacks(patience=patience)
        
        # Train
        print("\nStarting training...")
        print("="*80)
        
        self.history = self.model.get_model().fit(
            X_train_list,
            y_train_dict,
            validation_data=(X_val_list, y_val_dict),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        print("\n" + "="*80)
        print("✓ TRAINING COMPLETE")
        print("="*80)
        
        # Save training history
        import os
        history_df = pd.DataFrame(self.history.history)
        history_path = os.path.join(self.output_dir, 'reports', 'training_history.csv')
        os.makedirs(os.path.dirname(history_path), exist_ok=True)
        history_df.to_csv(history_path, index=False)
        print(f"✓ Training history saved to: {history_path}")
        
        return self.history
    
    def evaluate(self,
                X_test: Dict[str, np.ndarray],
                y_test_confluence: np.ndarray,
                y_test_bias: np.ndarray) -> Dict:
        """Evaluate model on test set"""
        
        # Prepare inputs
        X_test_list = [X_test[tf] for tf in sorted(X_test.keys())]
        
        # Evaluate
        results = self.model.get_model().evaluate(
            X_test_list,
            {
                'confluence_score': y_test_confluence,
                'directional_bias': y_test_bias
            },
            verbose=0
        )
        
        # Extract metrics
        metrics = {}
        for i, metric_name in enumerate(self.model.get_model().metrics_names):
            metrics[metric_name] = results[i]
        
        return metrics
    
    def predict(self,
               X: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions
        
        Args:
            X: Dict of data for each timeframe
            
        Returns:
            Tuple of (confluence_scores, directional_bias)
        """
        # Prepare inputs
        X_list = [X[tf] for tf in sorted(X.keys())]
        
        # Predict
        predictions = self.model.get_model().predict(X_list, verbose=0)
        
        confluence_scores = predictions[0].flatten()
        directional_bias = predictions[1].flatten()
        
        return confluence_scores, directional_bias


# Export classes
__all__ = [
    'MultiTimeframeDataPreparator',
    'MultiTimeframeConfluenceModel',
    'ConfluenceModelTrainer'
]