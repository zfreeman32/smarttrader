"""
Market Regime Classification Model
===================================

Classifies market state into 7 distinct regimes using multi-timeframe analysis.
This model helps the trading system adapt strategy selection based on current market conditions.

Regime Classes:
    0: Strong Uptrend
    1: Weak Uptrend
    2: Ranging (consolidation)
    3: Weak Downtrend
    4: Strong Downtrend
    5: High Volatility Breakout
    6: Low Volatility Compression

Architecture: Stacked LSTM with multi-timeframe input
Target Performance: >75% accuracy
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks, optimizers
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple, List, Optional
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)


class RegimeFeatureEngineering:
    """
    Feature engineering for market regime classification.
    Computes multi-timeframe indicators for regime detection.
    """
    
    def __init__(self, timeframes: List[str] = ['1min', '5min', '15min', '1H', '4H', '1D']):
        """
        Initialize feature engineering pipeline.
        
        Args:
            timeframes: List of timeframes to analyze
        """
        self.timeframes = timeframes
        self.scalers = {}
        
    def calculate_adx(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Calculate Average Directional Index (ADX) for trend strength."""
        high = df['high']
        low = df['low']
        close = df['close']
        
        # True Range
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Directional Movement
        up_move = high - high.shift(1)
        down_move = low.shift(1) - low
        
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        
        # Smoothed indicators
        atr = tr.rolling(window=period).mean()
        plus_di = 100 * pd.Series(plus_dm).rolling(window=period).mean() / atr
        minus_di = 100 * pd.Series(minus_dm).rolling(window=period).mean() / atr
        
        # ADX calculation
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(window=period).mean()
        
        df[f'adx_{period}'] = adx
        df[f'plus_di_{period}'] = plus_di
        df[f'minus_di_{period}'] = minus_di
        
        return df
    
    def calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Calculate Average True Range (ATR) for volatility."""
        high = df['high']
        low = df['low']
        close = df['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        atr = tr.rolling(window=period).mean()
        df[f'atr_{period}'] = atr
        
        # ATR as percentage of price
        df[f'atr_pct_{period}'] = (atr / close) * 100
        
        return df
    
    def calculate_bollinger_bands(self, df: pd.DataFrame, period: int = 20, std_dev: float = 2.0) -> pd.DataFrame:
        """Calculate Bollinger Bands for volatility measurement."""
        close = df['close']
        
        sma = close.rolling(window=period).mean()
        std = close.rolling(window=period).std()
        
        df[f'bb_upper_{period}'] = sma + (std_dev * std)
        df[f'bb_lower_{period}'] = sma - (std_dev * std)
        df[f'bb_mid_{period}'] = sma
        df[f'bb_width_{period}'] = (df[f'bb_upper_{period}'] - df[f'bb_lower_{period}']) / sma * 100
        
        # Position within bands (0-1)
        df[f'bb_position_{period}'] = (close - df[f'bb_lower_{period}']) / (df[f'bb_upper_{period}'] - df[f'bb_lower_{period}'])
        
        return df
    
    def calculate_roc(self, df: pd.DataFrame, periods: List[int] = [5, 10, 20, 50]) -> pd.DataFrame:
        """Calculate Rate of Change across multiple periods."""
        close = df['close']
        
        for period in periods:
            df[f'roc_{period}'] = ((close - close.shift(period)) / close.shift(period)) * 100
            
        return df
    
    def calculate_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate volume-based features."""
        volume = df['volume']
        close = df['close']
        
        # Volume moving averages
        df['volume_ma_20'] = volume.rolling(window=20).mean()
        df['volume_ma_50'] = volume.rolling(window=50).mean()
        
        # Volume ratio (current vs average)
        df['volume_ratio'] = volume / df['volume_ma_20']
        
        # On-Balance Volume
        df['obv'] = (np.sign(close.diff()) * volume).fillna(0).cumsum()
        df['obv_ma_20'] = df['obv'].rolling(window=20).mean()
        
        return df
    
    def calculate_trend_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate trend-related features."""
        close = df['close']
        
        # EMAs
        for period in [10, 20, 50, 100, 200]:
            df[f'ema_{period}'] = close.ewm(span=period, adjust=False).mean()
            df[f'price_to_ema_{period}'] = (close - df[f'ema_{period}']) / df[f'ema_{period}'] * 100
        
        # EMA crossover signals
        df['ema_10_20_cross'] = np.where(df['ema_10'] > df['ema_20'], 1, -1)
        df['ema_20_50_cross'] = np.where(df['ema_20'] > df['ema_50'], 1, -1)
        df['ema_50_200_cross'] = np.where(df['ema_50'] > df['ema_200'], 1, -1)
        
        # Price momentum
        df['momentum_10'] = close - close.shift(10)
        df['momentum_20'] = close - close.shift(20)
        
        return df
    
    def calculate_swing_features(self, df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """Calculate swing high/low features."""
        high = df['high']
        low = df['low']
        
        df['swing_high'] = high.rolling(window=period).max()
        df['swing_low'] = low.rolling(window=period).min()
        df['swing_range'] = df['swing_high'] - df['swing_low']
        df['price_in_range'] = (df['close'] - df['swing_low']) / df['swing_range']
        
        return df
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all regime classification features.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with engineered features
        """
        df = df.copy()
        
        # Ensure required columns
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"DataFrame must contain columns: {required_cols}")
        
        print("Calculating regime features...")
        
        # Trend strength indicators
        df = self.calculate_adx(df, period=14)
        df = self.calculate_adx(df, period=20)
        
        # Volatility indicators
        df = self.calculate_atr(df, period=14)
        df = self.calculate_atr(df, period=20)
        df = self.calculate_bollinger_bands(df, period=20)
        
        # Momentum indicators
        df = self.calculate_roc(df, periods=[5, 10, 20, 50])
        
        # Volume features
        df = self.calculate_volume_features(df)
        
        # Trend features
        df = self.calculate_trend_features(df)
        
        # Swing features
        df = self.calculate_swing_features(df, period=20)
        
        print(f"Feature engineering complete. Total features: {len(df.columns)}")
        
        return df


class RegimeLabeler:
    """
    Automatic labeling of market regimes based on technical indicators.
    """
    
    def __init__(self, 
                 adx_trend_threshold: float = 25.0,
                 adx_ranging_threshold: float = 20.0,
                 atr_high_vol_percentile: float = 75.0,
                 atr_low_vol_percentile: float = 25.0):
        """
        Initialize regime labeling parameters.
        
        Args:
            adx_trend_threshold: ADX value above which market is trending
            adx_ranging_threshold: ADX value below which market is ranging
            atr_high_vol_percentile: ATR percentile for high volatility
            atr_low_vol_percentile: ATR percentile for low volatility
        """
        self.adx_trend = adx_trend_threshold
        self.adx_ranging = adx_ranging_threshold
        self.atr_high_pct = atr_high_vol_percentile
        self.atr_low_pct = atr_low_vol_percentile
        
    def label_regime(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Label each bar with its market regime.
        
        Regime Classification Logic:
        - Strong Uptrend (0): ADX > 25, PLUS_DI > MINUS_DI, ROC > 0.5
        - Weak Uptrend (1): ADX 15-25, PLUS_DI > MINUS_DI
        - Ranging (2): ADX < 20
        - Weak Downtrend (3): ADX 15-25, MINUS_DI > PLUS_DI
        - Strong Downtrend (4): ADX > 25, MINUS_DI > PLUS_DI, ROC < -0.5
        - High Volatility (5): ATR > 75th percentile
        - Low Volatility (6): ATR < 25th percentile
        
        Args:
            df: DataFrame with calculated features
            
        Returns:
            DataFrame with 'regime' column
        """
        df = df.copy()
        
        # Calculate ATR percentiles
        atr_high_threshold = df['atr_pct_14'].quantile(self.atr_high_pct / 100)
        atr_low_threshold = df['atr_pct_14'].quantile(self.atr_low_pct / 100)
        
        print(f"ATR High Vol Threshold: {atr_high_threshold:.4f}%")
        print(f"ATR Low Vol Threshold: {atr_low_threshold:.4f}%")
        
        # Initialize regime column
        df['regime'] = 2  # Default to ranging
        
        # Volatility-based regimes (highest priority)
        high_vol_mask = df['atr_pct_14'] > atr_high_threshold
        low_vol_mask = df['atr_pct_14'] < atr_low_threshold
        
        df.loc[high_vol_mask, 'regime'] = 5  # High Volatility
        df.loc[low_vol_mask, 'regime'] = 6   # Low Volatility
        
        # Trend-based regimes (for normal volatility periods)
        normal_vol_mask = ~high_vol_mask & ~low_vol_mask
        
        # Strong Uptrend
        strong_up_mask = (
            normal_vol_mask &
            (df['adx_14'] > self.adx_trend) &
            (df['plus_di_14'] > df['minus_di_14']) &
            (df['roc_20'] > 0.5)
        )
        df.loc[strong_up_mask, 'regime'] = 0
        
        # Weak Uptrend
        weak_up_mask = (
            normal_vol_mask &
            (df['adx_14'] >= 15) &
            (df['adx_14'] <= self.adx_trend) &
            (df['plus_di_14'] > df['minus_di_14'])
        )
        df.loc[weak_up_mask, 'regime'] = 1
        
        # Strong Downtrend
        strong_down_mask = (
            normal_vol_mask &
            (df['adx_14'] > self.adx_trend) &
            (df['minus_di_14'] > df['plus_di_14']) &
            (df['roc_20'] < -0.5)
        )
        df.loc[strong_down_mask, 'regime'] = 4
        
        # Weak Downtrend
        weak_down_mask = (
            normal_vol_mask &
            (df['adx_14'] >= 15) &
            (df['adx_14'] <= self.adx_trend) &
            (df['minus_di_14'] > df['plus_di_14'])
        )
        df.loc[weak_down_mask, 'regime'] = 3
        
        # Ranging (default for normal volatility with weak trend)
        ranging_mask = (
            normal_vol_mask &
            (df['adx_14'] < self.adx_ranging)
        )
        df.loc[ranging_mask, 'regime'] = 2
        
        # Print regime distribution
        regime_names = {
            0: 'Strong Uptrend',
            1: 'Weak Uptrend',
            2: 'Ranging',
            3: 'Weak Downtrend',
            4: 'Strong Downtrend',
            5: 'High Volatility',
            6: 'Low Volatility'
        }
        
        print("\nRegime Distribution:")
        for regime_id, regime_name in regime_names.items():
            count = (df['regime'] == regime_id).sum()
            pct = (count / len(df)) * 100
            print(f"  {regime_name} ({regime_id}): {count:,} bars ({pct:.2f}%)")
        
        return df


class RegimeClassifierModel:
    """
    LSTM-based model for market regime classification.
    """
    
    def __init__(self,
                 lookback_period: int = 100,
                 lstm_units: List[int] = [128, 64],
                 dropout_rate: float = 0.3,
                 learning_rate: float = 0.001):
        """
        Initialize regime classifier model.
        
        Args:
            lookback_period: Number of bars to look back
            lstm_units: List of LSTM layer sizes
            dropout_rate: Dropout rate for regularization
            learning_rate: Learning rate for optimizer
        """
        self.lookback_period = lookback_period
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = None
        self.history = None
        
    def build_model(self, input_shape: Tuple[int, int], num_classes: int = 7) -> keras.Model:
        """
        Build stacked LSTM architecture for regime classification.
        
        Args:
            input_shape: (timesteps, features)
            num_classes: Number of regime classes
            
        Returns:
            Compiled Keras model
        """
        model = keras.Sequential(name='RegimeClassifier')
        
        # First LSTM layer
        model.add(layers.LSTM(
            units=self.lstm_units[0],
            return_sequences=True,
            input_shape=input_shape,
            name='lstm_1'
        ))
        model.add(layers.BatchNormalization(name='bn_1'))
        model.add(layers.Dropout(self.dropout_rate, name='dropout_1'))
        
        # Second LSTM layer
        if len(self.lstm_units) > 1:
            model.add(layers.LSTM(
                units=self.lstm_units[1],
                return_sequences=False,
                name='lstm_2'
            ))
            model.add(layers.BatchNormalization(name='bn_2'))
            model.add(layers.Dropout(self.dropout_rate, name='dropout_2'))
        
        # Dense layers
        model.add(layers.Dense(64, activation='relu', name='dense_1'))
        model.add(layers.Dropout(self.dropout_rate, name='dropout_3'))
        
        model.add(layers.Dense(32, activation='relu', name='dense_2'))
        
        # Output layer
        model.add(layers.Dense(num_classes, activation='softmax', name='output'))
        
        # Compile model
        model.compile(
            optimizer=optimizers.Adam(learning_rate=self.learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
        )
        
        return model
    
    def prepare_sequences(self, 
                         df: pd.DataFrame, 
                         feature_cols: List[str],
                         target_col: str = 'regime') -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare time series sequences for LSTM training.
        
        Args:
            df: DataFrame with features and labels
            feature_cols: List of feature column names
            target_col: Target column name
            
        Returns:
            Tuple of (X, y) arrays
        """
        # Store feature columns for later use
        self.feature_columns = feature_cols
        
        # Extract and scale features
        X_raw = df[feature_cols].values
        X_scaled = self.scaler.fit_transform(X_raw)
        
        # Extract labels
        y_raw = df[target_col].values
        
        # Create sequences
        X_sequences = []
        y_sequences = []
        
        for i in range(self.lookback_period, len(X_scaled)):
            X_sequences.append(X_scaled[i-self.lookback_period:i])
            y_sequences.append(y_raw[i])
        
        X = np.array(X_sequences)
        y = np.array(y_sequences)
        
        # Convert labels to categorical
        y_categorical = to_categorical(y, num_classes=7)
        
        print(f"Sequence shape: X={X.shape}, y={y_categorical.shape}")
        
        return X, y_categorical
    
    def train(self,
             X_train: np.ndarray,
             y_train: np.ndarray,
             X_val: np.ndarray,
             y_val: np.ndarray,
             epochs: int = 100,
             batch_size: int = 64,
             class_weights: Optional[Dict[int, float]] = None,
             verbose: int = 1) -> keras.callbacks.History:
        """
        Train the regime classifier model.
        
        Args:
            X_train: Training features
            y_train: Training labels (one-hot encoded)
            X_val: Validation features
            y_val: Validation labels (one-hot encoded)
            epochs: Number of training epochs
            batch_size: Batch size
            class_weights: Optional class weights for imbalanced data
            verbose: Verbosity mode
            
        Returns:
            Training history
        """
        # Build model if not already built
        if self.model is None:
            input_shape = (X_train.shape[1], X_train.shape[2])
            self.model = self.build_model(input_shape)
        
        print(self.model.summary())
        
        # Callbacks
        early_stop = callbacks.EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1
        )
        
        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=7,
            min_lr=1e-7,
            verbose=1
        )
        
        # Train model
        print("\nTraining regime classifier...")
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            class_weight=class_weights,
            callbacks=[early_stop, reduce_lr],
            verbose=verbose
        )
        
        return self.history
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """
        Evaluate model performance on test set.
        
        Args:
            X_test: Test features
            y_test: Test labels (one-hot encoded)
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Get predictions
        y_pred_proba = self.model.predict(X_test, verbose=0)
        y_pred = np.argmax(y_pred_proba, axis=1)
        y_true = np.argmax(y_test, axis=1)
        
        # Calculate metrics
        test_loss, test_acc, test_precision, test_recall = self.model.evaluate(
            X_test, y_test, verbose=0
        )
        
        # Classification report
        regime_names = [
            'Strong Up', 'Weak Up', 'Ranging',
            'Weak Down', 'Strong Down', 'High Vol', 'Low Vol'
        ]
        
        print("\n" + "="*70)
        print("CLASSIFICATION REPORT")
        print("="*70)
        print(classification_report(y_true, y_pred, target_names=regime_names))
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        return {
            'test_loss': test_loss,
            'test_accuracy': test_acc,
            'test_precision': test_precision,
            'test_recall': test_recall,
            'confusion_matrix': cm,
            'y_pred': y_pred,
            'y_true': y_true,
            'y_pred_proba': y_pred_proba
        }
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict regime for new data.
        
        Args:
            X: Input features (can be single sequence or batch)
            
        Returns:
            Tuple of (regime_classes, probabilities)
        """
        if len(X.shape) == 2:
            # Single sequence, add batch dimension
            X = np.expand_dims(X, axis=0)
        
        y_pred_proba = self.model.predict(X, verbose=0)
        y_pred = np.argmax(y_pred_proba, axis=1)
        
        return y_pred, y_pred_proba
    
    def save(self, filepath: str):
        """Save model and scaler."""
        self.model.save(filepath)
        print(f"Model saved to {filepath}")
        
        # Save scaler
        scaler_path = filepath.replace('.keras', '_scaler.npy')
        np.save(scaler_path, {
            'mean': self.scaler.mean_,
            'scale': self.scaler.scale_,
            'feature_columns': self.feature_columns
        }, allow_pickle=True)
        print(f"Scaler saved to {scaler_path}")
    
    def load(self, filepath: str):
        """Load model and scaler."""
        self.model = keras.models.load_model(filepath)
        print(f"Model loaded from {filepath}")
        
        # Load scaler
        scaler_path = filepath.replace('.keras', '_scaler.npy')
        scaler_data = np.load(scaler_path, allow_pickle=True).item()
        self.scaler.mean_ = scaler_data['mean']
        self.scaler.scale_ = scaler_data['scale']
        self.feature_columns = scaler_data['feature_columns']
        print(f"Scaler loaded from {scaler_path}")


def plot_training_history(history: keras.callbacks.History):
    """Plot training and validation metrics."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Accuracy
    axes[0, 0].plot(history.history['accuracy'], label='Train')
    axes[0, 0].plot(history.history['val_accuracy'], label='Validation')
    axes[0, 0].set_title('Model Accuracy')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Loss
    axes[0, 1].plot(history.history['loss'], label='Train')
    axes[0, 1].plot(history.history['val_loss'], label='Validation')
    axes[0, 1].set_title('Model Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Precision
    axes[1, 0].plot(history.history['precision'], label='Train')
    axes[1, 0].plot(history.history['val_precision'], label='Validation')
    axes[1, 0].set_title('Model Precision')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Precision')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Recall
    axes[1, 1].plot(history.history['recall'], label='Train')
    axes[1, 1].plot(history.history['val_recall'], label='Validation')
    axes[1, 1].set_title('Model Recall')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Recall')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/mnt/user-data/outputs/regime_training_history.png', dpi=300, bbox_inches='tight')
    print("Training history plot saved")
    plt.close()


def plot_confusion_matrix(cm: np.ndarray, normalize: bool = True):
    """Plot confusion matrix heatmap."""
    regime_names = [
        'Strong\nUp', 'Weak\nUp', 'Ranging',
        'Weak\nDown', 'Strong\nDown', 'High\nVol', 'Low\nVol'
    ]
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2%'
        title = 'Normalized Confusion Matrix'
    else:
        fmt = 'd'
        title = 'Confusion Matrix'
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues',
                xticklabels=regime_names, yticklabels=regime_names,
                cbar_kws={'label': 'Proportion' if normalize else 'Count'})
    plt.title(title, fontsize=14, fontweight='bold')
    plt.ylabel('True Regime', fontsize=12)
    plt.xlabel('Predicted Regime', fontsize=12)
    plt.tight_layout()
    plt.savefig('/mnt/user-data/outputs/regime_confusion_matrix.png', dpi=300, bbox_inches='tight')
    print("Confusion matrix plot saved")
    plt.close()


def plot_regime_distribution(df: pd.DataFrame):
    """Plot distribution of regimes over time."""
    regime_names = {
        0: 'Strong Uptrend',
        1: 'Weak Uptrend',
        2: 'Ranging',
        3: 'Weak Downtrend',
        4: 'Strong Downtrend',
        5: 'High Volatility',
        6: 'Low Volatility'
    }
    
    regime_colors = {
        0: '#00ff00',  # Bright green
        1: '#90ee90',  # Light green
        2: '#ffa500',  # Orange
        3: '#ffcccb',  # Light red
        4: '#ff0000',  # Bright red
        5: '#ff00ff',  # Magenta
        6: '#add8e6'   # Light blue
    }
    
    fig, axes = plt.subplots(2, 1, figsize=(15, 10))
    
    # Time series plot with regime colors
    for regime_id in sorted(regime_names.keys()):
        mask = df['regime'] == regime_id
        axes[0].scatter(df.index[mask], df['close'][mask],
                       c=regime_colors[regime_id], label=regime_names[regime_id],
                       alpha=0.6, s=1)
    
    axes[0].set_title('EURUSD Price with Market Regimes', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Time')
    axes[0].set_ylabel('Price')
    axes[0].legend(loc='best', markerscale=10)
    axes[0].grid(True, alpha=0.3)
    
    # Regime distribution bar chart
    regime_counts = df['regime'].value_counts().sort_index()
    bars = axes[1].bar(range(len(regime_counts)), regime_counts.values,
                       color=[regime_colors[i] for i in regime_counts.index])
    axes[1].set_xticks(range(len(regime_counts)))
    axes[1].set_xticklabels([regime_names[i] for i in regime_counts.index], rotation=45, ha='right')
    axes[1].set_title('Regime Distribution', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('Count')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    # Add percentage labels on bars
    for bar, count in zip(bars, regime_counts.values):
        height = bar.get_height()
        pct = (count / len(df)) * 100
        axes[1].text(bar.get_x() + bar.get_width()/2., height,
                    f'{pct:.1f}%', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('/mnt/user-data/outputs/regime_distribution.png', dpi=300, bbox_inches='tight')
    print("Regime distribution plot saved")
    plt.close()


def calculate_class_weights(y: np.ndarray) -> Dict[int, float]:
    """
    Calculate class weights for imbalanced data.
    
    Args:
        y: Array of class labels
        
    Returns:
        Dictionary of class weights
    """
    unique_classes, class_counts = np.unique(y, return_counts=True)
    total_samples = len(y)
    n_classes = len(unique_classes)
    
    class_weights = {}
    for cls, count in zip(unique_classes, class_counts):
        weight = total_samples / (n_classes * count)
        class_weights[int(cls)] = weight
    
    print("\nClass Weights:")
    regime_names = {
        0: 'Strong Uptrend',
        1: 'Weak Uptrend',
        2: 'Ranging',
        3: 'Weak Downtrend',
        4: 'Strong Downtrend',
        5: 'High Volatility',
        6: 'Low Volatility'
    }
    
    for cls, weight in class_weights.items():
        print(f"  {regime_names[cls]}: {weight:.4f}")
    
    return class_weights


if __name__ == "__main__":
    """
    Example usage and testing pipeline.
    """
    print("="*70)
    print("MARKET REGIME CLASSIFIER - TRAINING PIPELINE")
    print("="*70)
    
    # This is a template - replace with actual data loading
    print("\nNOTE: This is a template. Replace with actual EURUSD data loading.")
    print("Expected data format: DataFrame with columns ['open', 'high', 'low', 'close', 'volume']")
    print("Recommended: 2+ years of 1-minute EURUSD data")
    
    # Example workflow (uncomment and adapt when you have data):
    """
    # 1. Load data
    df = pd.read_csv('eurusd_1min.csv', parse_dates=['timestamp'], index_col='timestamp')
    
    # 2. Feature engineering
    feature_engineer = RegimeFeatureEngineering()
    df = feature_engineer.engineer_features(df)
    
    # 3. Label regimes
    labeler = RegimeLabeler()
    df = labeler.label_regime(df)
    
    # 4. Plot regime distribution
    plot_regime_distribution(df)
    
    # 5. Select features for training
    feature_cols = [
        'adx_14', 'plus_di_14', 'minus_di_14',
        'atr_pct_14', 'bb_width_20', 'bb_position_20',
        'roc_5', 'roc_10', 'roc_20',
        'volume_ratio', 'obv',
        'price_to_ema_20', 'price_to_ema_50',
        'ema_10_20_cross', 'ema_20_50_cross',
        'momentum_10', 'momentum_20',
        'price_in_range'
    ]
    
    # 6. Prepare sequences
    model = RegimeClassifierModel(lookback_period=100, lstm_units=[128, 64])
    X, y = model.prepare_sequences(df.dropna(), feature_cols)
    
    # 7. Train/val/test split (time-series)
    train_size = int(0.7 * len(X))
    val_size = int(0.15 * len(X))
    
    X_train, y_train = X[:train_size], y[:train_size]
    X_val, y_val = X[train_size:train_size+val_size], y[train_size:train_size+val_size]
    X_test, y_test = X[train_size+val_size:], y[train_size+val_size:]
    
    print(f"Training set: {len(X_train):,} samples")
    print(f"Validation set: {len(X_val):,} samples")
    print(f"Test set: {len(X_test):,} samples")
    
    # 8. Calculate class weights
    y_train_classes = np.argmax(y_train, axis=1)
    class_weights = calculate_class_weights(y_train_classes)
    
    # 9. Train model
    history = model.train(
        X_train, y_train,
        X_val, y_val,
        epochs=100,
        batch_size=64,
        class_weights=class_weights
    )
    
    # 10. Plot training history
    plot_training_history(history)
    
    # 11. Evaluate on test set
    results = model.evaluate(X_test, y_test)
    
    # 12. Plot confusion matrix
    plot_confusion_matrix(results['confusion_matrix'])
    
    # 13. Save model
    model.save('/mnt/user-data/outputs/regime_classifier.keras')
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    print(f"Test Accuracy: {results['test_accuracy']:.4f}")
    print(f"Test Precision: {results['test_precision']:.4f}")
    print(f"Test Recall: {results['test_recall']:.4f}")
    """
