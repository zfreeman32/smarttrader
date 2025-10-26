"""
Real-Time Regime Inference Module
==================================

Provides real-time market regime classification for live trading.
Optimized for low-latency (<50ms) predictions.

Features:
- Real-time regime prediction
- Confidence scoring
- Regime transition detection
- Historical regime tracking
- Performance monitoring
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import time
from collections import deque

from regime_classifier import (
    RegimeClassifierModel,
    RegimeFeatureEngineering
)


class RealtimeRegimePredictor:
    """
    Real-time regime prediction with efficient feature computation
    and regime transition tracking.
    """
    
    def __init__(self, 
                 model_path: str,
                 lookback_period: int = 100,
                 min_confidence: float = 0.65):
        """
        Initialize real-time regime predictor.
        
        Args:
            model_path: Path to trained model file
            lookback_period: Number of bars needed for prediction
            min_confidence: Minimum confidence threshold
        """
        self.model_path = model_path
        self.lookback_period = lookback_period
        self.min_confidence = min_confidence
        
        # Load model
        print(f"Loading regime classifier from {model_path}...")
        self.model = RegimeClassifierModel()
        self.model.load(model_path)
        
        # Feature engineering
        self.feature_engineer = RegimeFeatureEngineering()
        
        # Regime names
        self.regime_names = {
            0: 'Strong Uptrend',
            1: 'Weak Uptrend',
            2: 'Ranging',
            3: 'Weak Downtrend',
            4: 'Strong Downtrend',
            5: 'High Volatility',
            6: 'Low Volatility'
        }
        
        # Regime tracking
        self.current_regime: Optional[int] = None
        self.current_confidence: float = 0.0
        self.regime_history: deque = deque(maxlen=1000)
        self.last_prediction_time: Optional[datetime] = None
        
        # Performance tracking
        self.prediction_times: deque = deque(maxlen=100)
        
        print("Regime predictor initialized successfully")
        print(f"Lookback period: {lookback_period} bars")
        print(f"Minimum confidence: {min_confidence:.2%}")
    
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare features for regime prediction.
        
        Args:
            df: DataFrame with OHLCV data (at least lookback_period + 200 rows)
            
        Returns:
            DataFrame with engineered features
        """
        if len(df) < self.lookback_period + 200:
            raise ValueError(
                f"Need at least {self.lookback_period + 200} bars for feature calculation. "
                f"Got {len(df)} bars."
            )
        
        # Engineer features
        df_features = self.feature_engineer.engineer_features(df)
        
        return df_features
    
    def predict(self, df: pd.DataFrame, return_probabilities: bool = False) -> Dict:
        """
        Predict current market regime.
        
        Args:
            df: DataFrame with recent OHLCV data (>= lookback_period + 200 bars)
            return_probabilities: If True, return probabilities for all regimes
            
        Returns:
            Dictionary with regime prediction and metadata
        """
        start_time = time.time()
        
        # Prepare features
        df_features = self.prepare_features(df)
        
        # Drop NaN and get last lookback_period bars
        df_clean = df_features.dropna()
        
        if len(df_clean) < self.lookback_period:
            raise ValueError(
                f"After feature engineering and NaN removal, need at least "
                f"{self.lookback_period} bars. Got {len(df_clean)} bars."
            )
        
        # Extract feature columns (use same columns as training)
        feature_cols = self.model.feature_columns
        
        # Get recent data
        recent_data = df_clean[feature_cols].iloc[-self.lookback_period:].values
        
        # Scale features
        recent_scaled = self.model.scaler.transform(recent_data)
        
        # Create sequence (add batch dimension)
        sequence = np.expand_dims(recent_scaled, axis=0)
        
        # Predict
        regime_class, regime_proba = self.model.predict(sequence)
        
        # Extract results
        regime = int(regime_class[0])
        confidence = float(regime_proba[0][regime])
        
        # Update tracking
        self.current_regime = regime
        self.current_confidence = confidence
        self.last_prediction_time = datetime.now()
        
        # Store in history
        self.regime_history.append({
            'timestamp': self.last_prediction_time,
            'regime': regime,
            'confidence': confidence
        })
        
        # Track prediction time
        prediction_time = (time.time() - start_time) * 1000  # ms
        self.prediction_times.append(prediction_time)
        
        # Build result
        result = {
            'regime': regime,
            'regime_name': self.regime_names[regime],
            'confidence': confidence,
            'is_confident': confidence >= self.min_confidence,
            'timestamp': self.last_prediction_time,
            'prediction_time_ms': prediction_time
        }
        
        if return_probabilities:
            result['probabilities'] = {
                self.regime_names[i]: float(regime_proba[0][i])
                for i in range(len(self.regime_names))
            }
        
        return result
    
    def get_regime_trend(self, lookback: int = 10) -> Dict:
        """
        Analyze recent regime transitions.
        
        Args:
            lookback: Number of recent predictions to analyze
            
        Returns:
            Dictionary with regime trend analysis
        """
        if len(self.regime_history) < lookback:
            return {
                'status': 'insufficient_data',
                'available': len(self.regime_history),
                'required': lookback
            }
        
        # Get recent regimes
        recent = list(self.regime_history)[-lookback:]
        regimes = [r['regime'] for r in recent]
        
        # Count regime changes
        changes = sum(1 for i in range(1, len(regimes)) if regimes[i] != regimes[i-1])
        
        # Most common regime
        from collections import Counter
        regime_counts = Counter(regimes)
        dominant_regime = regime_counts.most_common(1)[0][0]
        
        # Stability score (0-1, higher = more stable)
        stability = 1.0 - (changes / (lookback - 1))
        
        return {
            'status': 'ok',
            'dominant_regime': self.regime_names[dominant_regime],
            'regime_changes': changes,
            'stability_score': stability,
            'is_stable': stability >= 0.7,  # 70% stability threshold
            'recent_regimes': [self.regime_names[r] for r in regimes]
        }
    
    def is_regime_suitable_for_pattern(self, pattern_type: str) -> bool:
        """
        Check if current regime is suitable for given pattern type.
        
        Args:
            pattern_type: Type of ICT pattern (fvg, order_block, liquidity_sweep)
            
        Returns:
            True if regime is suitable for pattern
        """
        if self.current_regime is None:
            return False
        
        if self.current_confidence < self.min_confidence:
            return False
        
        # Pattern-regime compatibility rules
        pattern_rules = {
            'order_block': {
                'suitable_regimes': [0, 1, 3, 4],  # Any trend
                'min_confidence': 0.70
            },
            'fvg': {
                'suitable_regimes': [0, 4],  # Strong trends only
                'min_confidence': 0.75
            },
            'fvg_fill': {
                'suitable_regimes': [0, 1, 3, 4],  # Trending markets
                'min_confidence': 0.70
            },
            'liquidity_sweep': {
                'suitable_regimes': [1, 2, 3],  # Weak trends and ranging
                'min_confidence': 0.65
            },
            'breakout': {
                'unsuitable_regimes': [5],  # Avoid high volatility
                'min_confidence': 0.65
            },
            'structure_break': {
                'suitable_regimes': [0, 1, 3, 4],  # Trending
                'min_confidence': 0.70
            }
        }
        
        # Get rules for pattern
        rules = pattern_rules.get(pattern_type.lower())
        if rules is None:
            # Unknown pattern, allow by default
            return True
        
        # Check confidence
        if self.current_confidence < rules.get('min_confidence', self.min_confidence):
            return False
        
        # Check suitable regimes
        if 'suitable_regimes' in rules:
            return self.current_regime in rules['suitable_regimes']
        
        # Check unsuitable regimes
        if 'unsuitable_regimes' in rules:
            return self.current_regime not in rules['unsuitable_regimes']
        
        return True
    
    def get_trading_bias(self) -> Dict:
        """
        Get directional trading bias based on current regime.
        
        Returns:
            Dictionary with trading bias information
        """
        if self.current_regime is None:
            return {
                'bias': 'neutral',
                'strength': 0.0,
                'description': 'No regime prediction available'
            }
        
        # Regime to bias mapping
        regime_bias = {
            0: ('bullish', 1.0, 'Strong uptrend - favor longs'),
            1: ('bullish', 0.6, 'Weak uptrend - cautious longs'),
            2: ('neutral', 0.0, 'Ranging - mean reversion'),
            3: ('bearish', 0.6, 'Weak downtrend - cautious shorts'),
            4: ('bearish', 1.0, 'Strong downtrend - favor shorts'),
            5: ('neutral', 0.0, 'High volatility - wait for clarity'),
            6: ('neutral', 0.0, 'Low volatility - potential breakout')
        }
        
        bias, strength, description = regime_bias[self.current_regime]
        
        # Adjust strength by confidence
        adjusted_strength = strength * self.current_confidence
        
        return {
            'bias': bias,
            'strength': adjusted_strength,
            'confidence': self.current_confidence,
            'regime': self.regime_names[self.current_regime],
            'description': description
        }
    
    def get_performance_stats(self) -> Dict:
        """
        Get performance statistics of the predictor.
        
        Returns:
            Dictionary with performance metrics
        """
        if not self.prediction_times:
            return {
                'status': 'no_predictions',
                'message': 'No predictions made yet'
            }
        
        times = list(self.prediction_times)
        
        return {
            'status': 'ok',
            'total_predictions': len(self.regime_history),
            'avg_prediction_time_ms': np.mean(times),
            'max_prediction_time_ms': np.max(times),
            'min_prediction_time_ms': np.min(times),
            'std_prediction_time_ms': np.std(times),
            'p95_prediction_time_ms': np.percentile(times, 95),
            'within_50ms_target': np.mean([t < 50 for t in times]) * 100
        }
    
    def print_status(self):
        """Print current regime status."""
        if self.current_regime is None:
            print("No prediction available yet")
            return
        
        print("\n" + "="*60)
        print("CURRENT MARKET REGIME")
        print("="*60)
        print(f"Regime:     {self.regime_names[self.current_regime]}")
        print(f"Confidence: {self.current_confidence:.2%}")
        print(f"Status:     {'✅ Confident' if self.current_confidence >= self.min_confidence else '⚠️  Low confidence'}")
        print(f"Timestamp:  {self.last_prediction_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Trading bias
        bias = self.get_trading_bias()
        print(f"\nTrading Bias: {bias['bias'].upper()} (strength: {bias['strength']:.2f})")
        print(f"Description:  {bias['description']}")
        
        # Performance
        perf = self.get_performance_stats()
        if perf['status'] == 'ok':
            print(f"\nPrediction Time: {perf['avg_prediction_time_ms']:.1f}ms (avg)")
            print(f"Within 50ms:     {perf['within_50ms_target']:.1f}%")
        
        # Regime trend
        trend = self.get_regime_trend()
        if trend['status'] == 'ok':
            print(f"\nRegime Stability: {trend['stability_score']:.2%}")
            print(f"Recent Changes:   {trend['regime_changes']}")
        
        print("="*60)


def example_usage():
    """
    Example of using the RealtimeRegimePredictor.
    """
    print("="*80)
    print("REAL-TIME REGIME PREDICTION - EXAMPLE")
    print("="*80)
    
    # Initialize predictor
    predictor = RealtimeRegimePredictor(
        model_path='/mnt/user-data/outputs/regime_classifier.keras',
        lookback_period=100,
        min_confidence=0.70
    )
    
    # Simulate getting live data
    print("\n1. Loading recent market data...")
    # In production, this would be from your data feed
    # df_live = get_live_eurusd_data(bars=500)
    
    # For example purposes, create dummy data
    print("   (Using dummy data for demonstration)")
    dates = pd.date_range(end=datetime.now(), periods=500, freq='1min')
    df_live = pd.DataFrame({
        'timestamp': dates,
        'open': np.random.randn(500).cumsum() + 1.10,
        'high': np.random.randn(500).cumsum() + 1.101,
        'low': np.random.randn(500).cumsum() + 1.099,
        'close': np.random.randn(500).cumsum() + 1.10,
        'volume': np.random.randint(1000, 5000, 500)
    })
    df_live = df_live.set_index('timestamp')
    
    # Make prediction
    print("\n2. Predicting current regime...")
    result = predictor.predict(df_live, return_probabilities=True)
    
    print(f"\n   Regime: {result['regime_name']}")
    print(f"   Confidence: {result['confidence']:.2%}")
    print(f"   Prediction time: {result['prediction_time_ms']:.1f}ms")
    
    if result.get('probabilities'):
        print("\n   All probabilities:")
        for regime, prob in result['probabilities'].items():
            print(f"      {regime:20s}: {prob:.2%}")
    
    # Check pattern suitability
    print("\n3. Checking pattern suitability...")
    patterns = ['order_block', 'fvg', 'liquidity_sweep', 'breakout']
    
    for pattern in patterns:
        suitable = predictor.is_regime_suitable_for_pattern(pattern)
        status = "✅ YES" if suitable else "❌ NO"
        print(f"   {pattern:20s}: {status}")
    
    # Get trading bias
    print("\n4. Getting trading bias...")
    bias = predictor.get_trading_bias()
    print(f"   Direction: {bias['bias'].upper()}")
    print(f"   Strength: {bias['strength']:.2f}")
    print(f"   Note: {bias['description']}")
    
    # Print full status
    print("\n5. Full status:")
    predictor.print_status()
    
    print("\n" + "="*80)
    print("Example complete!")
    print("="*80)


if __name__ == "__main__":
    # Run example
    try:
        example_usage()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("Note: This example requires a trained model at:")
        print("      /mnt/user-data/outputs/regime_classifier.keras")
        print("\nTrain a model first using:")
        print("      python train_regime_classifier.py --data_path eurusd_1min.csv")
