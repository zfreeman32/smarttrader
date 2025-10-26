"""
Multi-Timeframe Confluence Real-Time Inference Module
=====================================================

Production-ready inference for generating confluence scores and directional bias
in real-time trading scenarios.

Usage:
    analyzer = ConfluenceAnalyzer(model_path='mtf_confluence_model_best.keras')
    score, bias = analyzer.analyze(current_market_data)
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from typing import Dict, Tuple, Optional
import pickle
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


class ConfluenceAnalyzer:
    """Real-time multi-timeframe confluence analyzer for trading"""
    
    def __init__(self, 
                 model_path: str,
                 scaler_path: str,
                 timeframes: Dict[str, int] = None,
                 lookback_periods: Dict[str, int] = None):
        """
        Initialize confluence analyzer with trained model
        
        Args:
            model_path: Path to trained keras model
            scaler_path: Path to saved scalers
            timeframes: Dict of timeframe names to minutes
            lookback_periods: Required bars per timeframe
        """
        # Load model
        print(f"Loading confluence model from: {model_path}")
        self.model = keras.models.load_model(model_path)
        
        # Load scalers
        print(f"Loading scalers from: {scaler_path}")
        with open(scaler_path, 'rb') as f:
            self.scalers = pickle.load(f)
        
        # Timeframe configuration
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
        
        print("Confluence analyzer ready!")
        
    def calculate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all features for a timeframe (same as training)"""
        features = df.copy()
        
        # Trend Features
        features['ema_9'] = features['close'].ewm(span=9, adjust=False).mean()
        features['ema_21'] = features['close'].ewm(span=21, adjust=False).mean()
        features['ema_50'] = features['close'].ewm(span=50, adjust=False).mean()
        
        features['ema_9_21_cross'] = np.where(features['ema_9'] > features['ema_21'], 1, -1)
        features['ema_21_50_cross'] = np.where(features['ema_21'] > features['ema_50'], 1, -1)
        
        features['ema_50_slope'] = features['ema_50'].pct_change(5)
        features['price_slope'] = features['close'].pct_change(10)
        
        # Momentum Features
        delta = features['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        features['rsi'] = 100 - (100 / (1 + rs))
        features['rsi_signal'] = np.where(features['rsi'] > 50, 1, -1)
        
        exp1 = features['close'].ewm(span=12, adjust=False).mean()
        exp2 = features['close'].ewm(span=26, adjust=False).mean()
        features['macd'] = exp1 - exp2
        features['macd_signal'] = features['macd'].ewm(span=9, adjust=False).mean()
        features['macd_histogram'] = features['macd'] - features['macd_signal']
        features['macd_direction'] = np.where(features['macd'] > features['macd_signal'], 1, -1)
        
        rsi_min = features['rsi'].rolling(window=14).min()
        rsi_max = features['rsi'].rolling(window=14).max()
        features['stoch_rsi'] = (features['rsi'] - rsi_min) / (rsi_max - rsi_min)
        features['stoch_rsi_signal'] = np.where(features['stoch_rsi'] > 0.5, 1, -1)
        
        # Volume Features
        features['volume_sma'] = features['volume'].rolling(window=20).mean()
        features['volume_ratio'] = features['volume'] / features['volume_sma']
        features['volume_trend'] = np.where(features['volume_ratio'] > 1.2, 1, 
                                            np.where(features['volume_ratio'] < 0.8, -1, 0))
        
        features['vwap'] = (features['volume'] * (features['high'] + features['low'] + features['close']) / 3).cumsum() / features['volume'].cumsum()
        features['price_vs_vwap'] = np.where(features['close'] > features['vwap'], 1, -1)
        
        # Volatility Features
        high_low = features['high'] - features['low']
        high_close = np.abs(features['high'] - features['close'].shift())
        low_close = np.abs(features['low'] - features['close'].shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        features['atr'] = true_range.rolling(window=14).mean()
        features['atr_pct'] = features['atr'] / features['close']
        
        features['bb_middle'] = features['close'].rolling(window=20).mean()
        bb_std = features['close'].rolling(window=20).std()
        features['bb_upper'] = features['bb_middle'] + (bb_std * 2)
        features['bb_lower'] = features['bb_middle'] - (bb_std * 2)
        features['bb_width'] = (features['bb_upper'] - features['bb_lower']) / features['bb_middle']
        features['bb_position'] = (features['close'] - features['bb_lower']) / (features['bb_upper'] - features['bb_lower'])
        
        features['high_20'] = features['high'].rolling(window=20).max()
        features['low_20'] = features['low'].rolling(window=20).min()
        features['price_range_position'] = (features['close'] - features['low_20']) / (features['high_20'] - features['low_20'])
        
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
        
        features = features.fillna(method='ffill').fillna(0)
        
        return features
    
    def resample_to_timeframe(self, df: pd.DataFrame, target_minutes: int) -> pd.DataFrame:
        """Resample 1-minute data to target timeframe"""
        df = df.copy()
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        
        resampled = df.resample(f'{target_minutes}T').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()
        
        return resampled
    
    def prepare_inference_data(self, df_1m: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Prepare current market data for inference
        
        Args:
            df_1m: Recent 1-minute data (enough to create all timeframes)
        
        Returns:
            Dict of prepared feature arrays per timeframe
        """
        inference_data = {}
        
        for tf_name, tf_minutes in self.timeframes.items():
            # Resample if needed
            if tf_minutes == 1:
                df_tf = df_1m.copy()
            else:
                df_tf = self.resample_to_timeframe(df_1m, tf_minutes)
            
            # Calculate features
            df_tf = self.calculate_features(df_tf)
            
            # Get feature columns (exclude OHLCV)
            feature_cols = [col for col in df_tf.columns if col not in ['open', 'high', 'low', 'close', 'volume']]
            
            # Extract most recent sequence
            lookback = self.lookback_periods[tf_name]
            
            if len(df_tf) < lookback:
                raise ValueError(f"Insufficient data for {tf_name}. Need {lookback}, have {len(df_tf)}")
            
            sequence = df_tf[feature_cols].iloc[-lookback:].values
            
            # Scale using saved scaler
            sequence_flat = sequence.reshape(-1, sequence.shape[-1])
            sequence_scaled = self.scalers[tf_name].transform(sequence_flat).reshape(sequence.shape)
            
            # Add batch dimension
            inference_data[tf_name] = np.expand_dims(sequence_scaled, axis=0)
        
        return inference_data
    
    def analyze(self, df_1m: pd.DataFrame) -> Tuple[float, float, Dict]:
        """
        Analyze current market state and return confluence assessment
        
        Args:
            df_1m: Recent 1-minute OHLCV data
        
        Returns:
            confluence_score: [0-1] agreement across timeframes
            directional_bias: [-1 to +1] predicted direction
            details: Dict with additional analysis
        """
        # Prepare data
        X = self.prepare_inference_data(df_1m)
        
        # Create input list in correct order
        X_list = [X[tf] for tf in sorted(X.keys())]
        
        # Predict
        predictions = self.model.predict(X_list, verbose=0)
        confluence_score = float(predictions[0][0][0])
        directional_bias = float(predictions[1][0][0])
        
        # Generate detailed analysis
        details = self._generate_detailed_analysis(df_1m, confluence_score, directional_bias)
        
        return confluence_score, directional_bias, details
    
    def _generate_detailed_analysis(self, 
                                    df_1m: pd.DataFrame, 
                                    confluence_score: float, 
                                    directional_bias: float) -> Dict:
        """Generate detailed confluence breakdown"""
        
        details = {
            'timestamp': df_1m.index[-1],
            'confluence_score': confluence_score,
            'directional_bias': directional_bias,
            'confluence_level': self._classify_confluence(confluence_score),
            'direction': 'BULLISH' if directional_bias > 0.2 else 'BEARISH' if directional_bias < -0.2 else 'NEUTRAL',
            'strength': abs(directional_bias),
            'timeframe_analysis': {},
            'trading_recommendation': None
        }
        
        # Analyze each timeframe
        for tf_name, tf_minutes in self.timeframes.items():
            if tf_minutes == 1:
                df_tf = df_1m.copy()
            else:
                df_tf = self.resample_to_timeframe(df_1m, tf_minutes)
            
            df_tf = self.calculate_features(df_tf)
            
            latest = df_tf.iloc[-1]
            
            tf_analysis = {
                'trend': 'UP' if latest['ema_9_21_cross'] > 0 else 'DOWN',
                'momentum_rsi': float(latest['rsi']),
                'momentum_macd': 'BULLISH' if latest['macd_direction'] > 0 else 'BEARISH',
                'volume_status': 'HIGH' if latest['volume_ratio'] > 1.2 else 'LOW' if latest['volume_ratio'] < 0.8 else 'NORMAL',
                'volatility_percentile': float(latest['atr_pct']),
                'price_vs_ema50': 'ABOVE' if latest['close'] > latest['ema_50'] else 'BELOW'
            }
            
            details['timeframe_analysis'][tf_name] = tf_analysis
        
        # Generate trading recommendation
        details['trading_recommendation'] = self._generate_recommendation(
            confluence_score, directional_bias, details['timeframe_analysis']
        )
        
        return details
    
    def _classify_confluence(self, score: float) -> str:
        """Classify confluence level"""
        if score >= 0.80:
            return "VERY HIGH"
        elif score >= 0.70:
            return "HIGH"
        elif score >= 0.60:
            return "MEDIUM"
        elif score >= 0.50:
            return "LOW"
        else:
            return "VERY LOW"
    
    def _generate_recommendation(self, 
                                confluence_score: float, 
                                directional_bias: float,
                                tf_analysis: Dict) -> Dict:
        """Generate actionable trading recommendation"""
        
        recommendation = {
            'action': 'HOLD',
            'confidence': 0.0,
            'rationale': [],
            'entry_conditions': [],
            'risk_level': 'HIGH'
        }
        
        # Determine action
        if confluence_score >= 0.70:
            if directional_bias > 0.3:
                recommendation['action'] = 'LONG'
                recommendation['confidence'] = min(confluence_score * abs(directional_bias), 1.0)
            elif directional_bias < -0.3:
                recommendation['action'] = 'SHORT'
                recommendation['confidence'] = min(confluence_score * abs(directional_bias), 1.0)
            else:
                recommendation['action'] = 'WAIT'
                recommendation['rationale'].append("High confluence but weak directional bias")
        else:
            recommendation['action'] = 'WAIT'
            recommendation['rationale'].append("Insufficient confluence across timeframes")
        
        # Risk assessment
        if confluence_score >= 0.80 and abs(directional_bias) >= 0.5:
            recommendation['risk_level'] = 'LOW'
        elif confluence_score >= 0.70 and abs(directional_bias) >= 0.3:
            recommendation['risk_level'] = 'MEDIUM'
        else:
            recommendation['risk_level'] = 'HIGH'
        
        # Entry conditions
        if recommendation['action'] in ['LONG', 'SHORT']:
            recommendation['entry_conditions'].append(f"Confluence score: {confluence_score:.3f}")
            recommendation['entry_conditions'].append(f"Directional bias: {directional_bias:.3f}")
            
            # Check timeframe alignment
            aligned_count = 0
            for tf, analysis in tf_analysis.items():
                if directional_bias > 0 and analysis['trend'] == 'UP':
                    aligned_count += 1
                elif directional_bias < 0 and analysis['trend'] == 'DOWN':
                    aligned_count += 1
            
            recommendation['entry_conditions'].append(f"Timeframes aligned: {aligned_count}/{len(tf_analysis)}")
        
        return recommendation
    
    def get_signal_quality(self, confluence_score: float, directional_bias: float) -> str:
        """
        Assess overall signal quality
        
        Returns: 'EXCELLENT', 'GOOD', 'FAIR', 'POOR'
        """
        if confluence_score >= 0.80 and abs(directional_bias) >= 0.5:
            return 'EXCELLENT'
        elif confluence_score >= 0.70 and abs(directional_bias) >= 0.3:
            return 'GOOD'
        elif confluence_score >= 0.60 and abs(directional_bias) >= 0.2:
            return 'FAIR'
        else:
            return 'POOR'
    
    def should_trade(self, 
                    confluence_score: float, 
                    directional_bias: float,
                    min_confluence: float = 0.70,
                    min_bias: float = 0.3) -> bool:
        """
        Determine if conditions are suitable for trading
        
        Args:
            confluence_score: Current confluence score
            directional_bias: Current directional bias
            min_confluence: Minimum acceptable confluence
            min_bias: Minimum acceptable bias magnitude
        
        Returns:
            True if should trade, False otherwise
        """
        return (confluence_score >= min_confluence and 
                abs(directional_bias) >= min_bias)


def format_analysis_report(score: float, bias: float, details: Dict) -> str:
    """Format analysis results into readable report"""
    
    report = []
    report.append("=" * 80)
    report.append("MULTI-TIMEFRAME CONFLUENCE ANALYSIS")
    report.append("=" * 80)
    report.append(f"\nTimestamp: {details['timestamp']}")
    report.append(f"\nCONFLUENCE SCORE: {score:.3f} ({details['confluence_level']})")
    report.append(f"DIRECTIONAL BIAS: {bias:.3f} ({details['direction']})")
    report.append(f"SIGNAL STRENGTH: {details['strength']:.3f}")
    
    report.append("\n" + "-" * 80)
    report.append("TIMEFRAME BREAKDOWN")
    report.append("-" * 80)
    
    for tf, analysis in details['timeframe_analysis'].items():
        report.append(f"\n{tf.upper()}:")
        report.append(f"  Trend: {analysis['trend']}")
        report.append(f"  RSI: {analysis['momentum_rsi']:.1f}")
        report.append(f"  MACD: {analysis['momentum_macd']}")
        report.append(f"  Volume: {analysis['volume_status']}")
        report.append(f"  Price vs EMA50: {analysis['price_vs_ema50']}")
    
    report.append("\n" + "-" * 80)
    report.append("TRADING RECOMMENDATION")
    report.append("-" * 80)
    
    rec = details['trading_recommendation']
    report.append(f"\nACTION: {rec['action']}")
    report.append(f"CONFIDENCE: {rec['confidence']:.3f}")
    report.append(f"RISK LEVEL: {rec['risk_level']}")
    
    if rec['rationale']:
        report.append("\nRationale:")
        for reason in rec['rationale']:
            report.append(f"  • {reason}")
    
    if rec['entry_conditions']:
        report.append("\nEntry Conditions:")
        for condition in rec['entry_conditions']:
            report.append(f"  • {condition}")
    
    report.append("\n" + "=" * 80)
    
    return "\n".join(report)


def demo_usage():
    """Demonstrate usage of the confluence analyzer"""
    
    print("=" * 80)
    print("Multi-Timeframe Confluence Analyzer - Demo")
    print("=" * 80)
    
    # Initialize analyzer
    analyzer = ConfluenceAnalyzer(
        model_path='/home/claude/mtf_confluence_model_best.keras',
        scaler_path='/home/claude/mtf_confluence_scalers.pkl'
    )
    
    # Create sample data (in production, this comes from your data feed)
    print("\nGenerating sample market data...")
    dates = pd.date_range(start='2024-10-20', end='2024-10-25 16:00', freq='1T')
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
    
    print(f"Sample data: {len(df_1m)} bars from {df_1m.index[0]} to {df_1m.index[-1]}")
    
    # Analyze current market state
    print("\nAnalyzing current market state...")
    score, bias, details = analyzer.analyze(df_1m)
    
    # Print formatted report
    report = format_analysis_report(score, bias, details)
    print("\n" + report)
    
    # Check if should trade
    should_trade = analyzer.should_trade(score, bias, min_confluence=0.70, min_bias=0.3)
    quality = analyzer.get_signal_quality(score, bias)
    
    print(f"\nSHOULD TRADE: {should_trade}")
    print(f"SIGNAL QUALITY: {quality}")
    
    return analyzer


if __name__ == "__main__":
    analyzer = demo_usage()
