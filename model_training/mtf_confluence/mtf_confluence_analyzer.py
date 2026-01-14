"""
Multi-Timeframe Confluence Model - Analyzer
===========================================

Provides analysis and inference utilities for the trained confluence model.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
import pickle
import tensorflow as tf
from tensorflow import keras


class ConfluenceAnalyzer:
    """Analyzes and uses trained confluence model for trading decisions"""
    
    def __init__(self, model_path: str, scaler_path: str):
        """
        Args:
            model_path: Path to trained .keras model file
            scaler_path: Path to scalers .pkl file
        """
        self.model = keras.models.load_model(model_path)
        
        with open(scaler_path, 'rb') as f:
            self.scalers = pickle.load(f)
        
        print(f"✓ Model loaded from: {model_path}")
        print(f"✓ Scalers loaded from: {scaler_path}")
    
    def preprocess_data(self, data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Preprocess new data using trained scalers
        
        Args:
            data: Dict of {timeframe: array} with shape (samples, timesteps, features)
            
        Returns:
            Scaled data
        """
        scaled_data = {}
        
        for tf_name, arr in data.items():
            if tf_name not in self.scalers:
                raise ValueError(f"No scaler found for timeframe: {tf_name}")
            
            scaler = self.scalers[tf_name]
            original_shape = arr.shape
            
            # Reshape to 2D
            arr_2d = arr.reshape(-1, original_shape[-1])
            
            # Scale
            scaled = scaler.transform(arr_2d)
            
            # Reshape back
            scaled_data[tf_name] = scaled.reshape(original_shape)
        
        return scaled_data
    
    def analyze(self, data: Dict[str, np.ndarray]) -> Tuple[float, float, Dict]:
        """
        Analyze current market state
        
        Args:
            data: Dict of {timeframe: array} - should be single sample per timeframe
            
        Returns:
            Tuple of (confluence_score, directional_bias, details_dict)
        """
        # Ensure data is in correct format
        for tf_name in data:
            if len(data[tf_name].shape) == 2:
                # Add batch dimension
                data[tf_name] = np.expand_dims(data[tf_name], axis=0)
        
        # Preprocess
        scaled_data = self.preprocess_data(data)
        
        # Prepare input list (sorted by timeframe name)
        X_list = [scaled_data[tf] for tf in sorted(scaled_data.keys())]
        
        # Predict
        predictions = self.model.predict(X_list, verbose=0)
        
        confluence_score = float(predictions[0][0][0])
        directional_bias = float(predictions[1][0][0])
        
        # Determine signal quality
        if confluence_score >= 0.85:
            quality = "EXCELLENT"
        elif confluence_score >= 0.70:
            quality = "GOOD"
        elif confluence_score >= 0.60:
            quality = "FAIR"
        else:
            quality = "POOR"
        
        # Determine direction
        if abs(directional_bias) >= 0.50:
            strength = "STRONG"
        elif abs(directional_bias) >= 0.30:
            strength = "MODERATE"
        elif abs(directional_bias) >= 0.20:
            strength = "WEAK"
        else:
            strength = "NEUTRAL"
        
        direction = "BULLISH" if directional_bias > 0 else "BEARISH" if directional_bias < 0 else "NEUTRAL"
        
        details = {
            'confluence_score': confluence_score,
            'directional_bias': directional_bias,
            'quality': quality,
            'direction': direction,
            'strength': strength,
            'tradeable': confluence_score >= 0.65 and abs(directional_bias) >= 0.20
        }
        
        return confluence_score, directional_bias, details
    
    def should_trade(self, 
                    confluence_score: float, 
                    directional_bias: float,
                    min_confluence: float = 0.70,
                    min_bias: float = 0.30) -> bool:
        """
        Determine if signal is strong enough to trade
        
        Args:
            confluence_score: Predicted confluence [0-1]
            directional_bias: Predicted bias [-1, +1]
            min_confluence: Minimum confluence threshold
            min_bias: Minimum bias threshold
            
        Returns:
            True if should trade, False otherwise
        """
        return (confluence_score >= min_confluence and 
                abs(directional_bias) >= min_bias)
    
    def get_signal(self, 
                  data: Dict[str, np.ndarray],
                  min_confluence: float = 0.70,
                  min_bias: float = 0.30) -> str:
        """
        Get trading signal from current data
        
        Args:
            data: Multi-timeframe data
            min_confluence: Minimum confluence threshold
            min_bias: Minimum bias threshold
            
        Returns:
            'LONG', 'SHORT', or 'HOLD'
        """
        confluence, bias, details = self.analyze(data)
        
        if not self.should_trade(confluence, bias, min_confluence, min_bias):
            return 'HOLD'
        
        if bias > 0:
            return 'LONG'
        else:
            return 'SHORT'


def calculate_detailed_metrics(y_true_conf: np.ndarray,
                               y_pred_conf: np.ndarray,
                               y_true_bias: np.ndarray,
                               y_pred_bias: np.ndarray) -> Dict:
    """
    Calculate comprehensive metrics for model evaluation
    
    Returns:
        Dictionary of metrics
    """
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    
    metrics = {}
    
    # Confluence metrics
    metrics['conf_mae'] = mean_absolute_error(y_true_conf, y_pred_conf)
    metrics['conf_mse'] = mean_squared_error(y_true_conf, y_pred_conf)
    metrics['conf_rmse'] = np.sqrt(metrics['conf_mse'])
    metrics['conf_r2'] = r2_score(y_true_conf, y_pred_conf)
    
    # Bias metrics
    metrics['bias_mae'] = mean_absolute_error(y_true_bias, y_pred_bias)
    metrics['bias_mse'] = mean_squared_error(y_true_bias, y_pred_bias)
    metrics['bias_rmse'] = np.sqrt(metrics['bias_mse'])
    metrics['bias_r2'] = r2_score(y_true_bias, y_pred_bias)
    
    # Directional accuracy
    true_direction = np.sign(y_true_bias)
    pred_direction = np.sign(y_pred_bias)
    metrics['directional_accuracy'] = np.mean(true_direction == pred_direction) * 100
    
    # Correlation
    metrics['conf_correlation'] = np.corrcoef(y_true_conf, y_pred_conf)[0, 1]
    metrics['bias_correlation'] = np.corrcoef(y_true_bias, y_pred_bias)[0, 1]
    
    # Trading signal analysis
    high_conf_mask = y_pred_conf >= 0.70
    metrics['pct_high_confluence'] = np.mean(high_conf_mask) * 100
    
    strong_bias_mask = np.abs(y_pred_bias) >= 0.30
    tradeable_mask = high_conf_mask & strong_bias_mask
    metrics['pct_tradeable_signals'] = np.mean(tradeable_mask) * 100
    
    # Quality distribution
    excellent_mask = (y_pred_conf >= 0.85) & (np.abs(y_pred_bias) >= 0.50)
    good_mask = (y_pred_conf >= 0.70) & (np.abs(y_pred_bias) >= 0.30)
    fair_mask = (y_pred_conf >= 0.60) & (np.abs(y_pred_bias) >= 0.20)
    
    metrics['excellent_signals'] = np.sum(excellent_mask)
    metrics['good_signals'] = np.sum(good_mask)
    metrics['fair_signals'] = np.sum(fair_mask)
    
    # Directional accuracy by confluence level
    for threshold in [60, 70, 75, 80, 85]:
        mask = y_pred_conf >= (threshold / 100)
        if np.sum(mask) > 0:
            acc = np.mean(true_direction[mask] == pred_direction[mask]) * 100
            metrics[f'dir_acc_conf_{threshold}'] = acc
    
    return metrics


def format_analysis_report(metrics: Dict) -> str:
    """Format metrics into readable report"""
    
    report = []
    report.append("="*80)
    report.append("MODEL PERFORMANCE ANALYSIS")
    report.append("="*80)
    report.append("")
    
    report.append("CONFLUENCE SCORE METRICS:")
    report.append(f"  MAE: {metrics['conf_mae']:.4f}")
    report.append(f"  RMSE: {metrics['conf_rmse']:.4f}")
    report.append(f"  R²: {metrics['conf_r2']:.4f}")
    report.append(f"  Correlation: {metrics['conf_correlation']:.4f}")
    report.append("")
    
    report.append("DIRECTIONAL BIAS METRICS:")
    report.append(f"  MAE: {metrics['bias_mae']:.4f}")
    report.append(f"  RMSE: {metrics['bias_rmse']:.4f}")
    report.append(f"  R²: {metrics['bias_r2']:.4f}")
    report.append(f"  Correlation: {metrics['bias_correlation']:.4f}")
    report.append(f"  Directional Accuracy: {metrics['directional_accuracy']:.2f}%")
    report.append("")
    
    report.append("TRADING SIGNAL QUALITY:")
    report.append(f"  High Confluence Signals (≥0.70): {metrics['pct_high_confluence']:.2f}%")
    report.append(f"  Tradeable Signals: {metrics['pct_tradeable_signals']:.2f}%")
    report.append("")
    
    report.append("SIGNAL DISTRIBUTION:")
    report.append(f"  EXCELLENT (≥0.85 conf, |bias|≥0.50): {metrics['excellent_signals']:,}")
    report.append(f"  GOOD (≥0.70 conf, |bias|≥0.30): {metrics['good_signals']:,}")
    report.append(f"  FAIR (≥0.60 conf, |bias|≥0.20): {metrics['fair_signals']:,}")
    report.append("")
    
    report.append("DIRECTIONAL ACCURACY BY CONFLUENCE:")
    for threshold in [60, 70, 75, 80, 85]:
        key = f'dir_acc_conf_{threshold}'
        if key in metrics:
            report.append(f"  ≥{threshold/100:.2f}: {metrics[key]:.2f}%")
    
    report.append("="*80)
    
    return "\n".join(report)


__all__ = [
    'ConfluenceAnalyzer',
    'calculate_detailed_metrics',
    'format_analysis_report'
]