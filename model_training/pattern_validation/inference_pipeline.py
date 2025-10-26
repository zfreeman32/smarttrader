"""
Pattern Validation Inference Pipeline
=====================================
Real-time pattern quality scoring for live trading system.

This module loads the trained model and provides fast inference
for validating ICT patterns detected by rule-based algorithms.
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from typing import Dict, List, Optional, Tuple
import json
import os
from dataclasses import dataclass
from datetime import datetime


@dataclass
class PatternScore:
    """Result of pattern validation"""
    pattern_id: str
    pattern_type: str
    direction: str
    quality_score: float
    quality_class: str  # 'high', 'medium', 'low'
    tradeable: bool  # True if score > threshold
    timestamp: datetime
    confidence_level: str  # 'very_high', 'high', 'medium', 'low'
    
    def to_dict(self) -> Dict:
        return {
            'pattern_id': self.pattern_id,
            'pattern_type': self.pattern_type,
            'direction': self.direction,
            'quality_score': float(self.quality_score),
            'quality_class': self.quality_class,
            'tradeable': self.tradeable,
            'timestamp': self.timestamp.isoformat(),
            'confidence_level': self.confidence_level
        }


class PatternValidator:
    """Real-time pattern validation for trading system"""
    
    def __init__(
        self,
        model_path: str,
        config_path: Optional[str] = None,
        quality_threshold: float = 0.75,
        batch_inference: bool = False
    ):
        """
        Initialize pattern validator
        
        Args:
            model_path: Path to trained model (.keras file)
            config_path: Path to model config (optional)
            quality_threshold: Minimum score to consider pattern tradeable
            batch_inference: Use batch inference for efficiency
        """
        self.model_path = model_path
        self.config_path = config_path
        self.quality_threshold = quality_threshold
        self.batch_inference = batch_inference
        
        self.model = None
        self.config = None
        
        self._load_model()
        
        # Performance tracking
        self.inference_times = []
        self.total_patterns_scored = 0
    
    def _load_model(self):
        """Load trained model and configuration"""
        print(f"Loading model from {self.model_path}...")
        
        try:
            self.model = keras.models.load_model(self.model_path)
            print("Model loaded successfully")
            
            # Load config if provided
            if self.config_path and os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    self.config = json.load(f)
                print(f"Config loaded from {self.config_path}")
            
            # Warm up model with dummy data
            self._warmup()
            
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")
    
    def _warmup(self):
        """Warm up model with dummy prediction"""
        dummy_chart = np.random.randn(1, 100, 5)
        dummy_volume = np.random.randn(1, 100, 1)
        dummy_pattern = np.random.randn(1, 13)
        dummy_indicators = np.random.randn(1, 15)
        
        _ = self.model.predict(
            [dummy_chart, dummy_volume, dummy_pattern, dummy_indicators],
            verbose=0
        )
        print("Model warmed up")
    
    def validate_pattern(
        self,
        chart_window: np.ndarray,
        volume_profile: np.ndarray,
        pattern_features: Dict,
        indicator_values: np.ndarray,
        pattern_id: str,
        pattern_type: str,
        direction: str
    ) -> PatternScore:
        """
        Validate a single detected pattern
        
        Args:
            chart_window: OHLCV array shape (100, 5)
            volume_profile: Volume array shape (100,) or (100, 1)
            pattern_features: Dict with pattern metadata
            indicator_values: Technical indicators array shape (15,)
            pattern_id: Unique pattern identifier
            pattern_type: 'FVG', 'OB', or 'Sweep'
            direction: 'bullish' or 'bearish'
        
        Returns:
            PatternScore with quality assessment
        """
        start_time = pd.Timestamp.now()
        
        # Prepare inputs
        X_chart = self._prepare_chart_input(chart_window)
        X_volume = self._prepare_volume_input(volume_profile)
        X_pattern = self._prepare_pattern_features(pattern_features, pattern_type, direction)
        X_indicators = self._prepare_indicator_input(indicator_values)
        
        # Inference
        quality_score = self.model.predict(
            [X_chart, X_volume, X_pattern, X_indicators],
            verbose=0
        )[0][0]
        
        # Track performance
        inference_time = (pd.Timestamp.now() - start_time).total_seconds() * 1000
        self.inference_times.append(inference_time)
        self.total_patterns_scored += 1
        
        # Classify quality
        quality_class = self._classify_quality(quality_score)
        confidence_level = self._get_confidence_level(quality_score)
        tradeable = quality_score >= self.quality_threshold
        
        return PatternScore(
            pattern_id=pattern_id,
            pattern_type=pattern_type,
            direction=direction,
            quality_score=quality_score,
            quality_class=quality_class,
            tradeable=tradeable,
            timestamp=datetime.now(),
            confidence_level=confidence_level
        )
    
    def validate_batch(
        self,
        patterns: List[Dict]
    ) -> List[PatternScore]:
        """
        Validate multiple patterns in batch (more efficient)
        
        Args:
            patterns: List of pattern dicts with all required fields
        
        Returns:
            List of PatternScore objects
        """
        if not patterns:
            return []
        
        start_time = pd.Timestamp.now()
        
        # Prepare batch inputs
        X_chart_batch = []
        X_volume_batch = []
        X_pattern_batch = []
        X_indicators_batch = []
        
        for pattern in patterns:
            X_chart_batch.append(self._prepare_chart_input(pattern['chart_window'])[0])
            X_volume_batch.append(self._prepare_volume_input(pattern['volume_profile'])[0])
            X_pattern_batch.append(
                self._prepare_pattern_features(
                    pattern['pattern_features'],
                    pattern['pattern_type'],
                    pattern['direction']
                )[0]
            )
            X_indicators_batch.append(self._prepare_indicator_input(pattern['indicator_values'])[0])
        
        # Batch inference
        quality_scores = self.model.predict(
            [
                np.array(X_chart_batch),
                np.array(X_volume_batch),
                np.array(X_pattern_batch),
                np.array(X_indicators_batch)
            ],
            verbose=0
        ).flatten()
        
        # Track performance
        total_time = (pd.Timestamp.now() - start_time).total_seconds() * 1000
        avg_time_per_pattern = total_time / len(patterns)
        self.inference_times.extend([avg_time_per_pattern] * len(patterns))
        self.total_patterns_scored += len(patterns)
        
        # Create results
        results = []
        for i, (pattern, score) in enumerate(zip(patterns, quality_scores)):
            results.append(PatternScore(
                pattern_id=pattern.get('pattern_id', f'pattern_{i}'),
                pattern_type=pattern['pattern_type'],
                direction=pattern['direction'],
                quality_score=float(score),
                quality_class=self._classify_quality(score),
                tradeable=score >= self.quality_threshold,
                timestamp=datetime.now(),
                confidence_level=self._get_confidence_level(score)
            ))
        
        return results
    
    def _prepare_chart_input(self, chart_window: np.ndarray) -> np.ndarray:
        """Normalize and reshape chart data"""
        # Ensure shape (100, 5)
        if chart_window.shape != (100, 5):
            raise ValueError(f"Expected chart shape (100, 5), got {chart_window.shape}")
        
        # Normalize
        normalized = chart_window.copy()
        
        # Normalize OHLC together
        price_min = chart_window[:, :4].min()
        price_max = chart_window[:, :4].max()
        if price_max > price_min:
            normalized[:, :4] = (chart_window[:, :4] - price_min) / (price_max - price_min)
        
        # Normalize volume
        vol_max = chart_window[:, 4].max()
        if vol_max > 0:
            normalized[:, 4] = chart_window[:, 4] / vol_max
        
        return normalized.reshape(1, 100, 5)
    
    def _prepare_volume_input(self, volume_profile: np.ndarray) -> np.ndarray:
        """Reshape volume profile"""
        if volume_profile.shape == (100,):
            volume_profile = volume_profile.reshape(100, 1)
        elif volume_profile.shape != (100, 1):
            raise ValueError(f"Expected volume shape (100,) or (100, 1), got {volume_profile.shape}")
        
        return volume_profile.reshape(1, 100, 1)
    
    def _prepare_pattern_features(
        self,
        pattern_features: Dict,
        pattern_type: str,
        direction: str
    ) -> np.ndarray:
        """Convert pattern metadata to feature array"""
        features = np.array([
            1.0 if pattern_type == 'FVG' else 0.0,
            1.0 if pattern_type == 'OB' else 0.0,
            1.0 if pattern_type == 'Sweep' else 0.0,
            1.0 if direction == 'bullish' else -1.0,
            pattern_features.get('pattern_size_normalized', 0.0),
            pattern_features.get('volume_ratio', 1.0),
            1.0 if pattern_features.get('volume_confirmed', False) else 0.0,
            1.0 if pattern_features.get('session', '') == 'London' else 0.0,
            1.0 if pattern_features.get('session', '') == 'NY' else 0.0,
            pattern_features.get('rsi', 50.0) / 100.0,
            pattern_features.get('adx', 0.0) / 100.0,
            1.0 if pattern_features.get('trend_alignment', False) else 0.0,
            pattern_features.get('distance_to_structure', 0.0)
        ])
        
        return features.reshape(1, -1)
    
    def _prepare_indicator_input(self, indicator_values: np.ndarray) -> np.ndarray:
        """Reshape indicator values"""
        if indicator_values.shape != (15,):
            raise ValueError(f"Expected 15 indicators, got {indicator_values.shape}")
        
        return indicator_values.reshape(1, -1)
    
    def _classify_quality(self, score: float) -> str:
        """Classify quality into categories"""
        if score >= 0.85:
            return 'high'
        elif score >= 0.50:
            return 'medium'
        else:
            return 'low'
    
    def _get_confidence_level(self, score: float) -> str:
        """Get confidence level for the score"""
        if score >= 0.90:
            return 'very_high'
        elif score >= 0.75:
            return 'high'
        elif score >= 0.50:
            return 'medium'
        else:
            return 'low'
    
    def get_performance_stats(self) -> Dict:
        """Get inference performance statistics"""
        if not self.inference_times:
            return {}
        
        times = np.array(self.inference_times)
        
        return {
            'total_patterns_scored': self.total_patterns_scored,
            'avg_inference_time_ms': float(times.mean()),
            'median_inference_time_ms': float(np.median(times)),
            'min_inference_time_ms': float(times.min()),
            'max_inference_time_ms': float(times.max()),
            'std_inference_time_ms': float(times.std()),
            'p95_inference_time_ms': float(np.percentile(times, 95)),
            'p99_inference_time_ms': float(np.percentile(times, 99))
        }
    
    def reset_stats(self):
        """Reset performance tracking"""
        self.inference_times = []
        self.total_patterns_scored = 0


class TradingSystemIntegration:
    """
    Integration layer for pattern validator in trading system.
    Filters detected patterns before they reach the ensemble decision layer.
    """
    
    def __init__(
        self,
        validator: PatternValidator,
        min_quality_score: float = 0.75,
        log_filtered_patterns: bool = True
    ):
        """
        Initialize trading system integration
        
        Args:
            validator: PatternValidator instance
            min_quality_score: Minimum score to pass pattern to trading system
            log_filtered_patterns: Log filtered patterns for analysis
        """
        self.validator = validator
        self.min_quality_score = min_quality_score
        self.log_filtered_patterns = log_filtered_patterns
        
        self.filtered_count = 0
        self.passed_count = 0
        self.filtered_log = []
    
    def filter_patterns(
        self,
        detected_patterns: List[Dict]
    ) -> Tuple[List[Dict], List[PatternScore]]:
        """
        Filter detected patterns by quality score
        
        Args:
            detected_patterns: List of patterns from ICT detectors
        
        Returns:
            validated_patterns: Patterns that passed validation
            all_scores: All pattern scores for logging
        """
        if not detected_patterns:
            return [], []
        
        # Batch validation
        scores = self.validator.validate_batch(detected_patterns)
        
        # Filter by quality
        validated_patterns = []
        
        for pattern, score in zip(detected_patterns, scores):
            if score.tradeable:
                # Add quality score to pattern
                pattern['quality_score'] = score.quality_score
                pattern['quality_class'] = score.quality_class
                validated_patterns.append(pattern)
                self.passed_count += 1
            else:
                self.filtered_count += 1
                
                if self.log_filtered_patterns:
                    self.filtered_log.append({
                        'timestamp': datetime.now().isoformat(),
                        'pattern_type': pattern['pattern_type'],
                        'direction': pattern['direction'],
                        'quality_score': score.quality_score,
                        'reason': 'quality_score_below_threshold'
                    })
        
        return validated_patterns, scores
    
    def get_filter_stats(self) -> Dict:
        """Get filtering statistics"""
        total = self.filtered_count + self.passed_count
        
        return {
            'total_patterns': total,
            'passed_patterns': self.passed_count,
            'filtered_patterns': self.filtered_count,
            'pass_rate': self.passed_count / total if total > 0 else 0.0,
            'filter_rate': self.filtered_count / total if total > 0 else 0.0
        }
    
    def save_filtered_log(self, filepath: str):
        """Save filtered patterns log"""
        df = pd.DataFrame(self.filtered_log)
        df.to_csv(filepath, index=False)
        print(f"Filtered patterns log saved to {filepath}")


# Example usage functions
def example_single_pattern_validation():
    """Example: Validate a single pattern"""
    
    # Initialize validator
    validator = PatternValidator(
        model_path='./pattern_validation_outputs/best_model.keras',
        config_path='./pattern_validation_outputs/model_config.json',
        quality_threshold=0.75
    )
    
    # Example pattern data (from your ICT detector)
    chart_window = np.random.randn(100, 5)  # Replace with real OHLCV
    volume_profile = np.random.randn(100)
    
    pattern_features = {
        'pattern_size_normalized': 0.8,
        'volume_ratio': 2.5,
        'volume_confirmed': True,
        'session': 'London',
        'rsi': 65.0,
        'adx': 35.0,
        'trend_alignment': True,
        'distance_to_structure': 0.3
    }
    
    indicator_values = np.random.randn(15)
    
    # Validate pattern
    score = validator.validate_pattern(
        chart_window=chart_window,
        volume_profile=volume_profile,
        pattern_features=pattern_features,
        indicator_values=indicator_values,
        pattern_id='FVG_20250101_120000',
        pattern_type='FVG',
        direction='bullish'
    )
    
    print("\nPattern Validation Result:")
    print(f"  Quality Score: {score.quality_score:.3f}")
    print(f"  Quality Class: {score.quality_class}")
    print(f"  Tradeable: {score.tradeable}")
    print(f"  Confidence: {score.confidence_level}")
    
    # Check performance
    stats = validator.get_performance_stats()
    print(f"\nInference Time: {stats['avg_inference_time_ms']:.2f} ms")
    
    return score


def example_batch_validation():
    """Example: Validate multiple patterns at once"""
    
    # Initialize validator
    validator = PatternValidator(
        model_path='./pattern_validation_outputs/best_model.keras',
        quality_threshold=0.75,
        batch_inference=True
    )
    
    # Example patterns (replace with real data)
    patterns = [
        {
            'pattern_id': f'pattern_{i}',
            'pattern_type': np.random.choice(['FVG', 'OB', 'Sweep']),
            'direction': np.random.choice(['bullish', 'bearish']),
            'chart_window': np.random.randn(100, 5),
            'volume_profile': np.random.randn(100),
            'pattern_features': {
                'pattern_size_normalized': np.random.uniform(0.5, 2.0),
                'volume_ratio': np.random.uniform(1.0, 3.0),
                'volume_confirmed': np.random.choice([True, False]),
                'session': np.random.choice(['London', 'NY', 'Asian']),
                'rsi': np.random.uniform(30, 70),
                'adx': np.random.uniform(10, 50),
                'trend_alignment': np.random.choice([True, False]),
                'distance_to_structure': np.random.uniform(0, 1)
            },
            'indicator_values': np.random.randn(15)
        }
        for i in range(10)
    ]
    
    # Batch validation
    scores = validator.validate_batch(patterns)
    
    print(f"\nValidated {len(scores)} patterns")
    print(f"Tradeable: {sum(s.tradeable for s in scores)}")
    print(f"Average quality score: {np.mean([s.quality_score for s in scores]):.3f}")
    
    # Performance stats
    stats = validator.get_performance_stats()
    print(f"\nAverage inference time: {stats['avg_inference_time_ms']:.2f} ms/pattern")
    
    return scores


def example_trading_system_integration():
    """Example: Integration with trading system"""
    
    # Initialize validator
    validator = PatternValidator(
        model_path='./pattern_validation_outputs/best_model.keras',
        quality_threshold=0.75
    )
    
    # Initialize integration
    integration = TradingSystemIntegration(
        validator=validator,
        min_quality_score=0.75,
        log_filtered_patterns=True
    )
    
    # Simulate ICT detector output (replace with real detections)
    detected_patterns = [
        {
            'pattern_id': f'pattern_{i}',
            'pattern_type': np.random.choice(['FVG', 'OB', 'Sweep']),
            'direction': np.random.choice(['bullish', 'bearish']),
            'chart_window': np.random.randn(100, 5),
            'volume_profile': np.random.randn(100),
            'pattern_features': {
                'pattern_size_normalized': np.random.uniform(0.5, 2.0),
                'volume_ratio': np.random.uniform(1.0, 3.0),
                'volume_confirmed': True,
                'session': 'London',
                'rsi': 65.0,
                'adx': 30.0,
                'trend_alignment': True,
                'distance_to_structure': 0.3
            },
            'indicator_values': np.random.randn(15)
        }
        for i in range(20)
    ]
    
    # Filter patterns
    validated_patterns, all_scores = integration.filter_patterns(detected_patterns)
    
    print(f"\nDetected patterns: {len(detected_patterns)}")
    print(f"Validated patterns: {len(validated_patterns)}")
    print(f"Filtered out: {len(detected_patterns) - len(validated_patterns)}")
    
    # Stats
    stats = integration.get_filter_stats()
    print(f"\nPass rate: {stats['pass_rate']*100:.1f}%")
    print(f"Filter rate: {stats['filter_rate']*100:.1f}%")
    
    # Save log
    integration.save_filtered_log('./filtered_patterns.csv')
    
    return validated_patterns, all_scores


if __name__ == "__main__":
    print("="*60)
    print("PATTERN VALIDATION INFERENCE - EXAMPLES")
    print("="*60)
    
    print("\n1. Single Pattern Validation:")
    score = example_single_pattern_validation()
    
    print("\n" + "="*60)
    print("\n2. Batch Validation:")
    scores = example_batch_validation()
    
    print("\n" + "="*60)
    print("\n3. Trading System Integration:")
    validated, all_scores = example_trading_system_integration()