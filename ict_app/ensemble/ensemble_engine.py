"""
Ensemble Decision Engine - TIER 4
Aggregates all ICT detectors and ML models into final trading signals.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Signal(Enum):
    """Trading signal types"""
    LONG = 1
    SHORT = -1
    HOLD = 0


class MarketRegime(Enum):
    """Market regime classifications"""
    STRONG_UPTREND = 0
    WEAK_UPTREND = 1
    RANGING = 2
    WEAK_DOWNTREND = 3
    STRONG_DOWNTREND = 4
    HIGH_VOLATILITY = 5
    LOW_VOLATILITY = 6


@dataclass
class ICTPattern:
    """Container for detected ICT pattern"""
    pattern_type: str  # 'FVG', 'OB', 'SWEEP'
    direction: str  # 'bullish', 'bearish'
    entry_price: float
    stop_loss: float
    take_profit: float
    confidence: float  # From pattern validator
    zone_top: Optional[float] = None
    zone_bottom: Optional[float] = None
    session: Optional[str] = None


@dataclass
class TradingSignal:
    """Final trading signal output"""
    signal: Signal
    confidence: float
    entry_price: float
    stop_loss: float
    take_profit: float
    position_size: int
    ict_patterns: List[ICTPattern]
    regime: str
    confluence_score: float
    timestamp: pd.Timestamp
    metadata: Dict


class EnsembleDecisionEngine:
    """
    Main ensemble system that combines:
    - ICT pattern detectors (TIER 1)
    - ML models (TIER 2)
    - Risk management (TIER 3)
    
    Produces final trading signals with confidence scores.
    """
    
    def __init__(
        self,
        long_classifier,
        short_classifier,
        returns_5_model,
        regime_classifier,
        pattern_validator,
        confluence_model,
        fvg_detector,
        ob_detector,
        sweep_detector,
        structure_analyzer,
        session_filter,
        min_confidence: float = 0.70,
        min_pattern_quality: float = 0.75,
        min_confluence: float = 0.65
    ):
        """
        Initialize ensemble with all components.
        
        Args:
            long_classifier: Trained long signal model
            short_classifier: Trained short signal model
            returns_5_model: 5-period returns forecaster
            regime_classifier: Market regime classifier
            pattern_validator: Pattern quality scorer
            confluence_model: Multi-timeframe confluence model
            fvg_detector: Fair Value Gap detector
            ob_detector: Order Block detector
            sweep_detector: Liquidity Sweep detector
            structure_analyzer: BOS/CHoCH analyzer
            session_filter: Session context filter
            min_confidence: Minimum confidence threshold for signals
            min_pattern_quality: Minimum pattern validation score
            min_confluence: Minimum confluence score
        """
        # ML Models
        self.long_classifier = long_classifier
        self.short_classifier = short_classifier
        self.returns_5_model = returns_5_model
        self.regime_classifier = regime_classifier
        self.pattern_validator = pattern_validator
        self.confluence_model = confluence_model
        
        # ICT Detectors
        self.fvg_detector = fvg_detector
        self.ob_detector = ob_detector
        self.sweep_detector = sweep_detector
        self.structure_analyzer = structure_analyzer
        self.session_filter = session_filter
        
        # Thresholds
        self.min_confidence = min_confidence
        self.min_pattern_quality = min_pattern_quality
        self.min_confluence = min_confluence
        
        # Model weights for ensemble (calibrate these on validation data)
        self.weights = {
            'long_classifier': 0.25,
            'short_classifier': 0.25,
            'returns_5': 0.10,
            'pattern_validation': 0.15,
            'confluence': 0.10,
            'regime': 0.08,
            'order_flow': 0.07  # Optional, can be 0 if not available
        }
        
        logger.info("Ensemble Decision Engine initialized")
    
    def generate_signal(
        self,
        current_data: pd.DataFrame,
        mtf_data: Dict[str, pd.DataFrame],
        current_price: float,
        account_balance: float,
        current_positions: int = 0
    ) -> Optional[TradingSignal]:
        """
        Main signal generation pipeline.
        
        Args:
            current_data: Current timeframe OHLCV + features
            mtf_data: Multi-timeframe data dict {'1m': df, '5m': df, ...}
            current_price: Current market price
            account_balance: Current account balance
            current_positions: Number of open positions
            
        Returns:
            TradingSignal or None if no signal
        """
        try:
            logger.info("=" * 80)
            logger.info("GENERATING TRADING SIGNAL")
            logger.info("=" * 80)
            
            # Step 1: Detect ICT patterns (TIER 1)
            logger.info("Step 1: Detecting ICT patterns...")
            patterns = self._detect_ict_patterns(current_data)
            logger.info(f"Detected {len(patterns)} ICT patterns")
            
            if not patterns:
                logger.info("No ICT patterns detected. Signal: HOLD")
                return None
            
            # Step 2: Validate patterns with ML
            logger.info("Step 2: Validating pattern quality...")
            validated_patterns = self._validate_patterns(patterns, current_data)
            logger.info(f"Validated {len(validated_patterns)}/{len(patterns)} patterns")
            
            if not validated_patterns:
                logger.info("No high-quality patterns. Signal: HOLD")
                return None
            
            # Step 3: Get ML model predictions
            logger.info("Step 3: Running ML predictions...")
            predictions = self._get_ml_predictions(current_data, mtf_data)
            logger.info(f"ML Predictions: {predictions}")
            
            # Step 4: Check entry requirements
            logger.info("Step 4: Checking entry requirements...")
            entry_check = self._check_entry_requirements(
                predictions,
                validated_patterns
            )
            
            if not entry_check['passed']:
                logger.info(f"Entry requirements not met: {entry_check['reason']}")
                return None
            
            # Step 5: Calculate ensemble confidence
            logger.info("Step 5: Calculating ensemble confidence...")
            signal_type, confidence = self._calculate_ensemble_confidence(
                predictions,
                validated_patterns
            )
            logger.info(f"Signal: {signal_type.name}, Confidence: {confidence:.3f}")
            
            if confidence < self.min_confidence:
                logger.info(f"Confidence {confidence:.3f} below threshold {self.min_confidence}")
                return None
            
            # Step 6: Determine position sizing
            logger.info("Step 6: Calculating position size...")
            position_size = self._calculate_position_size(
                confidence,
                account_balance,
                current_positions,
                predictions['regime']
            )
            logger.info(f"Position size: {position_size} contracts")
            
            # Step 7: Calculate stops and targets
            logger.info("Step 7: Calculating stops and targets...")
            stops_targets = self._calculate_stops_targets(
                signal_type,
                validated_patterns,
                current_price,
                current_data
            )
            logger.info(f"Entry: {current_price:.5f}, Stop: {stops_targets['stop']:.5f}, Target: {stops_targets['target']:.5f}")
            
            # Step 8: Build final signal
            trading_signal = TradingSignal(
                signal=signal_type,
                confidence=confidence,
                entry_price=current_price,
                stop_loss=stops_targets['stop'],
                take_profit=stops_targets['target'],
                position_size=position_size,
                ict_patterns=validated_patterns,
                regime=predictions['regime'],
                confluence_score=predictions['confluence_score'],
                timestamp=pd.Timestamp.now(),
                metadata={
                    'ml_predictions': predictions,
                    'entry_requirements': entry_check,
                    'risk_reward': stops_targets['risk_reward']
                }
            )
            
            logger.info("=" * 80)
            logger.info(f"FINAL SIGNAL: {signal_type.name} @ {current_price:.5f}")
            logger.info(f"Confidence: {confidence:.3f} | Size: {position_size} | R:R: {stops_targets['risk_reward']:.2f}")
            logger.info("=" * 80)
            
            return trading_signal
            
        except Exception as e:
            logger.error(f"Error generating signal: {str(e)}", exc_info=True)
            return None
    
    def _detect_ict_patterns(self, data: pd.DataFrame) -> List[ICTPattern]:
        """Detect all ICT patterns using rule-based detectors"""
        patterns = []
        
        # Detect FVGs
        fvgs = self.fvg_detector.detect(data)
        for fvg in fvgs:
            patterns.append(ICTPattern(
                pattern_type='FVG',
                direction=fvg['direction'],
                entry_price=fvg['entry'],
                stop_loss=fvg['stop'],
                take_profit=fvg['target'],
                confidence=0.0,  # Will be scored by validator
                zone_top=fvg.get('top'),
                zone_bottom=fvg.get('bottom')
            ))
        
        # Detect Order Blocks
        obs = self.ob_detector.detect(data)
        for ob in obs:
            patterns.append(ICTPattern(
                pattern_type='OB',
                direction=ob['direction'],
                entry_price=ob['entry'],
                stop_loss=ob['stop'],
                take_profit=ob['target'],
                confidence=0.0,
                zone_top=ob.get('top'),
                zone_bottom=ob.get('bottom')
            ))
        
        # Detect Liquidity Sweeps
        sweeps = self.sweep_detector.detect(data)
        for sweep in sweeps:
            patterns.append(ICTPattern(
                pattern_type='SWEEP',
                direction=sweep['direction'],
                entry_price=sweep['entry'],
                stop_loss=sweep['stop'],
                take_profit=sweep['target'],
                confidence=0.0
            ))
        
        return patterns
    
    def _validate_patterns(
        self,
        patterns: List[ICTPattern],
        data: pd.DataFrame
    ) -> List[ICTPattern]:
        """Validate pattern quality using ML model"""
        validated = []
        
        for pattern in patterns:
            # Get pattern features for validator
            pattern_features = self._extract_pattern_features(pattern, data)
            
            # Score with ML model
            quality_score = self.pattern_validator.predict(pattern_features)[0]
            pattern.confidence = quality_score
            
            # Keep only high-quality patterns
            if quality_score >= self.min_pattern_quality:
                validated.append(pattern)
                logger.debug(f"{pattern.pattern_type} {pattern.direction}: quality={quality_score:.3f} ✓")
            else:
                logger.debug(f"{pattern.pattern_type} {pattern.direction}: quality={quality_score:.3f} ✗")
        
        return validated
    
    def _get_ml_predictions(
        self,
        data: pd.DataFrame,
        mtf_data: Dict[str, pd.DataFrame]
    ) -> Dict:
        """Get predictions from all ML models"""
        
        # Prepare input features (use last N timesteps)
        features = self._prepare_features(data)
        
        # Long/Short probabilities
        long_prob = self.long_classifier.predict(features, verbose=0)[0][0]
        short_prob = self.short_classifier.predict(features, verbose=0)[0][0]
        
        # Returns forecast
        returns_5 = self.returns_5_model.predict(features, verbose=0)[0][0]
        
        # Market regime
        regime_probs = self.regime_classifier.predict(features, verbose=0)[0]
        regime_class = np.argmax(regime_probs)
        regime = MarketRegime(regime_class).name
        
        # Multi-timeframe confluence
        confluence_features = self._prepare_mtf_features(mtf_data)
        confluence_score = self.confluence_model.predict(confluence_features, verbose=0)[0][0]
        
        return {
            'long_prob': float(long_prob),
            'short_prob': float(short_prob),
            'returns_5': float(returns_5),
            'regime': regime,
            'regime_probs': regime_probs.tolist(),
            'confluence_score': float(confluence_score)
        }
    
    def _check_entry_requirements(
        self,
        predictions: Dict,
        patterns: List[ICTPattern]
    ) -> Dict:
        """Check if entry requirements are met"""
        
        # REQUIRED conditions
        checks = {
            'long_signal': predictions['long_prob'] > 0.70,
            'short_signal': predictions['short_prob'] > 0.70,
            'returns_positive': predictions['returns_5'] > 0.002,
            'returns_negative': predictions['returns_5'] < -0.002,
            'has_validated_pattern': len(patterns) > 0,
            'confluence_ok': predictions['confluence_score'] >= self.min_confluence
        }
        
        # For LONG entry
        long_required = (
            checks['long_signal'] and
            checks['returns_positive'] and
            checks['has_validated_pattern']
        )
        
        # For SHORT entry
        short_required = (
            checks['short_signal'] and
            checks['returns_negative'] and
            checks['has_validated_pattern']
        )
        
        # Check regime compatibility
        regime = predictions['regime']
        long_regime_ok = regime in ['STRONG_UPTREND', 'WEAK_UPTREND', 'RANGING']
        short_regime_ok = regime in ['STRONG_DOWNTREND', 'WEAK_DOWNTREND', 'RANGING']
        
        if long_required and long_regime_ok:
            return {
                'passed': True,
                'direction': 'LONG',
                'checks': checks,
                'reason': 'All LONG requirements met'
            }
        elif short_required and short_regime_ok:
            return {
                'passed': True,
                'direction': 'SHORT',
                'checks': checks,
                'reason': 'All SHORT requirements met'
            }
        else:
            reason = "Entry requirements not met: "
            if not long_required and not short_required:
                reason += "No strong ML signal"
            elif not long_regime_ok and not short_regime_ok:
                reason += f"Regime {regime} incompatible"
            return {
                'passed': False,
                'direction': None,
                'checks': checks,
                'reason': reason
            }
    
    def _calculate_ensemble_confidence(
        self,
        predictions: Dict,
        patterns: List[ICTPattern]
    ) -> Tuple[Signal, float]:
        """Calculate weighted ensemble confidence score"""
        
        # Determine direction based on strongest signal
        long_score = predictions['long_prob']
        short_score = predictions['short_prob']
        
        if long_score > short_score:
            signal_type = Signal.LONG
            primary_score = long_score
        else:
            signal_type = Signal.SHORT
            primary_score = short_score
        
        # Weighted ensemble calculation
        weighted_sum = (
            primary_score * self.weights['long_classifier'] +
            abs(predictions['returns_5']) * 100 * self.weights['returns_5'] +
            np.mean([p.confidence for p in patterns]) * self.weights['pattern_validation'] +
            predictions['confluence_score'] * self.weights['confluence'] +
            0.5 * self.weights['regime']  # Regime is binary (compatible or not)
        )
        
        # Normalize to [0, 1]
        confidence = np.clip(weighted_sum, 0, 1)
        
        return signal_type, float(confidence)
    
    def _calculate_position_size(
        self,
        confidence: float,
        account_balance: float,
        current_positions: int,
        regime: str
    ) -> int:
        """Calculate dynamic position size based on confidence and risk"""
        
        # Base risk: 1% of account per trade
        base_risk = account_balance * 0.01
        
        # Confidence multiplier (0.70-0.95 confidence -> 0.7-0.95 multiplier)
        confidence_multiplier = confidence
        
        # Reduce size in high volatility regimes
        if regime in ['HIGH_VOLATILITY']:
            confidence_multiplier *= 0.5
        
        # Reduce size if already have positions (max 2 contracts)
        if current_positions >= 2:
            return 0  # Don't open more positions
        
        # Calculate position size
        # For EURUSD: 1 micro lot = $1000, pip value ≈ $0.10
        # Assuming stop distance of ~20 pips, risk per contract = $2
        risk_per_contract = 2.0  # Adjust based on your broker
        contracts = int((base_risk * confidence_multiplier) / risk_per_contract)
        
        # Enforce limits: 1-3 contracts max
        return max(1, min(contracts, 3))
    
    def _calculate_stops_targets(
        self,
        signal_type: Signal,
        patterns: List[ICTPattern],
        current_price: float,
        data: pd.DataFrame
    ) -> Dict:
        """Calculate stop loss and take profit based on pattern type"""
        
        # Use the highest confidence pattern
        best_pattern = max(patterns, key=lambda p: p.confidence)
        
        # Calculate ATR for dynamic stops
        atr = data['ATR'].iloc[-1] if 'ATR' in data.columns else 0.0010  # Default 10 pips
        
        if best_pattern.pattern_type == 'FVG':
            # FVG: Stop beyond zone + 1 ATR, target opposite zone or 2R
            if signal_type == Signal.LONG:
                stop = best_pattern.zone_bottom - atr
                risk = current_price - stop
                target = current_price + (risk * 2)  # 1:2 R:R minimum
            else:
                stop = best_pattern.zone_top + atr
                risk = stop - current_price
                target = current_price - (risk * 2)
                
        elif best_pattern.pattern_type == 'OB':
            # Order Block: Stop beyond zone + 0.5 ATR, target previous swing
            if signal_type == Signal.LONG:
                stop = best_pattern.zone_bottom - (atr * 0.5)
                risk = current_price - stop
                target = current_price + (risk * 3)  # 1:3 R:R for OBs
            else:
                stop = best_pattern.zone_top + (atr * 0.5)
                risk = stop - current_price
                target = current_price - (risk * 3)
                
        else:  # SWEEP
            # Liquidity Sweep: Stop beyond swept level + 0.25 ATR
            if signal_type == Signal.LONG:
                stop = current_price - (atr * 1.25)
                risk = current_price - stop
                target = current_price + (risk * 2)
            else:
                stop = current_price + (atr * 1.25)
                risk = stop - current_price
                target = current_price - (risk * 2)
        
        risk_reward = abs((target - current_price) / (stop - current_price))
        
        return {
            'stop': round(stop, 5),
            'target': round(target, 5),
            'risk_reward': round(risk_reward, 2),
            'pattern_type': best_pattern.pattern_type
        }
    
    def _prepare_features(self, data: pd.DataFrame) -> np.ndarray:
        """Prepare features for ML models (last N timesteps)"""
        # Extract last 240 timesteps (as used in training)
        # Assumes data has all required features
        feature_cols = [col for col in data.columns if col not in ['timestamp', 'date']]
        features = data[feature_cols].tail(240).values
        
        # Reshape for LSTM input: (1, timesteps, features)
        features = features.reshape(1, features.shape[0], features.shape[1])
        return features
    
    def _prepare_mtf_features(self, mtf_data: Dict[str, pd.DataFrame]) -> np.ndarray:
        """Prepare multi-timeframe features for confluence model"""
        # Stack features from all timeframes
        # This depends on your confluence model's input structure
        # Placeholder implementation:
        mtf_features = []
        for tf in ['1m', '5m', '15m', '1h', '4h']:
            if tf in mtf_data:
                df = mtf_data[tf]
                # Extract key features per timeframe
                tf_feats = df[['close', 'volume', 'RSI', 'ADX']].tail(20).values
                mtf_features.append(tf_feats)
        
        # Concatenate and reshape
        mtf_features = np.concatenate(mtf_features, axis=0)
        return mtf_features.reshape(1, -1, mtf_features.shape[-1])
    
    def _extract_pattern_features(
        self,
        pattern: ICTPattern,
        data: pd.DataFrame
    ) -> np.ndarray:
        """Extract features for pattern validator"""
        # Get last 100 candles + pattern metadata
        window = data.tail(100)
        
        # Convert to image-like 2D array (OHLC)
        ohlc = window[['open', 'high', 'low', 'close']].values
        
        # Pattern metadata as features
        metadata = np.array([
            1 if pattern.pattern_type == 'FVG' else 0,
            1 if pattern.pattern_type == 'OB' else 0,
            1 if pattern.pattern_type == 'SWEEP' else 0,
            1 if pattern.direction == 'bullish' else 0,
            window['volume'].mean(),
        ])
        
        # Combine and reshape for CNN-LSTM
        features = np.concatenate([ohlc.flatten(), metadata])
        return features.reshape(1, -1, 1)  # Adjust shape based on your model


if __name__ == "__main__":
    print("Ensemble Decision Engine - Ready for Integration")
    print("=" * 80)
    print("This module aggregates:")
    print("  ✓ ICT pattern detectors (FVG, OB, Sweep)")
    print("  ✓ ML models (Long/Short, Returns, Regime, Validation, Confluence)")
    print("  ✓ Risk management rules")
    print("\nOutput: TradingSignal with confidence, stops, targets, and position size")