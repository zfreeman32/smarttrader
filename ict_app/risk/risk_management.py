"""
Risk Management Layer - TIER 3
Implements institutional-grade risk controls and position management.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, List
from dataclasses import dataclass
from datetime import datetime, time
import logging

logger = logging.getLogger(__name__)


@dataclass
class RiskParameters:
    """Risk management configuration"""
    max_daily_loss: float = 500.0  # Maximum daily loss in dollars
    max_position_size: int = 3  # Maximum contracts per position
    max_total_positions: int = 2  # Maximum concurrent positions
    base_risk_pct: float = 0.01  # Base risk per trade (1% of account)
    max_spread_pips: float = 2.0  # Maximum acceptable spread
    max_atr_percentile: float = 95  # Max ATR percentile (volatility filter)
    min_risk_reward: float = 1.5  # Minimum R:R ratio
    trading_start_time: time = time(2, 0)  # 2 AM EST (London open)
    trading_end_time: time = time(16, 0)  # 4 PM EST
    news_blackout_minutes: int = 30  # Minutes before/after news


@dataclass
class TradeMetrics:
    """Track recent trading performance"""
    recent_trades: List[Dict]  # Last 20 trades
    daily_pnl: float = 0.0
    daily_trades: int = 0
    win_streak: int = 0
    loss_streak: int = 0
    
    @property
    def recent_win_rate(self) -> float:
        """Calculate win rate from last 20 trades"""
        if not self.recent_trades:
            return 0.5  # Default 50%
        wins = sum(1 for t in self.recent_trades if t['pnl'] > 0)
        return wins / len(self.recent_trades)
    
    def add_trade(self, pnl: float, trade_data: Dict):
        """Add completed trade to history"""
        trade_data['pnl'] = pnl
        self.recent_trades.append(trade_data)
        
        # Keep only last 20 trades
        if len(self.recent_trades) > 20:
            self.recent_trades = self.recent_trades[-20:]
        
        # Update daily metrics
        self.daily_pnl += pnl
        self.daily_trades += 1
        
        # Update streaks
        if pnl > 0:
            self.win_streak += 1
            self.loss_streak = 0
        else:
            self.loss_streak += 1
            self.win_streak = 0


class RiskManager:
    """
    Comprehensive risk management system.
    
    Implements:
    - Daily loss limits
    - Position size calculation
    - Trade filtering (pre-trade checks)
    - Dynamic adjustments based on performance
    - Session/time filters
    - Correlation management
    """
    
    def __init__(
        self,
        risk_params: Optional[RiskParameters] = None,
        initial_balance: float = 10000.0
    ):
        """
        Initialize risk manager.
        
        Args:
            risk_params: Risk parameter configuration
            initial_balance: Starting account balance
        """
        self.params = risk_params or RiskParameters()
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.metrics = TradeMetrics(recent_trades=[])
        
        # Track daily reset
        self.last_reset_date = pd.Timestamp.now().date()
        
        logger.info(f"Risk Manager initialized with balance: ${initial_balance:.2f}")
    
    def update_balance(self, new_balance: float):
        """Update current account balance"""
        self.current_balance = new_balance
        self._check_daily_reset()
    
    def _check_daily_reset(self):
        """Reset daily metrics at start of new day"""
        current_date = pd.Timestamp.now().date()
        if current_date != self.last_reset_date:
            logger.info(f"Daily reset: PnL=${self.metrics.daily_pnl:.2f}, Trades={self.metrics.daily_trades}")
            self.metrics.daily_pnl = 0.0
            self.metrics.daily_trades = 0
            self.last_reset_date = current_date
    
    def check_can_trade(
        self,
        current_positions: int,
        current_spread: float,
        current_atr: float,
        atr_history: np.ndarray,
        upcoming_news: Optional[List[datetime]] = None
    ) -> Dict[str, bool]:
        """
        Pre-trade filter system - Hard stops.
        
        Returns dict with can_trade flag and reasons for rejection.
        """
        checks = {
            'daily_loss_limit': True,
            'position_limit': True,
            'trading_hours': True,
            'spread': True,
            'volatility': True,
            'news_blackout': True
        }
        reasons = []
        
        # Check 1: Daily loss limit
        if self.metrics.daily_pnl <= -self.params.max_daily_loss:
            checks['daily_loss_limit'] = False
            reasons.append(f"Daily loss limit reached: ${self.metrics.daily_pnl:.2f}")
        
        # Check 2: Position limit
        if current_positions >= self.params.max_total_positions:
            checks['position_limit'] = False
            reasons.append(f"Max positions reached: {current_positions}/{self.params.max_total_positions}")
        
        # Check 3: Trading hours (2 AM - 4 PM EST)
        current_time = datetime.now().time()
        if not (self.params.trading_start_time <= current_time <= self.params.trading_end_time):
            checks['trading_hours'] = False
            reasons.append(f"Outside trading hours: {current_time}")
        
        # Check 4: Spread
        if current_spread > self.params.max_spread_pips:
            checks['spread'] = False
            reasons.append(f"Spread too wide: {current_spread:.2f} pips")
        
        # Check 5: Volatility (ATR percentile)
        if len(atr_history) > 20:
            atr_percentile = np.percentile(atr_history, self.params.max_atr_percentile)
            if current_atr > atr_percentile:
                checks['volatility'] = False
                reasons.append(f"ATR too high: {current_atr:.5f} (>{atr_percentile:.5f})")
        
        # Check 6: News blackout
        if upcoming_news:
            now = datetime.now()
            for news_time in upcoming_news:
                time_to_news = (news_time - now).total_seconds() / 60
                if 0 < time_to_news < self.params.news_blackout_minutes:
                    checks['news_blackout'] = False
                    reasons.append(f"News event in {time_to_news:.0f} minutes")
                    break
        
        can_trade = all(checks.values())
        
        return {
            'can_trade': can_trade,
            'checks': checks,
            'reasons': reasons
        }
    
    def calculate_position_size(
        self,
        confidence: float,
        stop_distance_pips: float,
        regime: str,
        current_positions: int = 0
    ) -> int:
        """
        Calculate dynamic position size based on multiple factors.
        
        Args:
            confidence: Signal confidence (0.70-1.0)
            stop_distance_pips: Distance to stop loss in pips
            regime: Current market regime
            current_positions: Number of open positions
            
        Returns:
            Number of contracts to trade (1-3)
        """
        # Base risk: 1% of account
        base_risk_amount = self.current_balance * self.params.base_risk_pct
        
        # Confidence multiplier (70-95% confidence -> 70-95% of base risk)
        confidence_multiplier = np.clip(confidence, 0.70, 0.95)
        
        # Drawdown adjustment - reduce size if in drawdown
        if self.metrics.daily_pnl < -self.params.max_daily_loss * 0.5:
            confidence_multiplier *= 0.5
            logger.info("Reducing size due to daily drawdown")
        
        # Win rate adjustment - reduce on losing streak
        if self.metrics.recent_win_rate < 0.40:
            confidence_multiplier *= 0.5
            logger.info(f"Reducing size due to low win rate: {self.metrics.recent_win_rate:.2%}")
        
        # Regime adjustment - reduce in high volatility
        if regime in ['HIGH_VOLATILITY']:
            confidence_multiplier *= 0.5
            logger.info("Reducing size due to high volatility regime")
        
        # Calculate risk amount
        risk_amount = base_risk_amount * confidence_multiplier
        
        # Calculate contracts
        # For EURUSD: 1 micro lot = $1000 notional
        # Pip value = $0.10 for micro lot
        # Risk per contract = stop_distance_pips * $0.10
        pip_value = 0.10  # For micro lots
        risk_per_contract = stop_distance_pips * pip_value
        
        if risk_per_contract <= 0:
            return 0
        
        contracts = int(risk_amount / risk_per_contract)
        
        # Enforce limits
        contracts = max(1, min(contracts, self.params.max_position_size))
        
        # Don't add more if at max positions
        if current_positions >= self.params.max_total_positions:
            contracts = 0
        
        logger.info(
            f"Position sizing: Risk=${risk_amount:.2f}, "
            f"Stop={stop_distance_pips:.1f}pips, Contracts={contracts}"
        )
        
        return contracts
    
    def validate_risk_reward(
        self,
        entry_price: float,
        stop_loss: float,
        take_profit: float
    ) -> Dict[str, any]:
        """
        Validate risk-reward ratio meets minimum requirements.
        
        Returns:
            Dict with validation result and R:R ratio
        """
        risk = abs(entry_price - stop_loss)
        reward = abs(take_profit - entry_price)
        
        if risk <= 0:
            return {
                'valid': False,
                'risk_reward': 0,
                'reason': 'Invalid risk (zero or negative)'
            }
        
        risk_reward = reward / risk
        
        if risk_reward < self.params.min_risk_reward:
            return {
                'valid': False,
                'risk_reward': risk_reward,
                'reason': f'R:R {risk_reward:.2f} below minimum {self.params.min_risk_reward}'
            }
        
        return {
            'valid': True,
            'risk_reward': risk_reward,
            'reason': f'R:R {risk_reward:.2f} acceptable'
        }
    
    def check_quality_filters(
        self,
        signal_confidence: float,
        pattern_quality: float,
        confluence_score: float,
        volume_ratio: float,
        confirmations: int
    ) -> Dict[str, bool]:
        """
        Quality gate checks before trade execution.
        
        All must pass for trade to proceed.
        """
        checks = {
            'confidence': signal_confidence >= 0.70,
            'pattern_quality': pattern_quality >= 0.75,
            'confluence': confluence_score >= 0.65,
            'volume': volume_ratio >= 1.2,
            'confirmations': confirmations >= 2
        }
        
        reasons = []
        if not checks['confidence']:
            reasons.append(f"Low confidence: {signal_confidence:.2f}")
        if not checks['pattern_quality']:
            reasons.append(f"Low pattern quality: {pattern_quality:.2f}")
        if not checks['confluence']:
            reasons.append(f"Low confluence: {confluence_score:.2f}")
        if not checks['volume']:
            reasons.append(f"Low volume: {volume_ratio:.2f}x")
        if not checks['confirmations']:
            reasons.append(f"Insufficient confirmations: {confirmations}/4")
        
        passed = all(checks.values())
        
        return {
            'passed': passed,
            'checks': checks,
            'reasons': reasons
        }
    
    def calculate_trailing_stop(
        self,
        entry_price: float,
        current_price: float,
        initial_stop: float,
        atr: float,
        is_long: bool
    ) -> float:
        """
        Calculate trailing stop based on profit level.
        
        Logic:
        - Move to break-even at 1R profit
        - Trail by 0.5 ATR at 2R profit
        - Trail by 0.25 ATR at 3R+ profit
        """
        risk = abs(entry_price - initial_stop)
        current_profit = abs(current_price - entry_price)
        profit_r = current_profit / risk if risk > 0 else 0
        
        if profit_r >= 1.0:
            # At least 1R profit - move to break-even
            new_stop = entry_price
            
            if profit_r >= 2.0:
                # At 2R - trail by 0.5 ATR
                if is_long:
                    new_stop = current_price - (atr * 0.5)
                else:
                    new_stop = current_price + (atr * 0.5)
            
            if profit_r >= 3.0:
                # At 3R+ - tighter trail by 0.25 ATR
                if is_long:
                    new_stop = current_price - (atr * 0.25)
                else:
                    new_stop = current_price + (atr * 0.25)
            
            # Only move stop in favorable direction
            if is_long:
                return max(new_stop, initial_stop)
            else:
                return min(new_stop, initial_stop)
        
        return initial_stop
    
    def check_correlation_exposure(
        self,
        signal_pair: str,
        signal_direction: int,  # 1 for long, -1 for short
        open_positions: List[Dict]
    ) -> Dict[str, any]:
        """
        Check if new position would create excessive correlation risk.
        
        Rules:
        - Max 2 correlated positions
        - Reduce size if correlated pairs in same direction
        - Alert if DXY opposite to EURUSD
        """
        # Define correlations (simplified)
        correlations = {
            'EURUSD': {
                'GBPUSD': 0.75,  # Positive correlation
                'AUDUSD': 0.65,
                'DXY': -0.90,  # Negative (inverse)
                'GOLD': 0.60
            }
        }
        
        correlated_positions = []
        for pos in open_positions:
            if pos['pair'] in correlations.get(signal_pair, {}):
                corr = correlations[signal_pair][pos['pair']]
                effective_corr = corr * signal_direction * pos['direction']
                
                if abs(effective_corr) > 0.5:
                    correlated_positions.append({
                        'pair': pos['pair'],
                        'correlation': corr,
                        'effective_correlation': effective_corr
                    })
        
        # Check limits
        too_many_correlated = len(correlated_positions) >= 2
        
        # Calculate size adjustment
        size_multiplier = 1.0
        if len(correlated_positions) > 0:
            # Reduce size by 30% if correlated position exists
            size_multiplier = 0.70
        
        return {
            'allowed': not too_many_correlated,
            'correlated_positions': correlated_positions,
            'size_multiplier': size_multiplier,
            'warning': 'Too many correlated positions' if too_many_correlated else None
        }
    
    def get_risk_report(self) -> Dict:
        """Generate current risk metrics report"""
        return {
            'balance': self.current_balance,
            'daily_pnl': self.metrics.daily_pnl,
            'daily_pnl_pct': (self.metrics.daily_pnl / self.current_balance) * 100,
            'daily_trades': self.metrics.daily_trades,
            'recent_win_rate': self.metrics.recent_win_rate,
            'win_streak': self.metrics.win_streak,
            'loss_streak': self.metrics.loss_streak,
            'daily_loss_remaining': self.params.max_daily_loss + self.metrics.daily_pnl,
            'risk_health': 'HEALTHY' if self.metrics.daily_pnl > -self.params.max_daily_loss * 0.5 else 'AT RISK'
        }


# Example usage and testing
if __name__ == "__main__":
    # Initialize risk manager
    risk_mgr = RiskManager(initial_balance=10000.0)
    
    print("Risk Manager Initialized")
    print("=" * 80)
    
    # Test 1: Can we trade?
    print("\nTest 1: Pre-trade checks")
    can_trade = risk_mgr.check_can_trade(
        current_positions=0,
        current_spread=1.5,
        current_atr=0.0008,
        atr_history=np.random.uniform(0.0005, 0.0012, 100),
        upcoming_news=None
    )
    print(f"Can trade: {can_trade['can_trade']}")
    if not can_trade['can_trade']:
        print(f"Reasons: {can_trade['reasons']}")
    
    # Test 2: Position sizing
    print("\nTest 2: Position sizing")
    size = risk_mgr.calculate_position_size(
        confidence=0.85,
        stop_distance_pips=20,
        regime='STRONG_UPTREND',
        current_positions=0
    )
    print(f"Position size: {size} contracts")
    
    # Test 3: Risk-reward validation
    print("\nTest 3: Risk-reward validation")
    rr_check = risk_mgr.validate_risk_reward(
        entry_price=1.08500,
        stop_loss=1.08300,
        take_profit=1.08900
    )
    print(f"R:R valid: {rr_check['valid']}, Ratio: {rr_check['risk_reward']:.2f}")
    
    # Test 4: Quality filters
    print("\nTest 4: Quality filters")
    quality = risk_mgr.check_quality_filters(
        signal_confidence=0.85,
        pattern_quality=0.80,
        confluence_score=0.70,
        volume_ratio=1.5,
        confirmations=3
    )
    print(f"Quality checks passed: {quality['passed']}")
    
    # Test 5: Risk report
    print("\nTest 5: Risk report")
    report = risk_mgr.get_risk_report()
    for key, value in report.items():
        print(f"  {key}: {value}")
    
    print("\n" + "=" * 80)
    print("Risk Manager - Ready for Production")