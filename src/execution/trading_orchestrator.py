"""
Real-Time Trading System Orchestrator
Manages live data feeds, signal generation, and trade execution.
"""

import asyncio
import pandas as pd
import numpy as np
from typing import Dict, Optional, List
from datetime import datetime
import logging
from dataclasses import dataclass
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_system.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class Position:
    """Open position tracker"""
    entry_time: datetime
    entry_price: float
    stop_loss: float
    take_profit: float
    size: int
    direction: str  # 'LONG' or 'SHORT'
    ict_patterns: List[str]
    confidence: float
    current_pnl: float = 0.0
    current_stop: float = None
    
    def __post_init__(self):
        if self.current_stop is None:
            self.current_stop = self.stop_loss


class TradingSystemOrchestrator:
    """
    Main trading system coordinator.
    
    Responsibilities:
    1. Manage real-time data feeds
    2. Run ensemble signal generation
    3. Execute trades via broker API
    4. Monitor positions and update stops
    5. Log all activity
    """
    
    def __init__(
        self,
        ensemble_engine,
        risk_manager,
        broker_interface,
        data_provider,
        config: Dict
    ):
        """
        Initialize trading system.
        
        Args:
            ensemble_engine: EnsembleDecisionEngine instance
            risk_manager: RiskManager instance
            broker_interface: Broker API wrapper (e.g., IB, MT5)
            data_provider: Real-time data feed provider
            config: System configuration dict
        """
        self.ensemble = ensemble_engine
        self.risk_mgr = risk_manager
        self.broker = broker_interface
        self.data_provider = data_provider
        self.config = config
        
        # State tracking
        self.open_positions: List[Position] = []
        self.pending_orders: Dict = {}
        self.last_signal_time: Optional[datetime] = None
        self.is_running = False
        
        # Performance tracking
        self.signals_generated = 0
        self.trades_executed = 0
        self.total_pnl = 0.0
        
        logger.info("Trading System Orchestrator initialized")
        logger.info(f"Config: {json.dumps(config, indent=2)}")
    
    async def start(self):
        """Start the trading system"""
        logger.info("=" * 80)
        logger.info("STARTING TRADING SYSTEM")
        logger.info("=" * 80)
        
        self.is_running = True
        
        # Start parallel tasks
        await asyncio.gather(
            self._data_feed_loop(),
            self._signal_generation_loop(),
            self._position_monitoring_loop(),
            self._heartbeat_loop()
        )
    
    async def stop(self):
        """Gracefully stop the trading system"""
        logger.info("Stopping trading system...")
        self.is_running = False
        
        # Close all open positions
        for pos in self.open_positions:
            await self._close_position(pos, reason="System shutdown")
        
        logger.info("Trading system stopped")
    
    async def _data_feed_loop(self):
        """Continuously fetch and process live market data"""
        logger.info("Starting data feed loop...")
        
        while self.is_running:
            try:
                # Fetch latest bar from data provider
                current_bar = await self.data_provider.get_latest_bar('EURUSD', '1m')
                
                if current_bar is not None:
                    # Update internal data buffers
                    self._update_data_buffer(current_bar)
                
                # Sleep until next bar (for 1m bars, check every 5 seconds)
                await asyncio.sleep(5)
                
            except Exception as e:
                logger.error(f"Error in data feed loop: {str(e)}", exc_info=True)
                await asyncio.sleep(10)  # Wait before retry
    
    async def _signal_generation_loop(self):
        """Generate trading signals on new bars"""
        logger.info("Starting signal generation loop...")
        
        last_bar_time = None
        
        while self.is_running:
            try:
                # Get current data
                current_data = self.data_provider.get_dataframe('EURUSD', '1m', bars=500)
                mtf_data = {
                    '1m': current_data,
                    '5m': self.data_provider.get_dataframe('EURUSD', '5m', bars=200),
                    '15m': self.data_provider.get_dataframe('EURUSD', '15m', bars=100),
                    '1h': self.data_provider.get_dataframe('EURUSD', '1h', bars=50),
                    '4h': self.data_provider.get_dataframe('EURUSD', '4h', bars=25)
                }
                
                # Check if new bar has formed
                current_bar_time = current_data.index[-1]
                if current_bar_time != last_bar_time:
                    last_bar_time = current_bar_time
                    logger.info(f"\nNew bar at {current_bar_time}")
                    
                    # Generate signal
                    await self._generate_and_execute_signal(current_data, mtf_data)
                
                # Check every 5 seconds
                await asyncio.sleep(5)
                
            except Exception as e:
                logger.error(f"Error in signal generation loop: {str(e)}", exc_info=True)
                await asyncio.sleep(10)
    
    async def _generate_and_execute_signal(
        self,
        current_data: pd.DataFrame,
        mtf_data: Dict[str, pd.DataFrame]
    ):
        """Generate signal and execute if valid"""
        
        try:
            # Get current market conditions
            current_price = current_data['close'].iloc[-1]
            current_spread = self._calculate_spread(current_data)
            current_atr = current_data['ATR'].iloc[-1] if 'ATR' in current_data.columns else 0.0010
            
            # Pre-trade risk checks
            can_trade = self.risk_mgr.check_can_trade(
                current_positions=len(self.open_positions),
                current_spread=current_spread,
                current_atr=current_atr,
                atr_history=current_data['ATR'].values if 'ATR' in current_data.columns else np.array([]),
                upcoming_news=None  # TODO: Integrate economic calendar
            )
            
            if not can_trade['can_trade']:
                logger.info(f"Cannot trade: {can_trade['reasons']}")
                return
            
            # Generate signal from ensemble
            signal = self.ensemble.generate_signal(
                current_data=current_data,
                mtf_data=mtf_data,
                current_price=current_price,
                account_balance=self.risk_mgr.current_balance,
                current_positions=len(self.open_positions)
            )
            
            self.signals_generated += 1
            
            if signal is None:
                logger.debug("No trading signal generated")
                return
            
            logger.info("=" * 80)
            logger.info(f"SIGNAL GENERATED: {signal.signal.name}")
            logger.info(f"Confidence: {signal.confidence:.3f}")
            logger.info(f"Entry: {signal.entry_price:.5f}")
            logger.info(f"Stop: {signal.stop_loss:.5f}")
            logger.info(f"Target: {signal.take_profit:.5f}")
            logger.info(f"Size: {signal.position_size} contracts")
            logger.info(f"R:R: {signal.metadata['risk_reward']:.2f}")
            logger.info("=" * 80)
            
            # Execute trade
            await self._execute_trade(signal)
            
        except Exception as e:
            logger.error(f"Error generating/executing signal: {str(e)}", exc_info=True)
    
    async def _execute_trade(self, signal):
        """Execute trade via broker"""
        
        try:
            # Validate one more time before execution
            rr_check = self.risk_mgr.validate_risk_reward(
                signal.entry_price,
                signal.stop_loss,
                signal.take_profit
            )
            
            if not rr_check['valid']:
                logger.warning(f"Trade rejected: {rr_check['reason']}")
                return
            
            # Place order via broker
            order_result = await self.broker.place_order(
                symbol='EURUSD',
                direction=signal.signal.name,
                size=signal.position_size,
                entry_price=signal.entry_price,
                stop_loss=signal.stop_loss,
                take_profit=signal.take_profit
            )
            
            if order_result['success']:
                # Create position tracker
                position = Position(
                    entry_time=datetime.now(),
                    entry_price=order_result['fill_price'],
                    stop_loss=signal.stop_loss,
                    take_profit=signal.take_profit,
                    size=signal.position_size,
                    direction=signal.signal.name,
                    ict_patterns=[p.pattern_type for p in signal.ict_patterns],
                    confidence=signal.confidence
                )
                
                self.open_positions.append(position)
                self.trades_executed += 1
                
                logger.info(f"✅ TRADE EXECUTED: {order_result}")
                logger.info(f"Order ID: {order_result['order_id']}")
                logger.info(f"Fill price: {order_result['fill_price']:.5f}")
                
                # Log to trade journal
                self._log_trade(signal, order_result)
                
            else:
                logger.error(f"❌ TRADE FAILED: {order_result['error']}")
                
        except Exception as e:
            logger.error(f"Error executing trade: {str(e)}", exc_info=True)
    
    async def _position_monitoring_loop(self):
        """Monitor open positions and update trailing stops"""
        logger.info("Starting position monitoring loop...")
        
        while self.is_running:
            try:
                if not self.open_positions:
                    await asyncio.sleep(5)
                    continue
                
                # Get current price
                current_price = await self.broker.get_current_price('EURUSD')
                
                for position in self.open_positions[:]:  # Copy list to allow removal
                    # Update P&L
                    if position.direction == 'LONG':
                        position.current_pnl = (current_price - position.entry_price) * position.size * 100000  # For micro lots
                    else:
                        position.current_pnl = (position.entry_price - current_price) * position.size * 100000
                    
                    # Check if hit stop or target
                    if position.direction == 'LONG':
                        if current_price <= position.current_stop:
                            await self._close_position(position, reason="Stop loss hit")
                            continue
                        if current_price >= position.take_profit:
                            await self._close_position(position, reason="Take profit hit")
                            continue
                    else:  # SHORT
                        if current_price >= position.current_stop:
                            await self._close_position(position, reason="Stop loss hit")
                            continue
                        if current_price <= position.take_profit:
                            await self._close_position(position, reason="Take profit hit")
                            continue
                    
                    # Update trailing stop
                    current_data = self.data_provider.get_dataframe('EURUSD', '1m', bars=50)
                    atr = current_data['ATR'].iloc[-1] if 'ATR' in current_data.columns else 0.0010
                    
                    new_stop = self.risk_mgr.calculate_trailing_stop(
                        entry_price=position.entry_price,
                        current_price=current_price,
                        initial_stop=position.stop_loss,
                        atr=atr,
                        is_long=(position.direction == 'LONG')
                    )
                    
                    if new_stop != position.current_stop:
                        position.current_stop = new_stop
                        await self.broker.modify_stop_loss('EURUSD', new_stop)
                        logger.info(f"Trailing stop updated: {new_stop:.5f} (P&L: ${position.current_pnl:.2f})")
                
                await asyncio.sleep(5)
                
            except Exception as e:
                logger.error(f"Error in position monitoring: {str(e)}", exc_info=True)
                await asyncio.sleep(10)
    
    async def _close_position(self, position: Position, reason: str):
        """Close an open position"""
        try:
            close_result = await self.broker.close_position('EURUSD', position.size)
            
            if close_result['success']:
                final_pnl = position.current_pnl
                self.total_pnl += final_pnl
                
                logger.info("=" * 80)
                logger.info(f"POSITION CLOSED: {reason}")
                logger.info(f"Entry: {position.entry_price:.5f} → Exit: {close_result['exit_price']:.5f}")
                logger.info(f"P&L: ${final_pnl:.2f}")
                logger.info(f"Total P&L: ${self.total_pnl:.2f}")
                logger.info("=" * 80)
                
                # Update risk manager
                self.risk_mgr.metrics.add_trade(final_pnl, {
                    'entry_time': position.entry_time,
                    'exit_time': datetime.now(),
                    'entry_price': position.entry_price,
                    'exit_price': close_result['exit_price'],
                    'direction': position.direction,
                    'size': position.size,
                    'patterns': position.ict_patterns,
                    'confidence': position.confidence
                })
                
                # Remove from open positions
                self.open_positions.remove(position)
                
        except Exception as e:
            logger.error(f"Error closing position: {str(e)}", exc_info=True)
    
    async def _heartbeat_loop(self):
        """Regular system health checks and status updates"""
        logger.info("Starting heartbeat loop...")
        
        while self.is_running:
            try:
                # Print status every 5 minutes
                await asyncio.sleep(300)
                
                risk_report = self.risk_mgr.get_risk_report()
                
                logger.info("=" * 80)
                logger.info("SYSTEM STATUS")
                logger.info("=" * 80)
                logger.info(f"Balance: ${risk_report['balance']:.2f}")
                logger.info(f"Daily P&L: ${risk_report['daily_pnl']:.2f} ({risk_report['daily_pnl_pct']:.2f}%)")
                logger.info(f"Open positions: {len(self.open_positions)}")
                logger.info(f"Signals generated: {self.signals_generated}")
                logger.info(f"Trades executed: {self.trades_executed}")
                logger.info(f"Total P&L: ${self.total_pnl:.2f}")
                logger.info(f"Win rate: {risk_report['recent_win_rate']:.1%}")
                logger.info(f"Risk health: {risk_report['risk_health']}")
                logger.info("=" * 80)
                
            except Exception as e:
                logger.error(f"Error in heartbeat: {str(e)}", exc_info=True)
    
    def _update_data_buffer(self, bar: Dict):
        """Update internal data buffers with new bar"""
        # Implementation depends on your data provider
        pass
    
    def _calculate_spread(self, data: pd.DataFrame) -> float:
        """Calculate current bid-ask spread in pips"""
        # Simplified - actual implementation depends on broker feed
        return 1.5  # Default 1.5 pips
    
    def _log_trade(self, signal, order_result):
        """Log trade details to journal"""
        trade_record = {
            'timestamp': datetime.now().isoformat(),
            'signal': signal.signal.name,
            'confidence': signal.confidence,
            'entry_price': order_result['fill_price'],
            'stop_loss': signal.stop_loss,
            'take_profit': signal.take_profit,
            'size': signal.position_size,
            'patterns': [p.pattern_type for p in signal.ict_patterns],
            'regime': signal.regime,
            'confluence': signal.confluence_score,
            'order_id': order_result['order_id']
        }
        
        # Append to JSON log file
        try:
            with open('trade_journal.json', 'a') as f:
                f.write(json.dumps(trade_record) + '\n')
        except Exception as e:
            logger.error(f"Error logging trade: {str(e)}")


# Example main entry point
async def main():
    """
    Main entry point for live trading system.
    
    TODO: Implement actual broker connection and data providers.
    """
    
    # Configuration
    config = {
        'symbol': 'EURUSD',
        'timeframe': '1m',
        'initial_balance': 10000.0,
        'max_daily_loss': 500.0,
        'max_positions': 2,
        'live_mode': False  # Set to True for real trading
    }
    
    logger.info("Initializing trading system components...")
    
    # TODO: Load trained models
    # long_classifier = load_model('models/long_signal_classifier.keras')
    # short_classifier = load_model('models/short_signal_classifier.keras')
    # ... etc
    
    # TODO: Initialize components
    # ensemble = EnsembleDecisionEngine(...)
    # risk_mgr = RiskManager(...)
    # broker = BrokerInterface(...)
    # data_provider = DataProvider(...)
    
    # TODO: Create orchestrator
    # system = TradingSystemOrchestrator(
    #     ensemble_engine=ensemble,
    #     risk_manager=risk_mgr,
    #     broker_interface=broker,
    #     data_provider=data_provider,
    #     config=config
    # )
    
    # TODO: Start system
    # await system.start()
    
    logger.info("Trading system setup complete (example)")
    print("\n" + "=" * 80)
    print("TRADING SYSTEM ORCHESTRATOR")
    print("=" * 80)
    print("\nThis module provides:")
    print("  ✓ Real-time data feed management")
    print("  ✓ Automated signal generation")
    print("  ✓ Trade execution via broker API")
    print("  ✓ Position monitoring and trailing stops")
    print("  ✓ Risk management integration")
    print("  ✓ System health monitoring")
    print("\nNext steps:")
    print("  1. Integrate with your broker API (IB, MT5, etc.)")
    print("  2. Connect to real-time data provider")
    print("  3. Load your trained models")
    print("  4. Test in paper trading mode")
    print("  5. Deploy to live trading")


if __name__ == "__main__":
    asyncio.run(main())