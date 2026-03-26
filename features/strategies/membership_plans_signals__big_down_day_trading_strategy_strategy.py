import pandas as pd
import numpy as np

def membership_plans_signals__big_down_day_trading_strategy_strategy(stock_df, bonus_strategies_count=15, monthly_strategies_count=10):
    """
    Generate trading signals based on the quantified trading strategies membership plans.
    
    Param:
    stock_df: DataFrame containing stock data with 'Close' prices
    bonus_strategies_count: Number of strategies chosen as a signup bonus
    monthly_strategies_count: Number of backtested strategies received monthly
    
    Returns:
    signals: DataFrame with trading signals
    """
    signals = pd.DataFrame(index=stock_df.index)
    if bonus_strategies_count > 10:
        signals['signal'] = 'long'
    else:
        signals['signal'] = 'short'
    if monthly_strategies_count < 5:
        signals['signal'] = 'neutral'
    return signals
