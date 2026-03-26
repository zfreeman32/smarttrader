import pandas as pd
import numpy as np

def membership_plan_signals__membership_plan_strategy(stock_df, period=14):
    """
    Generates trading signals based on monthly trading strategies from membership plans.
    
    Parameters:
    stock_df (pd.DataFrame): DataFrame containing stock prices with DateTime index and a 'Close' column.
    period (int): The number of periods for calculating moving averages.

    Returns:
    pd.DataFrame: DataFrame containing trading signals ('long', 'short', 'neutral') based on the strategy.
    """
    signals = pd.DataFrame(index=stock_df.index)
    signals['MA_Fast'] = stock_df['Close'].rolling(window=period).mean()
    signals['MA_Slow'] = stock_df['Close'].rolling(window=period * 2).mean()
    signals['membership_signal'] = 'neutral'
    signals.loc[signals['MA_Fast'] > signals['MA_Slow'], 'membership_signal'] = 'long'
    signals.loc[signals['MA_Fast'] < signals['MA_Slow'], 'membership_signal'] = 'short'
    return signals[['membership_signal']]
