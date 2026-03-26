import pandas as pd
import numpy as np

def williams_r_signals__williams_r_trading_strategy_strategy(stock_df, lookback_period=5, overbought=-20, oversold=-80):
    """
    Generates trading signals based on the Williams %R indicator.
    
    Parameters:
        stock_df (pd.DataFrame): DataFrame containing stock price data with a 'Close' column.
        lookback_period (int): The lookback period for calculating Williams %R.
        overbought (float): The threshold for the overbought condition.
        oversold (float): The threshold for the oversold condition.
    
    Returns:
        pd.DataFrame: DataFrame with 'williams_r' and 'signal' columns.
    """
    signals = pd.DataFrame(index=stock_df.index)
    highest_high = stock_df['Close'].rolling(window=lookback_period).max()
    lowest_low = stock_df['Close'].rolling(window=lookback_period).min()
    signals['williams_r'] = (highest_high - stock_df['Close']) / (highest_high - lowest_low) * -100
    signals['signal'] = 'neutral'
    signals.loc[signals['williams_r'] < overbought, 'signal'] = 'short'
    signals.loc[signals['williams_r'] > oversold, 'signal'] = 'long'
    return signals[['williams_r', 'signal']]
