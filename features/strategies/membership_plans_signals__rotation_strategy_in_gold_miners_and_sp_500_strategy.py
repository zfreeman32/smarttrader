import pandas as pd
import numpy as np

def membership_plans_signals__rotation_strategy_in_gold_miners_and_sp_500_strategy(stock_df, entry_threshold=0.02, exit_threshold=0.01):
    """
    Generates trading signals based on the Membership Plans trading strategy.
    
    Parameters:
    stock_df (pd.DataFrame): DataFrame containing stock data with 'Close' prices.
    entry_threshold (float): The threshold for entering a long position.
    exit_threshold (float): The threshold for exiting a long position.
    
    Returns:
    pd.DataFrame: DataFrame with trading signals ('long', 'short', 'neutral').
    """
    signals = pd.DataFrame(index=stock_df.index)
    signals['Close'] = stock_df['Close']
    signals['Returns'] = signals['Close'].pct_change()
    signals['signal'] = 'neutral'
    signals.loc[signals['Returns'] > entry_threshold, 'signal'] = 'long'
    signals.loc[signals['Returns'] < -exit_threshold, 'signal'] = 'short'
    signals['signal'] = signals['signal'].replace({'neutral': np.nan}).ffill()
    return signals[['signal']]
