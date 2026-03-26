import pandas as pd
import numpy as np

def membership_plan_signals__geopolitical_risk_trading_strategy_strategy(stock_df):
    """
    Generates trading signals based on quantified patterns and anomalies for membership plans.
    
    Parameters:
    stock_df (pd.DataFrame): DataFrame containing stock prices with a 'Close' column.
    
    Returns:
    pd.DataFrame: DataFrame containing trading signals ('long', 'short', 'neutral').
    """
    signals = pd.DataFrame(index=stock_df.index)
    signals['membership_edge'] = np.where(stock_df['Close'].pct_change() > 0.02, 'long', np.where(stock_df['Close'].pct_change() < -0.02, 'short', 'neutral'))
    signals.dropna(inplace=True)
    return signals[['membership_edge']]
