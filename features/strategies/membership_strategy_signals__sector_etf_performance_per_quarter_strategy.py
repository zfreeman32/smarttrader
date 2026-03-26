import pandas as pd
import numpy as np

def membership_strategy_signals__sector_etf_performance_per_quarter_strategy(stock_df):
    signals = pd.DataFrame(index=stock_df.index)
    signals['Price_Change'] = stock_df['Close'].pct_change()
    signals['Volatility'] = stock_df['Close'].rolling(window=14).std()
    signals['signal'] = 'neutral'
    signals.loc[(signals['Price_Change'] > 0.01) & (signals['Volatility'] < 0.02), 'signal'] = 'long'
    signals.loc[(signals['Price_Change'] < -0.01) & (signals['Volatility'] < 0.02), 'signal'] = 'short'
    return signals[['signal']]
