import pandas as pd
import numpy as np
from ta import momentum, trend, volatility

def seasonal_strategy_signals__sp_500_and_gold_miners_rotation_system_strategy_strategy(stock_df, window=20):
    signals = pd.DataFrame(index=stock_df.index)
    signals['Avg_Return'] = stock_df['Close'].pct_change(periods=window).rolling(window).mean()
    signals['seasonal_signal'] = 'neutral'
    signals.loc[signals['Avg_Return'] > 0, 'seasonal_signal'] = 'long'
    signals.loc[signals['Avg_Return'] < 0, 'seasonal_signal'] = 'short'
    signals.drop(['Avg_Return'], axis=1, inplace=True)
    return signals
