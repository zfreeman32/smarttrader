import pandas as pd
from ta import momentum

def momentum_reversal_signals__oscillating_indicators_trading_strategies_strategy(stock_df, lookback_period=14, threshold=0.05):
    signals = pd.DataFrame(index=stock_df.index)
    signals['momentum'] = stock_df['Close'].pct_change(periods=lookback_period)
    signals['signal'] = 'neutral'
    signals.loc[signals['momentum'] > threshold, 'signal'] = 'long'
    signals.loc[signals['momentum'] < -threshold, 'signal'] = 'short'
    return signals[['signal']]
