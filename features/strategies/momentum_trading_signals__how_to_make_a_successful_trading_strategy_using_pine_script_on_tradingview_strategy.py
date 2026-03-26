import pandas as pd
import numpy as np
from ta import momentum, trend

def momentum_trading_signals__how_to_make_a_successful_trading_strategy_using_pine_script_on_tradingview_strategy(stock_df, window=14):
    signals = pd.DataFrame(index=stock_df.index)
    signals['Momentum'] = momentum.RSIIndicator(stock_df['Close'], window).rsi()
    signals['signal'] = 'neutral'
    signals.loc[signals['Momentum'] > 70, 'signal'] = 'short'
    signals.loc[signals['Momentum'] < 30, 'signal'] = 'long'
    return signals[['signal']]
