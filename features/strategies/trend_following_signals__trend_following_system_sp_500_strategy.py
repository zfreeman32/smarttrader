import pandas as pd
from ta import trend

def trend_following_signals__trend_following_system_sp_500_strategy(stock_df):
    signals = pd.DataFrame(index=stock_df.index)
    signals['200_MA'] = trend.SMAIndicator(stock_df['Close'], window=200).sma_indicator()
    signals['trend_signal'] = 'neutral'
    signals.loc[(stock_df['Close'] > signals['200_MA']) & (stock_df['Close'].shift(1) <= signals['200_MA'].shift(1)), 'trend_signal'] = 'long'
    signals.loc[(stock_df['Close'] < signals['200_MA']) & (stock_df['Close'].shift(1) >= signals['200_MA'].shift(1)), 'trend_signal'] = 'short'
    return signals[['trend_signal']]
