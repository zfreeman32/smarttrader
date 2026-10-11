import pandas as pd


def ichimoku_strategy_causal_v2(stock_df):
    """Causal Ichimoku strategy (feature v2 replacement for ``ichimoku_signals__ichimoku_strategy_strategy``).

    v1 compared the Chikou span (``close.shift(-26)``, the close 26 bars ahead) with the current close. The causal form
    of that confirmation compares the current close with the close 26 bars ago. Tenkan/Kijun are trailing as before.
    """
    signals = pd.DataFrame(index=stock_df.index)
    close = stock_df['Close']
    tenkan = (stock_df['High'].rolling(window=9).max() + stock_df['Low'].rolling(window=9).min()) / 2
    kijun = (stock_df['High'].rolling(window=26).max() + stock_df['Low'].rolling(window=26).min()) / 2
    lagged_close = close.shift(26)
    signals['signal'] = 'neutral'
    signals.loc[(tenkan > kijun) & (close > lagged_close), 'signal'] = 'long'
    signals.loc[(tenkan < kijun) & (close < lagged_close), 'signal'] = 'short'
    return signals[['signal']]
