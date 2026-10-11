import pandas as pd


def tesla_stock_boom_causal_v2(stock_df):
    """Causal Tesla-boom signal (feature v2 replacement for ``tesla_stock_boom_signals``).

    v1 marked the 'short' exit 6 bars *before* a long entry (``shift(-6)``). Here the exit is marked 6 bars after the
    entry, when it is actually known.
    """
    signals = pd.DataFrame(index=stock_df.index)
    price_change = stock_df['Close'].pct_change()
    entry = (price_change > 0).rolling(window=3).sum() == 3
    signals['signal'] = 'neutral'
    signals.loc[entry, 'signal'] = 'long'
    signals.loc[entry.shift(6, fill_value=False).astype(bool), 'signal'] = 'short'
    return signals[['signal']]
