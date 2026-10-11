import pandas as pd


def high_volume_points_causal_v2(stock_df, left_bars=15, filter_vol=2.0, lookback_period=300):
    """Causal high-volume pivots (feature v2 replacement for ``high_volume_points``).

    The v1 strategy marked a pivot at its own bar using a centered window (the next ``left_bars`` bars). Here a pivot at
    bar p is reported at p + ``left_bars``, the first bar at which its right side is known; every window is trailing.
    """
    signals = pd.DataFrame(index=stock_df.index)
    width = left_bars * 2 + 1
    high, low, close = stock_df['High'], stock_df['Low'], stock_df['Close']
    candidate_high = high.shift(left_bars)
    candidate_low = low.shift(left_bars)
    pivot_high = candidate_high.where(candidate_high == high.rolling(window=width).max())
    pivot_low = candidate_low.where(candidate_low == low.rolling(window=width).min())
    rolling_volume = stock_df['Volume'].rolling(window=left_bars * 2).sum()
    reference_vol = rolling_volume.rolling(lookback_period).quantile(0.95)
    norm_vol = rolling_volume / reference_vol * 5
    valid_high = pivot_high.notna() & (norm_vol > filter_vol)
    valid_low = pivot_low.notna() & (norm_vol > filter_vol)
    signals['high_volume_pivot'] = valid_high | valid_low
    signals['liquidity_grab'] = ((valid_high & (close < pivot_high.shift(1)))
                                 | (valid_low & (close > pivot_low.shift(1))))
    return signals
