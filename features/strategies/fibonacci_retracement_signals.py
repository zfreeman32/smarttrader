
import pandas as pd

# Fibonacci Retracement Trading Strategy
def fibonacci_retracement_signals(stock_df, window=14):
    signals = pd.DataFrame(index=stock_df.index)
    close = stock_df['Close']
    
    # Calculate the Fibonacci levels
    max_price = close.rolling(window=window).max()
    min_price = close.rolling(window=window).min()
    
    # Fibonacci levels
    levels = {
        'level_0': max_price,
        'level_1': max_price - (max_price - min_price) * 0.236,
        'level_2': max_price - (max_price - min_price) * 0.382,
        'level_3': max_price - (max_price - min_price) * 0.618,
        'level_4': min_price
    }
    
    # Initialize signals
    signals['fibonacci_signal'] = 'neutral'
    crossed_up = pd.Series(False, index=stock_df.index)
    crossed_down = pd.Series(False, index=stock_df.index)
    
    # Generate buy and sell signals based on Fibonacci retracement levels
    for level in levels.values():
        crossed_up |= (close.shift(1) < level.shift(1)) & (close >= level)
        crossed_down |= (close.shift(1) > level.shift(1)) & (close <= level)

    signals.loc[crossed_up, 'fibonacci_signal'] = 'long'
    signals.loc[crossed_down, 'fibonacci_signal'] = 'short'

    return signals
