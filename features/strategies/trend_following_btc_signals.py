
import pandas as pd
from ta import trend

# Trend Following Strategy using a 20-day Exponential Moving Average
def trend_following_btc_signals(btc_df, short_window=20):
    signals = pd.DataFrame(index=btc_df.index)
    close = btc_df['Close']
    signals['EWM'] = trend.EMAIndicator(close, window=short_window).ema_indicator()
    
    # Initialize the signal column
    signals['signal'] = 'neutral'
    
    # Create long and short signals
    signals.loc[close > signals['EWM'], 'signal'] = 'long'
    signals.loc[close < signals['EWM'], 'signal'] = 'short'
    
    return signals[['signal']]
