
import pandas as pd

# Price and Volume Trend (PVT) Strategy
def pvt_signals(stock_df):
    signals = pd.DataFrame(index=stock_df.index)
    if 'Volume' not in stock_df.columns:
        signals['pvt_signal'] = 'neutral'
        return signals
    
    # Calculate Price and Volume Trend (PVT)
    pvt = (stock_df['Volume'].astype(float) * stock_df['Close'].pct_change().fillna(0.0)).cumsum()
    
    # Create signal column
    signals['pvt_signal'] = 'neutral'
    
    # Generate buy/sell signals
    signals.loc[(pvt > pvt.shift(1)) & (pvt.shift(1) <= pvt.shift(2)), 'pvt_signal'] = 'long'
    signals.loc[(pvt < pvt.shift(1)) & (pvt.shift(1) >= pvt.shift(2)), 'pvt_signal'] = 'short'
    
    return signals
