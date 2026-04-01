
import pandas as pd

# Negative Volume Index (NVI) Strategy
def nvi_signals(stock_df):
    signals = pd.DataFrame(index=stock_df.index)
    if 'Volume' not in stock_df.columns:
        signals['nvi_signal'] = 'neutral'
        return signals[['nvi_signal']]
    
    volume = stock_df['Volume'].astype(float)
    close_returns = stock_df['Close'].pct_change().fillna(0.0)

    signals['NVI'] = 1000.0
    # Calculate NVI based on volume changes
    for i in range(1, len(stock_df)):
        if volume.iloc[i] < volume.iloc[i - 1]:
            signals.iloc[i, signals.columns.get_loc('NVI')] = signals.iloc[i - 1, signals.columns.get_loc('NVI')] * (1.0 + close_returns.iloc[i])
        else:
            signals.iloc[i, signals.columns.get_loc('NVI')] = signals.iloc[i - 1, signals.columns.get_loc('NVI')]
    
    # Generate trading signals based on NVI
    signals['nvi_signal'] = 'neutral'
    signals.loc[(signals['NVI'] > signals['NVI'].shift(1)), 'nvi_signal'] = 'long'
    signals.loc[(signals['NVI'] < signals['NVI'].shift(1)), 'nvi_signal'] = 'short'
    
    return signals[['nvi_signal']]
