import pandas as pd
import numpy as np

def mfi_signals__market_facilitation_index_trading_strategy_strategy(stock_df, volume_col='Volume', high_col='High', low_col='Low', close_col='Close'):
    signals = pd.DataFrame(index=stock_df.index)
    stock_df['Price_Range'] = stock_df[high_col] - stock_df[low_col]
    stock_df['MFI'] = (stock_df['Price_Range'] / stock_df[volume_col]).fillna(0)
    signals['mfi_signal'] = 'neutral'
    signals.loc[(stock_df['MFI'] > 0.5) & (stock_df['MFI'].shift(1) <= 0.5), 'mfi_signal'] = 'long'
    signals.loc[(stock_df['MFI'] < 0.2) & (stock_df['MFI'].shift(1) >= 0.2), 'mfi_signal'] = 'short'
    return signals
