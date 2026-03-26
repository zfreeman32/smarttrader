import pandas as pd
from ta.volume import MFIIndicator

def mfi_signals__money_flow_index_strategy_strategy(stock_df, window=14, overbought=80, oversold=20):
    signals = pd.DataFrame(index=stock_df.index)
    mfi = MFIIndicator(stock_df['High'], stock_df['Low'], stock_df['Close'], stock_df['Volume'], window)
    signals['MFI'] = mfi.mfi()
    signals['mfi_signal'] = 'neutral'
    signals.loc[signals['MFI'] < oversold, 'mfi_signal'] = 'long'
    signals.loc[signals['MFI'] > overbought, 'mfi_signal'] = 'short'
    return signals[['mfi_signal']]
