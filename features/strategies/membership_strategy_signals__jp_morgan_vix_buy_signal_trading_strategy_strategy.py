import pandas as pd

def membership_strategy_signals__jp_morgan_vix_buy_signal_trading_strategy_strategy(stock_df):
    signals = pd.DataFrame(index=stock_df.index)
    signals['signal'] = 'neutral'
    signals.loc[stock_df['Close'] > stock_df['Close'].shift(1), 'signal'] = 'long'
    signals.loc[stock_df['Close'] < stock_df['Close'].shift(1), 'signal'] = 'short'
    return signals
