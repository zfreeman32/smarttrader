import pandas as pd

def membership_strategy_signals__these_are_the_best_trading_days_for_stocks_strategy(stock_df):
    signals = pd.DataFrame(index=stock_df.index)
    signals['Short_MA'] = stock_df['Close'].rolling(window=10).mean()
    signals['Long_MA'] = stock_df['Close'].rolling(window=30).mean()
    signals['Signal'] = 0
    signals['Signal'][10:] = np.where(signals['Short_MA'][10:] > signals['Long_MA'][10:], 1, 0)
    signals['Signal'][10:] = np.where(signals['Short_MA'][10:] < signals['Long_MA'][10:], -1, signals['Signal'][10:])
    signals['trade_signal'] = 'neutral'
    signals.loc[signals['Signal'] == 1, 'trade_signal'] = 'long'
    signals.loc[signals['Signal'] == -1, 'trade_signal'] = 'short'
    return signals[['trade_signal']]
