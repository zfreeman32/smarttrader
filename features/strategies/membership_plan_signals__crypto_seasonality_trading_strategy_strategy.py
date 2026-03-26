import pandas as pd

def membership_plan_signals__crypto_seasonality_trading_strategy_strategy(stock_df):
    signals = pd.DataFrame(index=stock_df.index)
    signals['Moving_Average'] = stock_df['Close'].rolling(window=20).mean()
    signals['membership_signal'] = 'neutral'
    signals.loc[stock_df['Close'] > signals['Moving_Average'], 'membership_signal'] = 'long'
    signals.loc[stock_df['Close'] < signals['Moving_Average'], 'membership_signal'] = 'short'
    return signals[['membership_signal']]
