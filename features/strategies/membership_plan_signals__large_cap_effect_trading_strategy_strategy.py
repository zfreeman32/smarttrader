import pandas as pd

def membership_plan_signals__large_cap_effect_trading_strategy_strategy(stock_df):
    signals = pd.DataFrame(index=stock_df.index)
    signals['monthly_return'] = stock_df['Close'].pct_change()
    signals['signal'] = 'neutral'
    signals.loc[signals['monthly_return'] > 0.01, 'signal'] = 'long'
    signals.loc[signals['monthly_return'] < -0.01, 'signal'] = 'short'
    return signals[['signal']]
