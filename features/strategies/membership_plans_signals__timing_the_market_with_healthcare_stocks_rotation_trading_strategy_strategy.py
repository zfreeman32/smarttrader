import pandas as pd

def membership_plans_signals__timing_the_market_with_healthcare_stocks_rotation_trading_strategy_strategy(stock_df):
    signals = pd.DataFrame(index=stock_df.index)
    signals['signal'] = 'neutral'
    monthly_signal = stock_df['Close'].pct_change(periods=30)
    signals.loc[monthly_signal > 0.05, 'signal'] = 'long'
    signals.loc[monthly_signal < -0.05, 'signal'] = 'short'
    return signals
