import pandas as pd

def end_of_month_effect_signals__end_of_month_effect_in_nifty_50_does_it_exist_turn_of_the_month_strategy(stock_df):
    signals = pd.DataFrame(index=stock_df.index)
    signals['position'] = 'neutral'
    stock_df['Date'] = pd.to_datetime(stock_df.index)
    stock_df['TradingDay'] = stock_df['Date'].dt.day
    last_trading_day = stock_df['TradingDay'].max()
    entry_days = list(range(last_trading_day - 9, last_trading_day + 1))
    for day in entry_days:
        entry_condition = stock_df['TradingDay'] == day
        exit_condition = stock_df['TradingDay'] == 2
        signals.loc[entry_condition, 'position'] = 'long'
        signals.loc[exit_condition, 'position'] = 'neutral'
    signals['position'] = signals['position'].shift(1)
    signals.dropna(inplace=True)
    return signals
