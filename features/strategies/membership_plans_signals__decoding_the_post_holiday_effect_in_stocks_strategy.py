import pandas as pd

def membership_plans_signals__decoding_the_post_holiday_effect_in_stocks_strategy(stock_df):
    signals = pd.DataFrame(index=stock_df.index)
    signals['membership_signal'] = 'neutral'
    for i in range(1, len(stock_df)):
        month = stock_df.index[i].month
        if month in [7, 12]:
            continue
        if stock_df['Close'].iloc[i] > stock_df['Close'].iloc[i - 1]:
            signals['membership_signal'].iloc[i] = 'long'
        else:
            signals['membership_signal'].iloc[i] = 'short'
    return signals
