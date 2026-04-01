
import pandas as pd
from datetime import datetime

# Sell Rosh Hashanah - Buy Yom Kippur Trading Strategy
def sell_rosh_hashanah_buy_yom_kippur_signals(stock_df):
    if 'Date' in stock_df.columns:
        working_index = pd.to_datetime(stock_df['Date'])
    else:
        working_index = pd.to_datetime(stock_df.index)

    # Initialize signals DataFrame
    signals = pd.DataFrame(index=working_index)
    signals['signal'] = 'neutral'  # Default to neutral

    # Define the trading dates (for example, Rosh Hashanah and Yom Kippur in 2024)
    rosh_hashanah_date = datetime(2024, 9, 15)  # Example date
    yom_kippur_date = datetime(2024, 9, 24)     # Example date

    # Generate trading signals
    for date in working_index:
        if date == rosh_hashanah_date:
            signals.loc[date, 'signal'] = 'short'  # Sell at Rosh Hashanah
        elif date == yom_kippur_date:
            signals.loc[date, 'signal'] = 'long'   # Buy at Yom Kippur

    # Forward fill 'long' signals until the next short signal
    signals['signal'] = signals['signal'].replace('neutral', None)
    signals = signals.ffill()
    
    # Any remaining 'neutral' signals should stay as 'neutral'
    signals['signal'] = signals['signal'].fillna('neutral')

    return signals
