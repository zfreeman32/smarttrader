
import pandas as pd

# Best 6 Months Timing Model Strategy
def best_6_months_timing_signals(stock_df):
    if 'Date' in stock_df.columns:
        working_index = pd.DatetimeIndex(pd.to_datetime(stock_df['Date']))
    else:
        working_index = pd.DatetimeIndex(pd.to_datetime(stock_df.index))

    signals = pd.DataFrame(index=working_index)
    
    # Create month column for easier filtering
    signals['Month'] = working_index.month
    
    # Initialize the signal column
    signals['signal'] = 'neutral'
    
    # Define the trading periods for the Best 6 Months Timing Model (typically May to October)
    signals.loc[(signals['Month'] >= 5) & (signals['Month'] <= 10), 'signal'] = 'long'
    signals.loc[(signals['Month'] < 5) | (signals['Month'] > 10), 'signal'] = 'short'
    
    return signals[['signal']]
