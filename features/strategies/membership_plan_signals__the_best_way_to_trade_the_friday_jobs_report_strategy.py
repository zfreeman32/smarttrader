import pandas as pd
from datetime import datetime

def membership_plan_signals__the_best_way_to_trade_the_friday_jobs_report_strategy(stock_df):
    signals = pd.DataFrame(index=stock_df.index)
    signals['membership_signal'] = 'neutral'
    current_date = datetime.now()
    if 1 <= current_date.day <= 10:
        signals['membership_signal'] = 'long'
    elif current_date.day > 10:
        signals['membership_signal'] = 'short'
    return signals
