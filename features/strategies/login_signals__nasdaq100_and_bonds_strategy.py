import pandas as pd

def login_signals__nasdaq100_and_bonds_strategy(user_df):
    signals = pd.DataFrame(index=user_df.index)
    signals['login_attempt'] = 'not attempted'
    signals.loc[user_df['username'].notnull() & user_df['password'].notnull(), 'login_attempt'] = 'attempted'
    signals.loc[user_df['password'] == 'correct_password', 'login_attempt'] = 'success'
    signals.loc[user_df['password'] != 'correct_password', 'login_attempt'] = 'failure'
    return signals
