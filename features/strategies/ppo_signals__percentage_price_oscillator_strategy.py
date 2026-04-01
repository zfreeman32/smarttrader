import pandas as pd
from ta import momentum

def ppo_signals__percentage_price_oscillator_strategy(stock_df, fast_length=12, slow_length=26, signal_length=9):
    signals = pd.DataFrame(index=stock_df.index)
    ppo_indicator = momentum.PercentagePriceOscillator(
        stock_df['Close'],
        window_slow=slow_length,
        window_fast=fast_length,
        window_sign=signal_length,
    )
    signals['PPO'] = ppo_indicator.ppo()
    signals['Signal_Line'] = ppo_indicator.ppo_signal()
    signals['ppo_signal'] = 'neutral'
    signals.loc[(signals['PPO'] > signals['Signal_Line']) & (signals['PPO'].shift(1) <= signals['Signal_Line'].shift(1)), 'ppo_signal'] = 'long'
    signals.loc[(signals['PPO'] < signals['Signal_Line']) & (signals['PPO'].shift(1) >= signals['Signal_Line'].shift(1)), 'ppo_signal'] = 'short'
    signals.drop(['PPO', 'Signal_Line'], axis=1, inplace=True)
    return signals
