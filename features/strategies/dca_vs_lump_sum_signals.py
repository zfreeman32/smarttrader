
import pandas as pd

# Dollar Cost Averaging vs Lump Sum Strategy
def dca_vs_lump_sum_signals(stock_df, dca_investment=250, lump_sum_investment=3000, investment_duration_years=10, months_per_year=12):
    signals = pd.DataFrame(index=stock_df.index)
    close = stock_df['Close'].astype(float)
    
    # Assume we invest lump-sum at the start of the period
    lump_sum_result = lump_sum_investment * (close / close.iloc[0])
    
    # Implement Dollar Cost Averaging for each month
    total_investments = investment_duration_years * months_per_year
    purchase_prices = close.resample('ME').first().iloc[:total_investments]
    shares_bought = dca_investment / purchase_prices
    cumulative_shares = shares_bought.cumsum()
    aligned_shares = cumulative_shares.reindex(signals.index, method='ffill').fillna(0.0)
    dca_cumulative = aligned_shares * close
    
    signals['lump_sum_value'] = lump_sum_result
    signals['dca_value'] = dca_cumulative
    
    # Generate signals based on performance comparison
    signals['signal'] = 'neutral'
    signals.loc[signals['dca_value'] > signals['lump_sum_value'], 'signal'] = 'dca_superior'
    signals.loc[signals['dca_value'] < signals['lump_sum_value'], 'signal'] = 'lump_sum_superior'
    
    return signals
