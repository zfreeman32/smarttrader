
import pandas as pd
import numpy as np

# Linear Regression Indicator Strategy
def linear_regression_signals(stock_df, window=14):
    signals = pd.DataFrame(index=stock_df.index)
    x = np.arange(window, dtype=float)

    def _regression_endpoint(prices: np.ndarray) -> float:
        slope, intercept = np.polyfit(x, prices, 1)
        return float((slope * x[-1]) + intercept)

    signals['Linear_Regression'] = stock_df['Close'].rolling(window=window).apply(
        _regression_endpoint,
        raw=True,
    )
    
    signals['signal'] = 'neutral'
    signals.loc[(stock_df['Close'] > signals['Linear_Regression']), 'signal'] = 'long'
    signals.loc[(stock_df['Close'] < signals['Linear_Regression']), 'signal'] = 'short'
    
    return signals[['signal']]
