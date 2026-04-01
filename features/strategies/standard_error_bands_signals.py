
import pandas as pd
import numpy as np

# Standard Error Bands Strategy
def standard_error_bands_signals(stock_df, window_regression=21, window_sma=3, num_std_errors=2):
    signals = pd.DataFrame(index=stock_df.index)
    x = np.arange(window_regression, dtype=float)

    def _regression_endpoint(prices: np.ndarray) -> float:
        slope, intercept = np.polyfit(x, prices, 1)
        fitted = slope * x + intercept
        return float(fitted[-1])

    def _std_error(prices: np.ndarray) -> float:
        slope, intercept = np.polyfit(x, prices, 1)
        fitted = slope * x + intercept
        return float(np.sqrt(np.mean((prices - fitted) ** 2)))

    regression_line = stock_df['Close'].rolling(window=window_regression).apply(_regression_endpoint, raw=True)
    residuals = stock_df['Close'].rolling(window=window_regression).apply(_std_error, raw=True)

    # Calculate the middle line (3-period SMA of the regression endpoint)
    signals['Middle_Line'] = regression_line.rolling(window=window_sma).mean()
    
    # Calculate the upper and lower bands
    signals['Upper_Band'] = signals['Middle_Line'] + num_std_errors * residuals.rolling(window=window_sma).mean()
    signals['Lower_Band'] = signals['Middle_Line'] - num_std_errors * residuals.rolling(window=window_sma).mean()

    # Generate signals
    signals['Signal'] = 'neutral'
    signals.loc[(stock_df['Close'] > signals['Upper_Band']), 'Signal'] = 'short'
    signals.loc[(stock_df['Close'] < signals['Lower_Band']), 'Signal'] = 'long'
    
    return signals[['Signal']]
