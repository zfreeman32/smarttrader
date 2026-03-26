import pandas as pd
import numpy as np

def market_sentiment_signals__stock_market_price_factors_strategy(stock_df, news_sentiment_df, economic_indicators_df):
    signals = pd.DataFrame(index=stock_df.index)
    signals['price_change'] = stock_df['Close'].pct_change()
    sentiment_score = news_sentiment_df['Sentiment'].rolling(window=5).mean()
    economic_indicator_score = economic_indicators_df['Indicator'].rolling(window=5).mean()
    signals['signal'] = 'neutral'
    signals.loc[(sentiment_score > 0) & (economic_indicator_score > 0), 'signal'] = 'long'
    signals.loc[(sentiment_score < 0) & (economic_indicator_score < 0), 'signal'] = 'short'
    return signals[['signal']]
