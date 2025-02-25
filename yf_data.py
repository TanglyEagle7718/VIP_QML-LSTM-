import yfinance as yf
import pandas as pd
import numpy as np


ticker = "SPY"
end_date = "2025-11-01"

spy = yf.Ticker(ticker)
df = spy.history(period="max", end=end_date)


df.sort_index(inplace=True)




# A. Daily Return & Log Return
df['Daily_Return'] = df['Close'].pct_change()
df['Log_Return'] = np.log(df['Close']).diff()

# B. Moving Averages
df['SMA_20'] = df['Close'].rolling(window=20).mean()
df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()

# C. Bollinger Bands (20-day)
rolling_std_20 = df['Close'].rolling(window=20).std()
df['BB_Middle'] = df['SMA_20']
df['BB_Upper'] = df['SMA_20'] + (2 * rolling_std_20)
df['BB_Lower'] = df['SMA_20'] - (2 * rolling_std_20)

# D. Relative Strength Index (RSI) - 14 day
window_length = 14
delta = df['Close'].diff()
gain = delta.where(delta > 0, 0)
loss = -delta.where(delta < 0, 0)
avg_gain = gain.ewm(com=(window_length - 1), min_periods=window_length).mean()
avg_loss = loss.ewm(com=(window_length - 1), min_periods=window_length).mean()
rs = avg_gain / avg_loss
df['RSI_14'] = 100 - (100 / (1 + rs))

# E. MACD (12, 26, 9)
df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
df['MACD'] = df['EMA_12'] - df['EMA_26']
df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']

# F. Stochastic Oscillator (14)
lowest_low_14 = df['Low'].rolling(window=14).min()
highest_high_14 = df['High'].rolling(window=14).max()
df['Stoch_%K'] = 100 * ((df['Close'] - lowest_low_14) / (highest_high_14 - lowest_low_14))
df['Stoch_%D'] = df['Stoch_%K'].rolling(window=3).mean()

# G. Average True Range (ATR) - 14 day
high_low = df['High'] - df['Low']
high_close_prev = (df['High'] - df['Close'].shift()).abs()
low_close_prev = (df['Low'] - df['Close'].shift()).abs()


df['TR'] = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
df['ATR_14'] = df['TR'].rolling(window=14).mean()

# H. On-Balance Volume (OBV)
df['OBV'] = 0
df['OBV'] = np.where(
    df['Close'] > df['Close'].shift(1),
    df['Volume'],
    np.where(df['Close'] < df['Close'].shift(1), -df['Volume'], 0)
).cumsum()


df.dropna(inplace=True)

output_file = "spy_data.csv"
df.to_csv(output_file)
print(f"Data with expanded technical indicators saved to {output_file}")
