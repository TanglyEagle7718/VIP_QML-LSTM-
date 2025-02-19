import yfinance as yf
import pandas as pd

ticker = "SPY"
end_date = "2025-11-01"

spy = yf.Ticker(ticker)
df = spy.history(period="max", end=end_date)

output_file = "spy_data.csv"
df.to_csv(output_file)