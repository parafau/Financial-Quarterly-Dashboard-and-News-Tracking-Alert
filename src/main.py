import yfinance as yf
import pandas as pd

# Force one row per date, all columns on a single line
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

CompanySymbol = input("Enter Company Symbol: ")

ticker = yf.Ticker(CompanySymbol)

q_income_stmt = ticker.quarterly_financials.T
q_income_stmt.index.name = "Date"
q_income_stmt = q_income_stmt.sort_index(ascending=False)

print("Quarterly Income Statement (transposed):")
print(q_income_stmt)