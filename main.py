import yfinance as yf

CompanySymbol= input("Enter Company Symbol: ")

#Initialize the ticker for your target company
ticker = yf.Ticker(CompanySymbol)

#Fetch the quarterly income statement
q_income_stmt = ticker.quarterly_financials

#Display the top all from the most recent 4 quarters
print("Quarterly Income Statement (Top 5 rows):")
print(q_income_stmt.iloc[:, :4])
