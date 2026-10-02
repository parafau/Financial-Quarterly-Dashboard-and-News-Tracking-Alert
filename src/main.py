import yfinance as yf
import pandas as pd

# Curated Nifty 100 symbol list as a static fallback (as of Jan 2026)
NIFTY100_STATIC = [
    "ADANIENT","ADANIGREEN","ADANIPORTS","ADANIPOWER","AMBUJACEM",
    "APOLLOHOSP","ASIANPAINT","AXISBANK","BAJAJ-AUTO","BAJAJFINSV",
    "BAJFINANCE","BANKBARODA","BEL","BHARTIB","BPCL","BRIT","BRITANNIA",
    "CANBK","CHOLAFIN","CIPLA","COALINDIA","DMART","DLF","DRREDDY",
    "EICHERMOT","FEDERALBNK","GAIL","GODS","GRASIM","HCLTECH","HDFCBANK",
    "HDFCLIFE","HEROMOTOCO","HINDALCO","HINDUNILVR","ICICIBANK",
    "ICICIGI","ICICIPRULI","INDHOTEL","INDIGO","INDUSINDBK","INFRATEL",
    "IOC","IRCTC","ITC","JSWSTEEL","JUBLFOOD","KOTAKBANK","LT","LICI",
    "LTIM","LUPIN","M&M","MARICO","MARUTI","NESTLEIND","NTPC","ONGC",
    "PAGEIND","PFC","PIDILITIND","PIIND","POLYCAB","POWERGRID","RECLTD",
    "RELIANCE","SBICARD","SBILIFE","SBIN","SHREECEM","SIEMENS",
    "SUNPHARMA","SUNTV","TATACONSUM","TATAMOTORS","TATAPOWER","TATASTEEL",
    "TCS","TECHM","TITAN","TORNTPHARM","TRENT","ULTRACEMCO","UNIONBANK",
    "UPL","VEDL","WIPRO","ZOMATO","ZYDUSLIFE",
]

CompanySymbol = input("Enter Company Symbol: ")
# CompanySymbol = "tcs" #used for trial

#converts lowercase input to uppercase
CompanySymbol=CompanySymbol.upper()

#appending .ns for Indian stocks as input
if CompanySymbol in NIFTY100_STATIC: CompanySymbol+=".ns"

#creating input company symbol as object 
ticker = yf.Ticker(CompanySymbol)
q_income_stmt = ticker.quarterly_financials

#Display the top all from the most recent 4 quarters
print("Quarterly Income Statements: ")
print(q_income_stmt)



""" with reference to Screener, following parameters are worked with about a stock [eg: TCS]:
Sales + 	        59,381 	59,692 	60,583 	61,237 	62,613 	64,259 	63,973 	64,479 	63,437 	65,799 	67,087 	70,698 	72,275
Expenses + 	        44,383 	43,946 	44,195 	44,073 	45,951 	47,528 	46,939 	47,499 	46,562 	47,821 	48,818 	51,422 	53,719
Operating Profit 	14,998 	15,746 	16,388 	17,164 	16,662 	16,731 	17,034 	16,980 	16,875 	17,978 	18,269 	19,276 	18,556
OPM % 	            25% 	26% 	27% 	28% 	27% 	26% 	27% 	26% 	27% 	27% 	27% 	27% 	26%
Other Income + 	    1,397 	1,006 	-96 	1,157 	962 	729 	1,243 	1,028 	1,660 	-268 	-2,273 	757 	900
Interest 	        163 	159 	230 	226 	173 	162 	234 	227 	195 	229 	538 	265 	273
Depreciation 	    1,243 	1,263 	1,233 	1,246 	1,220 	1,266 	1,377 	1,379 	1,361 	1,413 	1,380 	1,406 	1,239
Profit before tax 	14,989 	15,330 	14,829 	16,849 	16,231 	16,032 	16,666 	16,402 	16,979 	16,068 	14,078 	18,362 	17,944
Tax % 	            26% 	26% 	25% 	26% 	25% 	25% 	25% 	25% 	24% 	24% 	24% 	25% 	25%
Net Profit +    	11,120 	11,380 	11,097 	12,502 	12,105 	11,955 	12,444 	12,293 	12,819 	12,131 	10,720 	13,784 	13,420
EPS in Rs           30.26 	31.00 	30.56 	34.37 	33.28 	32.92 	34.22 	33.79 	35.27 	33.37 	29.45 	37.92 	36.90
"""

ticker_parameters=[
"Market_Cap",
"Current_Price",
"High_Low",
"Stock_P/E",
"Book_Value",
"Dividend_Yield",
"ROCE",
"ROE",
"Value",
]
