import yfinance as yf
import pandas as pd
import numpy as np

nasdaq_100_tickers = ["AAPL", "MSFT", "AMZN", "GOOGL", "GOOG", "TSLA", "META", "NVDA", "PYPL", "NFLX",
    "ASML", "ADBE", "INTC", "CMCSA", "CSCO", "PEP", "AVGO", "TMUS", "COST", "PDD",
    "TXN", "QCOM", "AMAT", "MU", "AMGN", "INTU", "ISRG", "ZM", "CSX", "VRTX",
    "JD", "GILD", "BIDU", "MRVL", "REGN", "MDLZ", "ADSK", "ATVI", "BIIB", "ILMN",
    "LRCX", "ADP", "BKNG", "MELI", "KLAC", "DOCU", "NXPI", "MNST", "WDAY", "ROST",
    "KDP", "EA", "ALGN", "ADI", "IDXX", "DXCM", "XEL", "CTAS", "EXC", "MAR",
    "SNPS", "ASAN", "CDNS", "CPRT", "SGEN", "SPLK", "ORLY", "DLTR", "MTCH",
    "MCHP", "INCY", "PCAR", "CTSH", "FAST", "VRSK", "CHKP", "FOXA", "FOX", "ANSS",
    "SWKS", "OKTA", "TTD", "CDW", "TEAM", "WBA", "LULU", "PAYX",
    "VRSN", "AEP", "ZBRA", "PTON", "TCOM", "NTES", "BMRN", "ULTA", "EXPE",
    "CSGP", "SIRI", "EBAY", "WDC"
    ]
nasdaq_20_tickers = ["AAPL", "MSFT", "AMZN", "GOOGL", "GOOG", "TSLA", "META", "NVDA", "PYPL", "NFLX",
    "ASML", "ADBE", "INTC", "CMCSA", "CSCO", "PEP", "AVGO", "TMUS", "COST", "PDD"]

end_date = '2022-12-31'
start_date = '2018-1-1'

data = {}
daily_log_ret = {}
for ticker in nasdaq_100_tickers:
    stock_data = yf.download(ticker, start=start_date, end=end_date, progress=False)
    data[ticker] = stock_data['Adj Close']
    data[ticker].index = data[ticker].index.strftime('%Y/%m/%d')
    daily_log_return = np.log(data[ticker] / data[ticker].shift(1))
    daily_log_ret[ticker] = daily_log_return

nasdaq_df = pd.concat(data.values(), axis=1, keys=data.keys()).dropna()
daily_log_ret_df = pd.concat(daily_log_ret.values(), axis=1, keys=daily_log_ret.keys()).dropna()
