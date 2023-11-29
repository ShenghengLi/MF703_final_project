import pandas as pd
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import yfinance as yf
from scipy.optimize import minimize

tickers = ["AAPL", "MSFT", "AMZN", "GOOGL", "GOOG", "TSLA", "META", "NVDA", "PYPL", "NFLX",
    "ASML", "ADBE", "INTC", "CMCSA", "CSCO", "PEP", "AVGO", "TMUS", "COST",
    "TXN", "QCOM", "AMAT", "MU", "AMGN", "INTU", "ISRG", "CSX", "VRTX",
    "JD", "GILD", "BIDU", "MRVL", "REGN", "MDLZ", "ADSK", "ATVI", "BIIB", "ILMN",
    "LRCX", "ADP", "BKNG", "MELI", "KLAC", "NXPI", "MNST", "WDAY", "ROST",
    "KDP", "EA", "ALGN", "ADI", "IDXX", "DXCM", "XEL", "CTAS", "EXC", "MAR",
    "SNPS", "CDNS", "CPRT", "SGEN", "SPLK", "ORLY", "DLTR", "MTCH",
    "MCHP", "INCY", "PCAR", "CTSH", "FAST", "VRSK", "CHKP", "ANSS",
    "SWKS", "CDW", "TEAM", "WBA", "LULU", "PAYX",
    "VRSN", "AEP", "ZBRA", "TCOM", "NTES", "BMRN", "ULTA", "EXPE",
    "CSGP", "SIRI", "EBAY", "WDC"
    ]

real_data=pd.read_csv('real.csv', header=0)
real_data=real_data.set_index(['ticker', 'Date'])
real_adj_close=pd.DataFrame()
real_log_ret=pd.DataFrame()
for ticker in tickers:
    real_adj_close[ticker]=real_data.loc[ticker]['Adj Close']
    daily_log_return = np.log(real_adj_close[ticker] / real_adj_close[ticker].shift(1))
    real_log_ret[ticker] = daily_log_return
real_logret=real_log_ret.iloc[1:, :]

pred = pd.read_csv("pred.csv")
pred_logret = pred[pred.columns[1:]]

real_logret = real_logret#.iloc[:50,:10]
pred_logret = pred_logret#.iloc[:50,:10]

num_stocks = len(tickers)
value_invested = 1
alpha = 0.95
logret_list = []
optimized_weights_list = []

#MVO optimization
def mean_variance_optimization(curr_ret):
    mean_returns = curr_ret.mean()
    cov_matrix = curr_ret.cov()
    num_assets = len(curr_ret.columns)
    #print(num_assets)
    initial_weights = np.ones(num_assets) / num_assets
    # minimize negative portfolio returns
    def negative_portfolio_returns(weights):
        return -np.sum(mean_returns * weights)
    
    def portfolio_var(weights):
        #print(np.dot(weights.T, np.dot(cov_matrix, weights)))
        return np.dot(weights.T, np.dot(cov_matrix, weights))
    
    # min and max allocation of a single stock
    min_allocation = -0.25
    max_allocation = 0.25
    bounds = tuple((min_allocation, max_allocation) for _ in range(num_assets))

    # max variance 
    max_var = 0.20/252
    # constraints
    weight_sum_constraint = {'type': 'eq', 'fun': 
                             lambda weights: np.sum(weights) - 1}
    risk_constraint = {'type': 'ineq', 'fun': 
                       lambda weights: max_var - portfolio_var(weights)}
    # Perform MVO optimization
    result = minimize(negative_portfolio_returns, initial_weights, 
                      method='SLSQP', bounds=bounds, 
                      constraints=[weight_sum_constraint,risk_constraint])
    return result.x

def value_at_risk(value_invested, returns, weights, alpha=0.95, lookback_days=500):
    portfolio_returns = returns.iloc[-lookback_days:].dot(weights)
    return np.percentile(portfolio_returns, 100 * (1-alpha)) * value_invested

def cvar(value_invested, returns, weights, alpha=0.95, lookback_days=500):
    var = value_at_risk(value_invested, returns, weights, alpha, lookback_days=lookback_days)
    portfolio_returns = returns.iloc[-lookback_days:].dot(weights)
    var_pct_loss = var / value_invested
    return np.nanmean(portfolio_returns[portfolio_returns < var_pct_loss]) * value_invested

def roundNum(x):
    return round(x,4)

window_list = [90,120]
for i in window_list:
    for j in range(i,len(real_logret.index)):
        past_ret = real_logret.iloc[j-i:j-1]
        pred_ret = pred_logret.iloc[j-1:j]
        curr_ret = pd.concat([past_ret,pred_ret])
        
        optimized_weights = mean_variance_optimization(curr_ret)
        optimized_weights_list.append(optimized_weights)
        logret = np.dot(real_logret.iloc[j],optimized_weights)
        logret_list.append(logret)
        
        #print(pred_ret.index)
    #returns and annualized returns
    normal_ret_list = np.exp(logret_list)-1
    annualized_ret = np.mean(normal_ret_list)*252
    portfolio_ret = 1
    for ret in normal_ret_list:
        portfolio_ret *= (1+ret)
    #print(len(optimized_weights_list))
    
    # VaR and CVaR analysis
    portfolio_VaR = np.percentile(normal_ret_list, 100 * (1-alpha)) * value_invested
    portfolio_VaR_return = portfolio_VaR / value_invested
    
    portfolio_CVaR = np.nanmean(normal_ret_list[normal_ret_list 
                                                < portfolio_VaR_return]) * value_invested
    portfolio_CVaR_return = portfolio_CVaR / value_invested
    
    # grpah for VaR and CVaR analysis
    plt.hist(normal_ret_list)
    plt.axvline(portfolio_VaR_return, color='red', linestyle='solid')
    plt.axvline(portfolio_CVaR_return, color='red', linestyle='dashed')
    plt.legend(['VaR', 'CVaR', 'Returns'])
    plt.title(f'Historical VaR and CVaR for {i}-day Window')
    plt.xlabel('Return')
    plt.ylabel('Observation Frequency')
    plt.show()
    
    # print out report
    annualized_ret = roundNum(annualized_ret)
    portfolio_VaR_return = roundNum(portfolio_VaR_return)
    portfolio_CVaR_return = roundNum(portfolio_CVaR_return)
    print(f"{i}-day window returns:")
    print(f"  Annualized Return: {annualized_ret}")
    print(f"  VaR Return: {portfolio_VaR_return}")
    print(f"  CVaR Return: {portfolio_CVaR_return}")
    logret_list = []
    
optimal_weights_df = pd.DataFrame(optimized_weights_list)
#print(f"Portfolio VaR: {value_at_risk}")
