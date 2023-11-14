import pandas as pd
import numpy as np
from scipy.optimize import minimize

df = pd.read_csv("testing_data.csv")
data_df = df[df.columns[1:]]
ret_list = []
optimized_weights_list = []

def mean_variance_optimization(returns_df):
    mean_returns = returns_df.mean()
    cov_matrix = returns_df.cov()
    num_assets = len(returns_df.columns)

    # minimize negative portfolio returns
    def negative_portfolio_returns(weights):
        return -np.sum(mean_returns * weights)

    weight_sum_constraint = {'type': 'eq', 'fun': 
                             lambda weights: np.sum(weights) - 1}

    bounds = tuple((0, 1) for _ in range(num_assets))

    initial_weights = np.ones(num_assets) / num_assets

    # Perform MVO optimization
    result = minimize(negative_portfolio_returns, initial_weights, 
                      method='SLSQP', bounds=bounds, 
                      constraints=[weight_sum_constraint])

    return result.x

for i in range(len(data_df.index)-1):
    returns_df = data_df.iloc[i:i+2]
    mean_ret = returns_df.mean()
    optimized_weights = mean_variance_optimization(returns_df)
    optimized_weights_list.append(optimized_weights)
    ret = np.dot(mean_ret,optimized_weights)
    ret_list.append(ret)

optimal_weights_df = pd.DataFrame(optimized_weights_list)
print(np.sum(ret_list))



