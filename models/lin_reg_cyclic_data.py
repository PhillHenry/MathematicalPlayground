import numpy as np
import statsmodels.api as sm
from sklearn import linear_model

if __name__ == '__main__':
    n_cycles = 10
    train_y = np.linspace(0, n_cycles * 2 * np.pi, n_cycles * 100)
    sins = np.sin(train_y)
    coss = np.cos(train_y)
    train_x = np.asarray(list(zip(sins, coss)))
    model = linear_model.LinearRegression(fit_intercept=True)
    est = model.fit(train_x, train_y)
    print(f"{est.coef_}")
    # return regr, regr.coef_, regr.intercept_
    xs = sm.add_constant(train_x)
    est = sm.OLS(train_y, xs)
    est2 = est.fit()
    print(f"p_values = {est2.pvalues}")
    print(f"coeffs   = {est2.params}")
