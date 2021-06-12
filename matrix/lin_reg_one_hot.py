import statsmodels.api as sm
from sklearn import linear_model
import numpy as np

from data.one_hot_encodings import make_fake_1hot_encodings


def make_y(x):
    ys = []
    for row in x:
        y = [a * (b + 1) for a, b in zip(row, range(len(row)))]
        ys.append(sum(y))
    return ys


def no_drop_last(regr, drop_last=True):
    x = make_fake_1hot_encodings(drop_last=drop_last)
    ys = make_y(x)
    est = regr.fit(x, ys)
    print(f'{len(regr.coef_)} Coefficients: \n', regr.coef_)
    print("intercept", regr.intercept_)


def alpha_intercepts(m):
    for intercept in [True, False]:
        for drop in [True, False]:
            print(f"\n{m}, fit_interecpt = {intercept}, drop_last = {drop}")
            no_drop_last(m(alpha=.01, fit_intercept=intercept), drop_last=drop)


if __name__ == "__main__":
    np.set_printoptions(precision=3)
    no_drop_last(linear_model.LinearRegression(fit_intercept=True))
    no_drop_last(linear_model.LinearRegression(fit_intercept=False))
    alpha_intercepts(linear_model.Ridge)
    alpha_intercepts(linear_model.Lasso)
