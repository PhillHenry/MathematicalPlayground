import statsmodels.api as sm
from sklearn import linear_model
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from data.one_hot_encodings import make_fake_1hot_encodings
import random


def make_y(x):
    ys = []
    for row in x:
        y = [a * (b + 1) for b, a in enumerate(row)]
        ys.append(sum(y))
    return ys


def train(regr, n_train, x, ys):
    train_x = x[:n_train, :]
    train_y = ys[:n_train]
    est = regr.fit(train_x, train_y)
    print(f'{len(regr.coef_)} Coefficients: \n', ['%.3f' % x for x in regr.coef_])
    print("intercept", regr.intercept_)
    return regr, regr.coef_, regr.intercept_


def test(regr, n_train, x, ys):
    test_x = x[n_train:, :]
    test_y = ys[n_train:]
    y_pred = regr.predict(test_x)

    print('Mean squared error: %.2f'
          % mean_squared_error(test_y, y_pred))
    print('Coefficient of determination: %.2f'
          % r2_score(test_y, y_pred))


def check(x, ys, coeffs, intercept, index):
    random_row = x[index]
    random_value = ys[index]
    y = np.dot(coeffs, random_row) + intercept
    print(f"for example, {y} = {random_value}")


def no_drop_last(model, drop_last=True):
    n_rows = 1000
    x = make_fake_1hot_encodings(drop_last=drop_last, n_rows=n_rows)
    ys = make_y(x)
    n_train = int(n_rows * 0.8)
    model, coeffs, intercept = train(model, n_train, x, ys)
    test(model, n_train, x, ys)
    check(x, ys, coeffs, intercept, int(random.random() * n_rows))
    print()


def alpha_intercepts(m):
    for intercept in [True, False]:
        for drop in [True, False]:
            print(f"{m}, fit_interecpt = {intercept}, drop_last = {drop}")
            no_drop_last(m(alpha=.01, fit_intercept=intercept), drop_last=drop)


if __name__ == "__main__":
    np.set_printoptions(precision=3)
    for intercept in [True, False]:
        for drop in [True, False]:
            print(f"LinearRegression: fit_interecpt = {intercept}, drop_last = {drop}")
            no_drop_last(linear_model.LinearRegression(fit_intercept=intercept), drop_last=drop)
    alpha_intercepts(linear_model.Ridge)
    alpha_intercepts(linear_model.Lasso)
