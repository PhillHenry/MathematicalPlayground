import random

import numpy as np
from sklearn.base import RegressorMixin
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression

from data.one_hot_encodings import make_fake_1hot_encodings, make_y


def train(regr: RegressorMixin, n_train, x, ys):
    train_x = x[:n_train, :]
    train_y = ys[:n_train]
    est = regr.fit(train_x, train_y)
    # print(f'{len(regr.coef_)} Coefficients: \n', ['%.3f' % x for x in regr.coef_])
    return regr, regr.coef_, regr.intercept_


def test(regr, n_train, x, ys):
    test_x = x[n_train:, :]
    test_y = ys[n_train:]
    y_pred = regr.predict(test_x)

    error = mean_squared_error(test_y, y_pred)

    return error, r2_score(test_y, y_pred)


def check(x, ys, coeffs, intercept, index):
    random_row = x[index]
    random_value = ys[index]
    check_row(coeffs, intercept, random_row, random_value)


def check_row(coeffs, intercept, row, value):
    y = np.dot(coeffs, row) + intercept
    print(f"for example, {y} = {value}")


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

