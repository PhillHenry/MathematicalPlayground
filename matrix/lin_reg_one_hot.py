import statsmodels.api as sm
from sklearn import linear_model
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from data.one_hot_encodings import make_fake_1hot_encodings


def make_y(x):
    ys = []
    for row in x:
        y = [a * (b + 1) for b, a in enumerate(row)]
        ys.append(sum(y))
    return ys


def no_drop_last(regr, drop_last=True):
    n_rows = 1000
    x = make_fake_1hot_encodings(drop_last=drop_last, n_rows=n_rows)
    ys = make_y(x)
    n_train = int(n_rows * 0.8)
    train_x = x[:n_train, :]
    train_y = ys[:n_train]
    est = regr.fit(train_x, train_y)
    test_x = x[n_train:, :]
    test_y = ys[n_train:]
    y_pred = regr.predict(test_x)
    print(f'{len(regr.coef_)} Coefficients: \n', ['%.3f' % x for x in regr.coef_])
    print("intercept", regr.intercept_)
    print('Mean squared error: %.2f'
          % mean_squared_error(test_y, y_pred))
    print('Coefficient of determination: %.2f'
          % r2_score(test_y, y_pred))


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
