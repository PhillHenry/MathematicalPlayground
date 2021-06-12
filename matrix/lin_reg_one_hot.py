import statsmodels.api as sm
from sklearn import linear_model

from data.one_hot_encodings import make_fake_1hot_encodings


def make_y(x):
    ys = []
    for row in x:
        y = [a * (b + 1) for a, b in zip(row, range(len(row)))]
        ys.append(sum(y))
    return ys


def no_drop_last():
    x = make_fake_1hot_encodings(drop_last=True)
    ys = make_y(x)
    regr = linear_model.LinearRegression()
    est = regr.fit(x, ys)
    print(f'{len(regr.coef_)} Coefficients: \n', regr.coef_)


if __name__ == "__main__":
    no_drop_last()
    # X2 = sm.add_constant(x)
    # est = sm.OLS(y, X2)
    # est2 = est.fit()
    # print(est2.summary())
