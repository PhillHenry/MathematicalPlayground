from data.one_hot_encodings import make_fake_1hot_encodings
from sklearn import linear_model
from sklearn import linear_model

from data.one_hot_encodings import make_fake_1hot_encodings

if __name__ == "__main__":
    x = make_fake_1hot_encodings()

    ys = []
    for row in x:
        y = [a * (b + 1) for a, b in zip(row, range(len(row)))]
        ys.append(sum(y))

    regr = linear_model.LinearRegression()
    est = regr.fit(x, ys)

    # The coefficients
    print('Coefficients: \n', regr.coef_)
    print(est.summary())
