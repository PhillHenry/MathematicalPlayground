import numpy as np
from data.one_hot_encodings import make_fake_1hot_encodings
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score


if __name__ == "__main__":
    x = make_fake_1hot_encodings()

    ys = []
    for row in x:
        y = [a * (b + 1) for a, b in zip(row, range(len(row)))]
        ys.append(sum(y))

    print("y shape ", np.shape(x))
    print("y shape ", np.shape(y))

    regr = linear_model.LinearRegression()
    regr.fit(x, ys)

    # The coefficients
    print('Coefficients: \n', regr.coef_)
