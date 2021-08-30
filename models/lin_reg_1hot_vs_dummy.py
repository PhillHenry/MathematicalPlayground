from sklearn import linear_model
from sklearn.linear_model._coordinate_descent import Lasso
import numpy as np
from models.lin_reg_utils import train, check, check_row, test
from data.one_hot_encodings import make_fake_1hot_encodings, make_y, make_target, drop_last
import random


def train_and_check(x, ys, n_rows, model):
    n_train = int(n_rows * 0.8)
    m, coeffs, intercept = train(model, n_train, x, ys)
    return test(m, n_train, x, ys), coeffs


def deltas(xs):
    return [b - a for a, b in zip(xs, xs[1:])]


def skip_every(n, xs):
    return [x for i, x in enumerate(xs) if (i + 1) % n != 0]


def num_non_increasing(xs, skip):
    ds = deltas(xs)
    skipped = skip_every(skip, ds)
    return len([x for x in skipped if x < 0])


def compare_1hot_vs_dummy(error_to_model):
    n_rows = 1000
    n_categories = 4
    n_cardinality = 5
    m = make_fake_1hot_encodings(drop_last=False, n_rows=n_rows, n_categories=n_categories, n_cardinality=n_cardinality)
    m_dropped = drop_last(m, n_categories, n_cardinality)
    for noise in [0, 10, 100, 1000]:
        ys = make_y(m, error=noise)
        for intercept in [True, False]:
            model = error_to_model(noise, intercept)
            print("+" * 50)
            print("Intercept = %s, Noise level = %.4f," % (intercept, noise))
            print("One hot encoding")
            test_error, m_coeffs = train_and_check(m, ys, n_rows, model)
            print("Dummy variable encoding")
            m_dropped_error, m_dropped_coeffs = train_and_check(m_dropped, ys, n_rows, model)
            delta_error = test_error - m_dropped_error
            print("Difference in error %.4f" % delta_error)
            print(f"non increasing coefficients: {num_non_increasing(m_coeffs, n_cardinality)}")
            print(f"non increasing dropped coefficients: {num_non_increasing(m_dropped_coeffs, n_cardinality - 1)}")
        print("-" * 50)
        print("")


def error_to_lr(error, fit_intercept):
    return linear_model.LinearRegression(fit_intercept=fit_intercept)


def error_to_lasso(error, fit_intercept):
    return Lasso(alpha=(error/10) + 0.1)


if __name__ == "__main__":
    np.set_printoptions(precision=3)

    print("\n========== Linear Regression ===========\n")
    compare_1hot_vs_dummy(error_to_lr)

    print("\n========== Lasso ===========\n")
    compare_1hot_vs_dummy(error_to_lasso)

