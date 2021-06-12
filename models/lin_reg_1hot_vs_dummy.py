from sklearn import linear_model
import numpy as np
from models.lin_reg_utils import train, check, check_row, test
from data.one_hot_encodings import make_fake_1hot_encodings, make_y, make_target, drop_last
import random


def train_and_check(x, ys, n_rows, fit_intercept):
    n_train = int(n_rows * 0.8)
    print(f"\nIntercept = {fit_intercept}")
    model = linear_model.LinearRegression(fit_intercept=fit_intercept)
    m, coeffs, intercept = train(model, n_train, x, ys)
    check(x, ys, coeffs, intercept, int(random.random() * n_rows))
    zero_vector = np.zeros([np.shape(x)[1]])
    check_row(coeffs, intercept, zero_vector, make_target(zero_vector))
    test(m, n_train, x, ys)


def compare_1hot_vs_dummy():
    n_rows = 1000
    n_categories = 4
    n_cardinality = 5
    m = make_fake_1hot_encodings(drop_last=False, n_rows=n_rows, n_categories=n_categories, n_cardinality=n_cardinality)
    ys = make_y(m)
    m_dropped = drop_last(m, n_categories, n_cardinality)
    for intercept in [True, False]:
        train_and_check(m, ys, n_rows, intercept)
        train_and_check(m_dropped, ys, n_rows, intercept)


if __name__ == "__main__":
    np.set_printoptions(precision=3)
    compare_1hot_vs_dummy()

