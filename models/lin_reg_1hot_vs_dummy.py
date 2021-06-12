from sklearn import linear_model
import numpy as np
from models.lin_reg_utils import train, check
from data.one_hot_encodings import make_fake_1hot_encodings, make_y
import random


def drop_last(m, n_categories, n_cardinality):
    cols_to_drop = [(i * n_cardinality) + n_cardinality - 1 for i in range(n_categories)]
    return np.delete(m, cols_to_drop, 1)


def compare_1hot_vs_dummy():
    n_rows = 1000
    n_categories = 4
    n_cardinality = 5
    m = make_fake_1hot_encodings(drop_last=False, n_rows=n_rows, n_categories=n_categories, n_cardinality=n_cardinality)
    ys = make_y(m)
    m_dropped = drop_last(m, n_categories, n_cardinality)
    n_train = int(n_rows * 0.8)

    def train_and_check(x, fit_intercept):
        print(f"\nIntercept = {fit_intercept}")
        model = linear_model.LinearRegression(fit_intercept=fit_intercept)
        m, coeffs, intercept = train(model, n_train, x, ys)
        check(x, ys, coeffs, intercept, int(random.random() * n_rows))

    for intercept in [True, False]:
        train_and_check(m, intercept)
        train_and_check(m_dropped, intercept)


if __name__ == "__main__":
    np.set_printoptions(precision=3)
    compare_1hot_vs_dummy()

