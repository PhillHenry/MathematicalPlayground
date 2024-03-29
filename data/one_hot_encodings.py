import numpy as np
import random


def make_fake_1hot_encodings(n_rows=1000,
                             n_categories=4,
                             n_cardinality=5,
                             drop_last=True) -> np.ndarray:
    if drop_last:
        correction = 1
    else:
        correction = 0
    n_cols = n_categories * (n_cardinality - correction)
    m = np.zeros([n_rows, n_cols])
    for i in range(n_rows):
        for j in range(n_categories):
            index = int(random.random() * n_cardinality)
            if index < (n_cardinality - correction):
                m[i][(j * (n_cardinality - correction)) + index] = 1.
    return m


def make_y(x, intercept=42, error=0):
    ys = []
    for row in x:
        y = make_target(row, intercept, error)
        ys.append(y)
    return ys


def make_target(row, intercept=42, error=0):
    return sum([a * (b + 1) for b, a in enumerate(row)]) + intercept + (error * random.gauss(0, 1))


def drop_last(m, n_categories, n_cardinality):
    cols_to_drop = [(i * n_cardinality) + n_cardinality - 1 for i in range(n_categories)]
    return np.delete(m, cols_to_drop, 1)

