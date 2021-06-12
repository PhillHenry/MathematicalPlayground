import numpy as np
import random


def make_fake_1hot_encodings(n_rows=1000,
                             n_categories=4,
                             n_cardinality=5,
                             drop_last=True) -> np.ndarray:
    if drop_last:
        drop_last = 0
        n_cardinality = n_cardinality - 1
    else:
        drop_last = 1
    n_cols = n_categories * n_cardinality
    m = np.zeros([n_rows, n_cols])
    for i in range(n_rows):
        for j in range(n_categories):
            index = int(random.random() * (n_cardinality + drop_last))
            if index < n_cardinality:
                m[i][(j * n_cardinality) + index] = 1.
    return m


def square(m):
    return np.dot(m, m.transpose())


def standardize(m: np.ndarray) -> np.ndarray:
    return (m - m.mean())/(m.std())


def invert_and_standardize(m):
    m_squared = square(m)

    print(f"det(m_squared) = {np.linalg.det(m_squared)}")

    m_standardized = standardize(m)
    m_standardized_squared = square(m_standardized)

    print(f"det(m_squared) = {np.linalg.det(m_standardized_squared)}")


if __name__ == "__main__":
    m = make_fake_1hot_encodings()
    invert_and_standardize(m)
    m = make_fake_1hot_encodings(drop_last=False)
    invert_and_standardize(m)
