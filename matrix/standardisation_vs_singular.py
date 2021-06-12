import numpy as np
from data.one_hot_encodings import make_fake_1hot_encodings


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
    m = make_fake_1hot_encodings(drop_last=True)
    invert_and_standardize(m)
    m = make_fake_1hot_encodings(drop_last=False)
    invert_and_standardize(m)
