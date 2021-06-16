import numpy as np

from data.one_hot_encodings import make_fake_1hot_encodings


def square(m):
    return np.dot(m, m.transpose())


def standardize(m: np.ndarray) -> np.ndarray:
    return (m - m.mean()) / (m.std())


def invert_and_standardize(m):
    print(f"det(m_squared) = {np.linalg.det(square(m))}")

    print(f"det(standardized(m) * standardized(m).T) = {np.linalg.det(square(standardize(m)))}")

    print(f"det(standardized(m m.T) = {np.linalg.det(standardize(square(m)))}")


if __name__ == "__main__":
    print("\ndrop_last=True")
    m = make_fake_1hot_encodings(drop_last=True)
    invert_and_standardize(m)
    print("\ndrop_last=False")
    m = make_fake_1hot_encodings(drop_last=False)
    invert_and_standardize(m)
    print("\nSquare one hot encoding")
    square_matrix = make_fake_1hot_encodings(drop_last=False, n_rows=100, n_categories=20,
                                             n_cardinality=5)
    print(np.linalg.det(square_matrix))
    print("\nrandom matrix")
    print(np.linalg.det(np.random.rand(100, 100)))
