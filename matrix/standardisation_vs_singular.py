import numpy as np

from data.one_hot_encodings import make_fake_1hot_encodings, make_y


def square(m):
    return np.dot(m, m.transpose())


def standardize(m: np.ndarray) -> np.ndarray:
    return (m - m.mean()) / (m.std())


def invert_and_standardize(m):
    '''
    Formula for calculating the co-efficients in a linear regression is:
    β=(X′X)−1X′Y
    '''
    print(f"det((m'm)^-1)                            = {np.linalg.det(np.linalg.inv(np.dot(m.transpose(), m)))}")
    print(f"det(m_squared)                           = {np.linalg.det(square(m))}")
    msg = "det((mm')^-1)                            = "
    try:
        det_invert = np.linalg.det(np.linalg.inv(square(m)))
        print(msg, det_invert)  # blows up because mm' is singular
    except Exception as e:
        print(f"{msg}Could not find det((mm')^-1. Error = {e}")
    print(f"det(standardized(m) * standardized(m).T) = {np.linalg.det(square(standardize(m)))}")
    print(f"det(standardized(m m.T)                  = {np.linalg.det(standardize(square(m)))}")


def add_value_col(m):
    ys = make_y(m, error=False)
    ys = np.transpose(np.asmatrix(ys))
    return np.hstack((m, ys))


def det_of_square_matrix(drop_last: bool):
    print(f"\nSquare one hot encoding when dropping the last element is {drop_last}")
    n_rows = 100
    n_categories = 20
    if drop_last:
        n_rows = n_rows - n_categories
    square_matrix = make_fake_1hot_encodings(drop_last=drop_last,
                                             n_rows=n_rows,
                                             n_categories=n_categories,
                                             n_cardinality=5)
    print(f"{np.linalg.det(square_matrix)} for matrix of shape {np.shape(square_matrix)}")


if __name__ == "__main__":
    print("\ndrop_last=True")
    m = make_fake_1hot_encodings(drop_last=True)
    invert_and_standardize(m)

    print("\ndrop_last=False")
    m = make_fake_1hot_encodings(drop_last=False)
    invert_and_standardize(m)

    print("\ndrop_last=False, add column exactly correlated to others")
    m = add_value_col(make_fake_1hot_encodings(drop_last=False, n_rows=1001))
    invert_and_standardize(m)

    n_categories = 4
    n_cardinality = 5
    m = add_value_col(make_fake_1hot_encodings(drop_last=False,
                                               n_categories=n_categories,
                                               n_cardinality=n_cardinality,
                                               n_rows=((n_categories * n_cardinality) + 1)))
    print(f"\ncoincidentally square matrix {np.shape(m)} det = {np.linalg.det(square(m))}")
    m = add_value_col(make_fake_1hot_encodings(drop_last=True,
                                               n_categories=n_categories,
                                               n_cardinality=n_cardinality,
                                               n_rows=((n_categories * (n_cardinality - 1)) + 1)))
    print(f"\ncoincidentally square matrix {np.shape(m)} det = {np.linalg.det(square(m))}")

    det_of_square_matrix(False)
    det_of_square_matrix(True)
    print("\nrandom matrix")
    print(np.linalg.det(np.random.rand(100, 100)))
