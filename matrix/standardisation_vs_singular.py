import numpy as np
import statsmodels.api as sm
import pandas as pd
from data.one_hot_encodings import make_fake_1hot_encodings, make_y


def mmT(m):
    return np.dot(m, m.transpose())


def standardize(m: np.ndarray) -> np.ndarray:
    return (m - m.mean()) / (m.std())


def invert_and_standardize(m):
    '''
    Formula for calculating the co-efficients in a linear regression is:
    β=(X′X)−1X′Y
    '''
    print(f"det(m'm)                                 = {np.linalg.det(mTm(m))}")
    # print(f"det((m'm)^-1)                            = {np.linalg.det(np.linalg.inv(mTm(m)))}")
    print(f"det(mm')                                 = {np.linalg.det(mmT(m))}")
    msg = "det((mm')^-1)                            = "
    # try:
    #     det_invert = np.linalg.det(np.linalg.inv(mmT(m)))
    #     print(msg, det_invert)  # blows up because mm' is singular
    # except Exception as e:
    #     print(f"{msg}Could not find det((mm')^-1. Error = {e}")
    print(f"det(standardized(m) * standardized(m).T) = {np.linalg.det(mmT(standardize(m)))}")
    print(f"det(standardized(m m.T)                  = {np.linalg.det(standardize(mmT(m)))}")
    print(f"det(standardized(m).T * standardized(m)) = {np.linalg.det(mTm(standardize(m)))}")
    print(f"det(standardized(m.T m)                  = {np.linalg.det(standardize(mTm(m)))}")


def mTm(m):
    return np.dot(m.transpose(), m)


def add_value_col(m):
    ys = make_y(m, error=False)
    ys = np.transpose(np.asmatrix(ys))
    return np.hstack((m, ys))


def det_of_square_1hot_matrix(drop_last: bool):
    print(f"\nSquare one hot encoding when dropping the last element is {drop_last}")
    n_rows = 100
    n_categories = 20
    if drop_last:
        n_rows = n_rows - n_categories
    square_matrix = make_fake_1hot_encodings(drop_last=drop_last,
                                             n_rows=n_rows,
                                             n_categories=n_categories,
                                             n_cardinality=5)
    shape = np.shape(square_matrix)
    print(f"determinant(m)   = {np.linalg.det(square_matrix)} for matrix of shape {shape}")
    print(f"determinant(m'm) = {np.linalg.det(mTm(square_matrix))} for matrix of shape {shape}")


def invert_and_standardized_1hot(drop_last):
    print(f"\ndrop_last={drop_last}")
    m = make_fake_1hot_encodings(n_rows=100,
                                 n_categories=3,
                                 n_cardinality=4,
                                 drop_last=drop_last)
    invert_and_standardize(m)


def examine_with_correlated_column(drop_last):
    print(f"\ndrop_last={drop_last}, add column exactly correlated to others")
    m = add_value_col(make_fake_1hot_encodings(drop_last=drop_last, n_rows=1001))
    invert_and_standardize(m)


def coincidentally_square_1hot(drop_last: bool,
                               n_rows: int,
                               n_cardinality: int,
                               n_categories: int):
    m = add_value_col(make_fake_1hot_encodings(drop_last=drop_last,
                                               n_categories=n_categories,
                                               n_cardinality=n_cardinality,
                                               n_rows=n_rows))
    shape = np.shape(m)
    assert shape[0] == shape[1]
    print(f"\ncoincidentally square 1-hot matrix {shape} with drop_last={drop_last}")
    print(f"det(m'm) = {np.linalg.det(mTm(m))}")


def random_matrix():
    print("\nrandom matrix")
    m = np.random.rand(100, 100)
    print(f"det(m)   = {np.linalg.det(m)}")
    print(f"det(m'm) = {np.linalg.det(mTm(m))}")


def fit_model(pandas_inputs, outcomes: np.ndarray):
    inputs = sm.add_constant(pandas_inputs)
    model = sm.GLM(outcomes, inputs, family=sm.families.Binomial())
    results = model.fit()  #method='bfgs')
    return results


def log_regression(drop_last):
    print(f"\ndrop_last={drop_last}")
    m = make_fake_1hot_encodings(n_rows=100,
                                 n_categories=3,
                                 n_cardinality=4,
                                 drop_last=drop_last)
    inputs = numpy_to_pandas(m)
    ys = make_y(m, error=False)
    median = np.median(ys)

    def f(x):
        if x > median:
            return 1
        else:
            return 0

    outcomes = []
    for y in ys:
        outcomes.append([f(y)])
    outcomes = np.asarray(outcomes)
    print(np.shape(outcomes))
    # outcomes = np.array(map(f, outcomes))
    # print(np.shape(outcomes))
    outcomes = numpy_to_pandas(outcomes)
    fit_model(inputs, outcomes)


def numpy_to_pandas(m):
    shape = np.shape(m)
    if len(shape) == 2:
        n_rows, n_columns = shape
    else:
        n_rows = shape[0]
        n_columns = 0
    inputs = pd.DataFrame(data=m,
                          index=np.array(range(1, n_rows + 1)),
                          columns=np.array(range(1, n_columns + 1)))
    return inputs


if __name__ == "__main__":
    log_regression(True)
    invert_and_standardized_1hot(drop_last=True)
    invert_and_standardized_1hot(drop_last=False)

    examine_with_correlated_column(drop_last=True)
    examine_with_correlated_column(drop_last=False)

    n_categories = 4
    n_cardinality = 5
    coincidentally_square_1hot(True, ((n_categories * (n_cardinality - 1)) + 1), n_cardinality, n_categories)
    coincidentally_square_1hot(False, ((n_categories * n_cardinality) + 1), n_cardinality, n_categories)

    det_of_square_1hot_matrix(True)
    det_of_square_1hot_matrix(False)
    random_matrix()
