from sklearn import linear_model
import numpy as np
from models.lin_reg_utils import train
from data.one_hot_encodings import make_fake_1hot_encodings, make_y


if __name__ == "__main__":
    np.set_printoptions(precision=3)
    n_rows = 1000
    n_categories = 4
    n_cardinality = 5
    m = make_fake_1hot_encodings(drop_last=False, n_rows=n_rows, n_categories=n_categories, n_cardinality=n_cardinality)
    ys = make_y(m)
    cols_to_drop = [i * n_cardinality for i in range(n_categories)]
    m_dropped = np.delete(m, cols_to_drop, 1)
    n_train = int(n_rows * 0.8)

    for intercept in [True, False]:
        model = linear_model.LinearRegression(fit_intercept=intercept)
        train(model, n_train, m, ys)
        train(model, n_train, m_dropped, ys)
