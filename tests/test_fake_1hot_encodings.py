from data.one_hot_encodings import make_fake_1hot_encodings
import numpy as np


def check_columns_contain_ones(m):
    n_cols = np.shape(m)[1]
    for i in range(n_cols):
        assert sum(m[:, i]) != 0


def test_every_row_has_a_1_for_each_category_when_no_drop_last():
    n_categories = 4
    n_cardinality = 5
    m = make_fake_1hot_encodings(n_categories=n_categories, n_cardinality=n_cardinality, drop_last=False)
    check_columns_contain_ones(m)
    for row in m:
        assert sum(row) == n_categories
        assert len(row) == n_categories * n_cardinality


def test_fewer_rows_when_drop_last():
    n_categories = 4
    n_cardinality = 5
    m = make_fake_1hot_encodings(n_categories=n_categories, n_cardinality=n_cardinality, drop_last=True)
    check_columns_contain_ones(m)
    for row in m:
        assert len(row) == n_categories * (n_cardinality - 1)
        assert sum(row) <= n_categories
