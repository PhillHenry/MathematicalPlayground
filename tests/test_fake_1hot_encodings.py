from data.one_hot_encodings import make_fake_1hot_encodings, drop_last
import numpy as np


def check_columns_contain_ones(m):
    n_cols = np.shape(m)[1]
    for i in range(n_cols):
        assert sum(m[:, i]) != 0


def num_rows_all_zero(m):
    return len(list(filter(lambda v: sum(v) == 0., m)))


def test_drop_columns():
    n_categories = 4
    n_cardinality = 5
    m = np.zeros([1, n_categories * n_cardinality])
    for i in range(n_categories * n_cardinality):
        m[0, i] = i
    dropped = drop_last(m, n_categories, n_cardinality)
    assert np.allclose(dropped[0], [0,1,2,3, 5,6,7,8, 10,11,12,13, 15,16,17,18])


def test_every_row_has_a_1_for_each_category_when_no_drop_last():
    n_categories = 4
    n_cardinality = 5
    m = make_fake_1hot_encodings(n_categories=n_categories, n_cardinality=n_cardinality, drop_last=False)
    check_columns_contain_ones(m)
    assert num_rows_all_zero(m) == 0
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


def test_some_rows_zero_vector_when_drop_last():
    m = make_fake_1hot_encodings(n_rows=10000, drop_last=True)
    assert num_rows_all_zero(m) > 0
