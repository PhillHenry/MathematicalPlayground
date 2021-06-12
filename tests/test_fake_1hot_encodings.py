from matrix.standardisation_vs_singular import make_fake_1hot_encodings


def test_every_row_has_a_1_for_each_category():
    n_categories = 4
    m = make_fake_1hot_encodings(n_categories=n_categories)
    for row in m:
        assert sum(row) == n_categories
