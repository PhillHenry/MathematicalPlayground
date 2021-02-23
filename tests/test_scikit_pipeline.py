import pipelines.scikit_pipeline as to_test
import pandas as pd


def test_smoke():
    rows = [(1, "a", "A", "X"), (2, "b", "B", "Y"), (3, "c", "C", "Z"), (4, "", "D", "W")]
    col_number = "number"
    col_category = "cat"
    df = pd.DataFrame(rows, columns=[col_number, col_category, "CAT", "godknows"])

    imputed = to_test.vectorizer_df(df, [col_category], [col_number])
    assert(len(imputed) == len(rows))
