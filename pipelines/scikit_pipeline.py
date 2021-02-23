from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.compose import ColumnTransformer


def vectorizer_df(input_data, categorical_cols, numerical_cols):
    categorical_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent'))
    ])
    numerical_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('bucketizer', KBinsDiscretizer(n_bins=10, strategy='uniform', encode='ordinal'))  # ordinal
    ])
    preprocessing = ColumnTransformer(
        [('cat', categorical_pipe, categorical_cols),
         ('num', numerical_pipe, numerical_cols)
         ])
    vectorizer_pipeline = Pipeline([
        ('vectorize', preprocessing)
    ])
    return vectorizer_pipeline.fit_transform(input_data)