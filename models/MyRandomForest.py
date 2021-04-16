import pandas as pd
import sys
from sklearn.ensemble import RandomForestClassifier

if __name__ == "__main__":
    df = pd.read_csv(sys.argv[1], header=None)
    labels = pd.read_csv(sys.argv[2], header=None)
    # df_with_labels = df['labels'] = labels
    y = labels # df['labels']
    X = df # df[~df['labels']]
    clf = RandomForestClassifier(max_depth=20, random_state=0)
    clf.fit(X, y)
