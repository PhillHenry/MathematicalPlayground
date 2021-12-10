import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import sys


def interesting(fun):
    return fun[~fun["feature"].str.contains("intercept") &
        ~fun["feature"].str.contains("arrival") &
        ~fun["feature"].str.contains("acuity") &
        ~fun["feature"].str.contains("priority") &
        ~fun["feature"].str.contains("RTT")]


def plot(df):
    # fig, ax = plt.subplots(figsize=(12,8))
    # df.plot(ax=ax,legend=True)
    df.plot(kind='scatter', x='coefficients', y=['standard_error']).plot(figsize=(12,8))
    plt.show()


if __name__ == '__main__':
    for file in sys.argv[1:]:
        print(f"Reading {file}")
        df = pd.read_csv(file, sep="\t")
        cleaned = df[(df["p_values"] < 0.05) & ((df["coefficients"] > 0.1) | (df["coefficients"] < -0.1))]
        significant = interesting(cleaned)
        plot(significant)
