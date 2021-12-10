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


def plot(df, c):
    plt.scatter(df.coefficients, df.standard_error, c=c)


if __name__ == '__main__':
    colours = ['r', 'b']
    for i, file in enumerate(sys.argv[1:]):
        print(f"Reading {file}")
        df = pd.read_csv(file, sep="\t")
        cleaned = df[(df["p_values"] < 0.05) & ((df["coefficients"] > 0.1) | (df["coefficients"] < -0.1))]
        significant = interesting(cleaned)
        plot(significant, colours[i])
    plt.xlabel("Coefficients")
    plt.ylabel("Standard error")
    plt.show()
