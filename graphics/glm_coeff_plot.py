import pandas as pd
import seaborn as sns
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import sys
import re as re


def interesting(fun):
    return fun[fun["feature"].str.contains("ethnic") |
        ~fun["feature"].str.contains("deprivation")]


def plot(df, c, label):
    plt.scatter(np.log(df.coefficients), np.log(df.standard_error), c=c, label=label)


if __name__ == '__main__':
    colours = ['r', 'b', 'g']
    for i, file in enumerate(sys.argv[1:]):
        label = re.sub(".*/", "", file)
        label = re.sub("\..*", "", label)
        print(f"Reading {file} and giving it label {label}")
        df = pd.read_csv(file, sep="\t")
        # if "1hot" not in label:
        # 'correct' IMDs if they're deciles
        df["coefficients"] = np.where(df["feature"].str.contains("imd19_decile"), df["coefficients"] * 10, df["coefficients"])
        cleaned = df[(df["p_values"] < 0.05)
                     & ((df["coefficients"] > 0.1) | (df["coefficients"] < -0.1))]
        significant = interesting(cleaned)
        plot(significant, colours[i], label)
    plt.xlabel("Log coefficients")
    plt.ylabel("Log standard error")
    plt.legend()
    plt.show()
