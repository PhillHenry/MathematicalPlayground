import pandas as pd
import seaborn as sns
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import sys
import re as re


COEFFICIENTS = "coefficients"


def interesting(fun):
    return fun[fun["feature"].str.contains("ethnic") |
               ~fun["feature"].str.contains("deprivation") |
               ~fun["feature"].str.contains("decile")]


def plot(df, c, label):
    plt.scatter(np.log(df[COEFFICIENTS].abs()), np.log(df.standard_error), c=c, label=label,
                s=np.log((1/df["p_values"]))
                # s=2
                )


if __name__ == '__main__':
    colours = ['r', 'b', 'g']
    files = sys.argv[1:]

    for i, file in enumerate(files):
        label = re.sub(".*/", "", file)
        label = re.sub("\..*", "", label)
        print(f"{i} Reading {file} and giving it label {label}")
        df = pd.read_csv(file, sep="\t")
        # if "1hot" not in label:
        # 'correct' IMDs if they're deciles
        df[COEFFICIENTS] = np.where(df["feature"].str.contains("imd19_decile"), df[
            COEFFICIENTS] * 10, df[COEFFICIENTS])
        cleaned = df[(df["p_values"] < 0.05)
                     & ((df[COEFFICIENTS] > 0.1) | (df[COEFFICIENTS] < -0.1))]
        significant = interesting(cleaned)
        plot(significant, colours[i], label)
    lgnd = plt.legend(loc="lower right", numpoints=len(files), fontsize=10)
    for i in range(len(files)):
        lgnd.legendHandles[i]._sizes = [30]
    plt.xlabel("Log |coefficients|")
    plt.ylabel("Log standard error")
    # plt.legend()
    plt.show()
