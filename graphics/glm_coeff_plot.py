import re as re
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COEFFICIENTS = "coefficients"
FEATURE = "feature"


def interesting(fun):
    return fun[fun[FEATURE].str.contains("ethnic") |
               fun[FEATURE].str.contains("deprivation") |
               fun[FEATURE].str.contains("decile")]


def plot(df, c, label, marker=None):
    plt.scatter(np.log(df[COEFFICIENTS].abs()), np.log(df.standard_error), c=c, label=label,
                # s=np.log((1/df["p_values"])) * 0.1
                # s=10,
                marker=marker
                )


if __name__ == '__main__':
    colours = ['r', 'b', 'g', 'y', 'k', 'c', 'm']
    markers = ["1", "2", "3", "4", "s", "p", "P"]
    files = sys.argv[1:]
    labels =[]
    for i, file in enumerate(files):
        label = re.sub(".*/", "", file)
        label = re.sub("\..*", "", label)
        print(f"{i} Reading {file} and giving it label {label}")
        df = pd.read_csv(file, sep="\t")
        # 'correct' IMDs if they're deciles
        df[COEFFICIENTS] = np.where(df[FEATURE].str.contains("imd19_decile"), df[
            COEFFICIENTS] * 10, df[COEFFICIENTS])
        cleaned = df[(df["p_values"] < 0.05)
                     & ((df[COEFFICIENTS] > 0.1) | (df[COEFFICIENTS] < -0.1))]
        significant = interesting(cleaned)
        print(f"Number of significant features for {label} = {len(significant)}")
        plot(significant, colours[i], label, markers[i])
        labels.append(label)
    lgnd = plt.legend(loc="lower right", numpoints=len(files), fontsize=10)
    for i in range(len(files)):
        lgnd.legendHandles[i]._sizes = [30]
    plt.xlabel("Log |coefficients|")
    plt.ylabel("Log standard error")
    plt.savefig(f"/tmp/{'-'.join(labels)}.png")
    plt.show()
