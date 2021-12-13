from glm_coeff_plot import df_and_label_for
import matplotlib.pyplot as plt
import sys
import numpy as np


if __name__ == '__main__':
    colours = ['r', 'b', 'g', 'y', 'k', 'c', 'm']
    markers = ["1", "2", "3", "4", "s", "p", "P"]
    files = sys.argv[1:]
    labels =[]
    plt.xlabel("STP")
    plt.ylabel("Negative Log Likelihood (000s)")
    for i, file in enumerate(files):
        df, label = df_and_label_for(file)
        df.reindex(df.stp_code.sort_values().index)
        print(f"{label} has {len(df)} STPs")
        xs = map(lambda x: x + (i * 0.2), range(42))
        plt.bar(list(xs), (df['log_likelihood'].abs() / 1000), color=colours[i])
    plt.xticks(range(42), df['stp_code'], rotation='vertical')
    lgnd = plt.legend(loc="lower right", numpoints=len(files), fontsize=10)
    plt.show()
