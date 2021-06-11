import sys
import re
import matplotlib.pyplot as plt
import numpy as np


if __name__ == "__main__":
    with open(sys.argv[1], "r") as file:
        content = str(file.readlines())

        num_rows = int(re.search(""""numRows":"(\d+)""", content, re.IGNORECASE).group(1))
        num_cols = int(re.search(""""numCols":"(\d+)""", content, re.IGNORECASE).group(1))
        values = re.search(""""values":"\[([0-9\s\.\,\-]+).*""", content, re.IGNORECASE).group(1)

        print(num_rows)
        print(num_cols)
        print(len(values.split(",")))

        cells = [float(x) for x in values.split(",")]
        m = np.reshape(cells, [num_rows, num_cols])

        labels = ["specialty_9", "specialty_8", "specialty_6", "specialty_5", "specialty_4", "specialty_3", "specialty_2", "specialty_1", "IMD_9", "IMD_8", "IMD_7", "IMD_6", "IMD_5", "IMD_4", "IMD_3", "IMD_2", "IMD_10", "IMD_1", "RoP", "80+", "75-79", "70-74", "65-69", "60-64", "55-59", "50-54", "40-49", "30-39", "18-29", "White", "Other", "Mixed", "Black", "Asian"]
        print(len(labels))
        fig = plt.figure(figsize=(7,8))
        ax1 = fig.add_subplot(111)
        for i in range(num_rows - 1):
        #     for j in range(num_cols):
            j = 0
            ax1.text(j, i, "{:.3f}".format(m[i, num_cols - 1]), va='center', ha='center', color='#00ffff', fontsize=10)
        correlations = m[:num_rows - 1, num_cols - 1:num_cols]
        print(correlations)
        heatmap = plt.imshow(correlations, cmap='hot', interpolation='nearest', aspect="auto")
        plt.colorbar(heatmap)
        ax1.set_xticks([1])
        ax1.set_yticks(np.arange(len(labels)))
        ax1.set_xticklabels(["weeks"], rotation=45, fontsize=9)
        ax1.set_yticklabels(labels, fontsize=10)
        plt.savefig("/tmp/heatmap.png")
        plt.show()
