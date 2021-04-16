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

        heatmap = plt.imshow(m, cmap='hot', interpolation='nearest')
        plt.colorbar(heatmap)
        plt.show()
