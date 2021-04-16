import json
import sys
import re


if __name__ == "__main__":
    with open(sys.argv[1], "r") as file:
        content = str(file.readlines())

        num_rows = re.search(""""numRows":"(\d+)""", content, re.IGNORECASE).group(1)
        num_cols = re.search(""""numCols":"(\d+)""", content, re.IGNORECASE).group(1)
        values = re.search(""""values":"\[([0-9\s\.\,\-]+).*""", content, re.IGNORECASE).group(1)

        print(num_rows)
        print(num_cols)
        print(len(values.split(",")))
