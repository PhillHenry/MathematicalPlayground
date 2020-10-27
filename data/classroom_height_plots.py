import random as r

import numpy as np

import distributions.covariance as cov
import graphics.files as f
import graphics.heatmap as g
from data.ClassroomHeights import ClassroomHeights

if __name__ == "__main__":
    d = ClassroomHeights(10, 100)
    (height, width) = np.shape(d.m)
    # add some dependency
    rows = []
    for i in range(height):
        row = d.m[i, :]
        if r.uniform(0, 1) < 0.05:
            ignore_1 = row[:, 0:-1]
            mean = cov.row_mean_of(ignore_1)
            centred = ignore_1 - mean
            new_row = np.append([0.], centred)
            print("new_row = {}".format(new_row))
            rows.append(np.asmatrix(new_row))
        else:
            rows.append(row - cov.row_mean_of(row))
    c = cov.squared_with_bessel(np.asmatrix(np.stack(rows)))

    p = np.linalg.inv(c)

    g.add_heatmap(d.m, "Data")
    f.save_plot("/tmp/class_data.png")
    g.add_heatmap(c, "Covariance Matrix")
    f.save_plot("/tmp/class_covariance.png")
    g.add_heatmap(p, "Precision Matrix")
    f.save_plot("/tmp/class_precision.png")
