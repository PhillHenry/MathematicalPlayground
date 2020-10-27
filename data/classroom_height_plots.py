import graphics.heatmap as g
import distributions.covariance as cov
import graphics.files as f
import matrix.ops as ops
from data.ClassroomHeights import ClassroomHeights
import numpy as np


if __name__ == "__main__":
    d = ClassroomHeights(10, 100)
    c = cov.covariance_of(d.m)
    (height, width) = np.shape(c)
    # add some dependency
    for i in range(10):
        x = int(np.random.uniform(0, height))
        y = int(np.random.uniform(0, width))
        c[x, y] = 0.
        c[y, x] = 0.
    p = np.linalg.inv(c)

    g.add_heatmap(d.m, "Data")
    f.save_plot("/tmp/class_data.png")
    g.add_heatmap(c, "Covariance Matrix")
    f.save_plot("/tmp/class_covariance.png")
    g.add_heatmap(p, "Precision Matrix")
    f.save_plot("/tmp/class_precision.png")
