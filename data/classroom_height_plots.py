import matplotlib.pyplot as plt
import distributions.covariance as cov
import graphics.files as f
import matrix.ops as ops
from data.ClassroomHeights import ClassroomHeights


def add_heatmap(m, title):
    heatmap = plt.imshow(m, cmap='hot', interpolation='nearest')
    plt.colorbar(heatmap)
    plt.title(title)


if __name__ == "__main__":
    d = ClassroomHeights(10, 100)
    c = cov.covariance_of(d.m)
    p = ops.condition_and_invert(c)

    add_heatmap(d.m, "Data")
    f.save_plot("/tmp/class_data.png")
    add_heatmap(c, "Covariance Matrix")
    f.save_plot("/tmp/class_covariance.png")
    add_heatmap(p, "Precision Matrix")
    f.save_plot("/tmp/class_precision.png")
