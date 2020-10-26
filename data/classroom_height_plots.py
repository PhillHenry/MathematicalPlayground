import graphics.heatmap as g
import distributions.covariance as cov
import graphics.files as f
import matrix.ops as ops
from data.ClassroomHeights import ClassroomHeights


if __name__ == "__main__":
    d = ClassroomHeights(10, 100)
    c = cov.covariance_of(d.m)
    p = ops.condition_and_invert(c)

    g.add_heatmap(d.m, "Data")
    f.save_plot("/tmp/class_data.png")
    g.add_heatmap(c, "Covariance Matrix")
    f.save_plot("/tmp/class_covariance.png")
    g.add_heatmap(p, "Precision Matrix")
    f.save_plot("/tmp/class_precision.png")
