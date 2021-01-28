import pymc3 as pm
import numpy as np
import theano.tensor as T

import matplotlib.pyplot as plt


# see https://en.wikipedia.org/wiki/Himmelblau%27s_function
def himmelblau(x, y):
    return (x ** 2 + y - 11) ** 2 + (x + y**2 - 7)**2


def extract(name, traces):
    return list(map(lambda t: t[name], traces))


def true_himmelblau(lim, step):
    grid_coords = [(x, y) for x in range(step) for y in range(step)]
    m = np.zeros([step, step])


    def grid_coord_for(coord):
        return (coord / (2 * lim)) - lim


    for grid_x, grid_y in grid_coords:
        coord_x = grid_coord_for(grid_x)
        coord_y = grid_coord_for(grid_y)
        z = himmelblau(coord_x, coord_y)
        # print("x = {}, y = {}, z = {}".format(coord_x, coord_y, z))
        m[grid_x, grid_y] = z

    return m


def heat_map_of(m, side, ax):
    heatmap = ax.imshow(m, cmap='hot', interpolation='nearest')
    # plt.colorbar(heatmap)
    ax.set_title("True Contours")
    ticks = list(map(lambda x: str(x), np.arange(-lim, lim)))
    tick_pos = list(range(0, side, side // len(ticks)))
    ax.set_xticks(tick_pos)
    ax.set_yticks(tick_pos)
    ax.set_xticklabels(ticks)
    ax.set_yticklabels(ticks)


def explore_himmelblau(lim, x_label, y_label, z_label):
    with pm.Model() as my_model:
        grid_x = pm.Uniform(x_label, -lim, lim)
        grid_y = pm.Uniform(y_label, -lim, lim)
        p = pm.Deterministic(z_label, himmelblau(grid_x, grid_y))
        step1 = pm.Metropolis(vars=[grid_x, grid_y])

        trace = pm.sample(10000, step=[step1])

    return list(trace)


if __name__ == "__main__":
    x_label = "x"
    y_label = "y"
    z_label = "obs"

    lim = 5

    f, (ax1, ax2) = plt.subplots(2, 1, sharey=False)

    step = 100
    m = true_himmelblau(lim, step)
    heat_map_of(m, step, ax1)

    traces = explore_himmelblau(lim, x_label, y_label, z_label)
    print(traces[:5])
    xs = extract(x_label, traces)
    ys = extract(y_label, traces)
    cs = extract(z_label, traces)

    ax2.scatter(xs, ys, s=2, cmap='jet', c=cs)
    ax2.set_title("Traces")

    plt.show()
