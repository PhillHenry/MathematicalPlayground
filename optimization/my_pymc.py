import pymc3 as pm
import numpy as np
import theano.tensor as T

import matplotlib.pyplot as plt


# see https://en.wikipedia.org/wiki/Himmelblau%27s_function
def himmelblau(x, y):
    return (x ** 2 + y - 11) ** 2 + (x + y**2 - 7)**2


def extract(name, traces):
    return list(map(lambda t: t[name], traces))


if __name__ == "__main__":
    x_label = "x"
    y_label = "y"
    z_label = "obs"

    lim = 5

    f, (ax1, ax2) = plt.subplots(2, 1, sharey=False)

    step = 100
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

    heatmap = ax1.imshow(m, cmap='hot', interpolation='nearest')
    # plt.colorbar(heatmap)
    ax1.set_title("True Contours")
    ticks = list(map(lambda x: str(x), np.arange(-lim, lim)))
    tick_pos = list(range(0, step, step // len(ticks)))
    ax1.set_xticks(tick_pos)
    ax1.set_yticks(tick_pos)
    ax1.set_xticklabels(ticks)
    ax1.set_yticklabels(ticks)

    with pm.Model() as my_model:
        grid_x = pm.Uniform(x_label, -lim, lim)
        grid_y = pm.Uniform(y_label, -lim, lim)
        p = pm.Deterministic(z_label, himmelblau(grid_x, grid_y))

        step1 = pm.Metropolis(vars=[p, grid_x, grid_y])
        trace = pm.sample(2000, step=[step1])

    traces = list(trace)
    print(traces)
    xs = extract(x_label, traces)
    ys = extract(y_label, traces)
    cs = extract(z_label, traces)

    ax2.scatter(xs, ys, s=2, cmap='jet', c=cs)
    ax2.set_title("Traces")


    plt.show()
