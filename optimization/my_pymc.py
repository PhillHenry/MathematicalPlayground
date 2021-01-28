import pymc3 as pm
import numpy as np
import theano.tensor as T
from theano.tensor import _shared
import scipy.stats as stats

import matplotlib.pyplot as plt


# see https://en.wikipedia.org/wiki/Himmelblau%27s_function
def himmelblau(x, y):
    # print("x = {}, y ={}".format(x, y))
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

# define a theano Op for our likelihood function
class LogLike(T.Op):

    """
    Specify what type of object will be passed and returned to the Op when it is
    called. In our case we will be passing it a vector of values (the parameters
    that define our model) and returning a single "scalar" value (the
    log-likelihood)
    """

    itypes = [T.dvector]  # expects a vector of parameter values when called
    otypes = [T.dscalar]  # outputs a single scalar value (the log likelihood)

    def __init__(self, loglike, x, data):
        """
        Initialise the Op with various things that our log-likelihood function
        requires. Below are the things that are needed in this particular
        example.

        Parameters
        ----------
        loglike:
            The log-likelihood (or whatever) function we've defined
        """

        # add inputs as class attributes
        self.likelihood = loglike
        self.x = x
        self.data = data

    def perform(self, node, inputs, outputs):
        # the method that is used when calling the Op

        theta, = inputs  # this will contain my variables

        # print("inputs = {}".format(inputs))
        # print("len(inputs) = {}".format(len(inputs)))
        # print("len(theta) = {}".format(len(theta)))
        x, y = theta
        # print("x = {}, y = {}".format(x, y))

        # call the log-likelihood function
        logl = self.likelihood(x, y)

        outputs[0][0] = np.array(logl)  # output the log-likelihood


def explore_himmelblau(lim, x_label, y_label, z_label):
    with pm.Model() as my_model:
        grid_x = pm.Uniform(x_label, -lim, lim)
        grid_y = pm.Uniform(y_label, -lim, lim)
        p = pm.Deterministic(z_label, himmelblau(grid_x, grid_y))
        step1 = pm.Metropolis(vars=[grid_x, grid_y])

        trace = pm.sample(10000, step=[step1])

    return list(trace)


def gradient(lim, x_label, y_label, z_label, x, m):
    fn = lambda x, y: himmelblau(x, y)
    logl = LogLike(fn, x, m)
    with pm.Model() as my_model:
        grid_x = pm.Uniform(x_label, -lim, lim)
        grid_y = pm.Uniform(y_label, -lim, lim)
        theta = T.as_tensor_variable([grid_x, grid_y])
        theano_m = _shared(m)
        p = pm.DensityDist(z_label, lambda v: logl(v), observed=theta)
        normal_dist = stats.norm(loc=0, scale=1.)
        # pm.DensityDist(z_label, lambda x: x, observed=np.random.randn(100))
        trace = pm.sample(100000, tune=10000, discard_tuned_samples=True)

    return list(trace)


# see https://docs.pymc.io/notebooks/blackbox_external_likelihood.html
if __name__ == "__main__":
    x_label = "x"
    y_label = "y"
    z_label = "obs"

    lim = 5

    f, (ax1, ax2) = plt.subplots(2, 1, sharey=False)

    step = 100
    m = true_himmelblau(lim, step)
    heat_map_of(m, step, ax1)

    support = np.arange(0, 10)
    mesh = np.meshgrid(support, support)
    line = np.linspace(-5, 5., 100)

    traces = gradient(lim, x_label, y_label, z_label, mesh, m)

    print("sample traces = {}".format(traces[:5]))
    xs = extract(x_label, traces)
    ys = extract(y_label, traces)
    # cs = extract(z_label, traces)

    # ax2.scatter(xs, ys, s=2, cmap='jet', c=cs)
    ax2.plot(xs, ys)
    ax2.set_title("Traces")

    # _ = pm.traceplot(traces, lines=((x_label, {}, [0.1]), (y_label, {}, [0.1])))

    plt.show()
