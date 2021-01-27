import pymc3 as pm
import numpy as np
import theano.tensor as T


# see https://en.wikipedia.org/wiki/Himmelblau%27s_function
def himmelblau(x, y):
    return (x ** 2 + y - 11) ** 2 + (x + y**2 - 7)**2


if __name__ == "__main__":
    with pm.Model() as my_model:
        x = pm.Uniform("x", -10, 10)
        y = pm.Uniform("y", -10, 10)
        p = pm.Deterministic("obs", himmelblau(x, y))

        step1 = pm.Metropolis(vars=[p, x, y])
        trace = pm.sample(2000, step=[step1])

    print(list(trace))
