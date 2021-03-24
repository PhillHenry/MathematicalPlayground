import numpy as np
import random


def perturb(basis, stddev):
    perturbed = []
    for x in basis:
        perturbation = random.gauss(x, stddev)
        perturbed.append(x + perturbation)
    return perturbed


def perturbations(centroid, n_vecs, stddev):
    vectors = []
    for _ in range(n_vecs):
        vectors.append(perturb(centroid, stddev))
    return vectors


def euclidean_distances_between(point, others):
    ds = []
    b = np.array(point)
    for x in others:
        a = np.array(x)
        ds.append(np.linalg.norm(a - b, ord=1))
    return ds


def euclidean_distances(xs, ys):
    ds = []
    for x in xs:
        ds.append(euclidean_distances_between(x, ys))
    return ds


if __name__ == "__main__":
    n_dimensions = 2000
    low = 0
    high = 100
    centroid1 = np.random.uniform(low, high, n_dimensions)
    centroid2 = np.random.uniform(low, high, n_dimensions)
    n_neighbours = 10
    stdev = (high - low) / 10
    neighbourhood1 = perturbations(centroid1, n_neighbours, stdev)
    neighbourhood2 = perturbations(centroid2, n_neighbours, stdev)

    mean_neighbourhood1 = np.mean(euclidean_distances(neighbourhood1, neighbourhood1))
    mean_neighbourhood2 = np.mean(euclidean_distances(neighbourhood2, neighbourhood2))
    mean_intra_neighbourhoods = np.mean(euclidean_distances(neighbourhood1, neighbourhood2))
    print("mean distance in neighbourhood1 = {}".format(mean_neighbourhood1))
    print("mean distance in neighbourhood2 = {}".format(mean_neighbourhood2))
    print("mean distance between neighbourhoods = {}".format(mean_intra_neighbourhoods))
    print("ratio of means {} and {}".format(mean_neighbourhood1/mean_intra_neighbourhoods, mean_neighbourhood2/mean_intra_neighbourhoods))

