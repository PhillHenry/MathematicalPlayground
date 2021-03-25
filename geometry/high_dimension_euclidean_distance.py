import numpy as np
import random
from scipy.spatial import distance


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
        # dist = np.linalg.norm(a - b, ord=1)
        dist = distance.euclidean(point, x)
        ds.append(dist)
    return ds


def euclidean_distances(xs, ys):
    ds = []
    for x in xs:
        ds.append(euclidean_distances_between(x, ys))
    return ds


def calc_distances(neighbourhood1, neighbourhood2):
    distances_neighbourhood1 = euclidean_distances(neighbourhood1, neighbourhood1)
    distances_neighbourhood2 = euclidean_distances(neighbourhood2, neighbourhood2)
    distances_intra = euclidean_distances(neighbourhood1, neighbourhood2)
    mean_neighbourhood1 = np.mean(distances_neighbourhood1)
    mean_neighbourhood2 = np.mean(distances_neighbourhood2)
    mean_intra_neighbourhoods = np.mean(distances_intra)
    # print("mean distance in neighbourhood1      = {} max = {}".format(mean_neighbourhood1, np.max(distances_neighbourhood1)))
    # print("mean distance in neighbourhood2      = {} max = {}".format(mean_neighbourhood2, np.max(distances_neighbourhood2)))
    # print("mean distance between neighbourhoods = {} min = {}".format(mean_intra_neighbourhoods, np.min(distances_intra)))
    # print("ratio of means {} and {}".format(mean_neighbourhood1/mean_intra_neighbourhoods, mean_neighbourhood2/mean_intra_neighbourhoods))
    return mean_neighbourhood1, mean_neighbourhood2, mean_intra_neighbourhoods


def make_neighbourhoods(n_dimensions, low, high, n_neighbours, stdev):
    centroid1 = np.random.uniform(low, high, n_dimensions)
    centroid2 = np.random.uniform(low, high, n_dimensions)
    neighbourhood1 = perturbations(centroid1, n_neighbours, stdev)
    neighbourhood2 = perturbations(centroid2, n_neighbours, stdev)
    return neighbourhood1, neighbourhood2


if __name__ == "__main__":
    low = 0
    high = 100
    n_neighbours = 100
    stdev = (high - low) / 5
    n_samples = 30

    print("low = {}, high = {}, number of neighbour = {}, stddev = {}".format(low, high, n_neighbours, stdev))

    all_results = []

    for n_dimensions in range(3, 24, 2):
        means1 = []
        means2 = []
        intras = []
        for _ in range(n_samples):
            neighbourhood1, neighbourhood2 = make_neighbourhoods(n_dimensions, low, high, n_neighbours, stdev)
            mean1, mean2, mean_intra = calc_distances(neighbourhood1, neighbourhood2)
            means1.append(mean1)
            means2.append(mean2)
            intras.append(mean_intra)
        mu1 = np.mean(means1)
        mu2 = np.mean(means2)
        mu_intras = np.mean(intras)
        mean_ratio = (mu1 + mu2) / (2 * mu_intras)
        print("{}-dimensions: mean neighbourhoods #1 = {}, #2 = {}, mean intras = {}, mean ratios = {}".
              format(n_dimensions, mu1, mu2, mu_intras, mean_ratio))
        results = [n_dimensions, mu1, mu2, mu_intras, mean_ratio]
        all_results.append(",".join(map(lambda x: str(x), results)))
    for x in all_results:
        print(x)

