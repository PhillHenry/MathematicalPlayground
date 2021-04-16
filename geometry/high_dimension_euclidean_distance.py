import numpy as np
import random
from scipy.spatial import distance
import matplotlib.pyplot as plt


def perturb(basis, stddev):
    perturbed = []
    for x in basis:
        perturbation = random.gauss(x, stddev)
        perturbed.append(x + perturbation)
    return perturbed


def normalize(xs):
    return xs / np.linalg.norm(xs)


def perturbations(centroid, n_vecs, stddev):
    vectors = []
    for _ in range(n_vecs):
        vectors.append(perturb(centroid, stddev))
    return normalize(vectors)
    # return vectors


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
    flattened = [item for sublist in ds for item in sublist]
    return flattened


def distances_between(xs, ys, origin):
    if origin is None:
        return euclidean_distances(xs, ys)
    else:
        return euclidean_distances(xs, origin)


def calc_distances(neighbourhood1, neighbourhood2, origin):
    distances_neighbourhood1 = distances_between(neighbourhood1, neighbourhood1, origin)
    distances_neighbourhood2 = distances_between(neighbourhood2, neighbourhood2, origin)
    distances_intra = euclidean_distances(neighbourhood1, neighbourhood2)
    mean_neighbourhood1 = np.mean(distances_neighbourhood1)
    mean_neighbourhood2 = np.mean(distances_neighbourhood2)
    mean_intra_neighbourhoods = np.mean(distances_intra)
    return mean_neighbourhood1, mean_neighbourhood2, mean_intra_neighbourhoods


def make_neighbourhoods(n_dimensions, low, high, n_neighbours, stdev):
    centroid1 = np.random.uniform(low, high, n_dimensions)
    centroid2 = np.random.uniform(low, high, n_dimensions)
    neighbourhood1 = perturbations(centroid1, n_neighbours, stdev)
    neighbourhood2 = perturbations(centroid2, n_neighbours, stdev)
    return neighbourhood1, neighbourhood2


def display_results():
    low = 0.
    high = 1.
    n_neighbours = 30
    stdev = (high - low) / 5
    n_samples = 30

    print("low = {}, high = {}, number of neighbour = {}, stddev = {}".format(low, high, n_neighbours, stdev))

    all_results = []
    xs = []
    ys = []

    for n_dimensions in range(2, 100):
        means1 = []
        means2 = []
        intras = []
        origin = [[0] * n_dimensions]
        for _ in range(n_samples):
            neighbourhood1, neighbourhood2 = make_neighbourhoods(n_dimensions, low, high, n_neighbours, stdev)
            mean1, mean2, mean_intra = calc_distances(neighbourhood1, neighbourhood2, origin)
            means1.append(mean1)
            means2.append(mean2)
            intras.append(mean_intra)
        mu1 = np.mean(means1)
        mu2 = np.mean(means2)
        mu_intras = np.mean(intras)
        mean_ratio = (mu1 + mu2) / (2 * mu_intras)
        print("{:3d}-dimensions: mean neighbourhoods #1 = {:05f}, #2 = {:05f}, mean intras = {:05f}, mean ratios = {:05f}".
              format(n_dimensions, mu1, mu2, mu_intras, mean_ratio))
        results = [n_dimensions, mu1, mu2, mu_intras, mean_ratio]
        all_results.append(",".join(map(lambda x: str(x), results)))
        xs.append(n_dimensions)
        ys.append(mean_ratio)
    print("number_dimensions,mean_distance_cluster_1,mean_distance_cluster_2,mean_distance_intra_cluster,average_intra_vs_inter_disntance")
    for x in all_results:
        print(x)
    plot_results(xs, ys)


def plot_results(xs, ys):
    plt.plot(xs, ys)
    plt.xlabel("Dimensions")
    plt.ylabel("Ratios")
    plt.title("Intra vs Inter mean clusters for Normalized, Gaussian distributed points")
    plt.show()

if __name__ == "__main__":
    display_results()