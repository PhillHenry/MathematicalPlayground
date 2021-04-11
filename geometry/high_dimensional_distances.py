import random
import numpy as np
from scipy.spatial import distance
import matplotlib.pyplot as plt


def make_vector(d):
    return np.random.uniform(0., 1., d)


def make_vectors(d, n):
    vectors = []
    for _ in range(n):
        vectors.append(make_vector(d))
    return vectors


def min_max_distances_between(vectors):
    min_distance = float("inf")
    max_distance = float("-inf")
    n = len(vectors)
    for i in range(n):
        for j in range(n):
            if i < j:
                dist = distance.euclidean(vectors[i], vectors[j])
                min_distance = min(dist, min_distance)
                max_distance = max(dist, max_distance)
    return min_distance, max_distance


def find_ratios():
    with open("/tmp/high_dimensional_distances.csv", "w") as file:
        print("dimension,min,max,ratio", file=file)
        xs = []
        ys = []
        for d in range(2, 30):
            min_distance, max_distance = min_max_distances_between(make_vectors(d, 30))
            ratio = (max_distance - min_distance) / max_distance
            print("d = {}, min = {}, max = {}, ratio = {}".format(d, min_distance, max_distance, ratio))
            print(f"{d},{min_distance},{max_distance},{ratio}", file=file)
            xs.append(d)
            ys.append(ratio)
    plt.plot(xs, ys)
    plt.xlabel("Dimensions")
    plt.ylabel("Ratios")
    plt.title("Ratios of [min-max]/max for uniform distributed points")
    plt.show()


if __name__ == "__main__":
    find_ratios()
