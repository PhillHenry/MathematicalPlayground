import numpy as np
import random
import math


np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)


def transitions(n=6):
    a = np.random.rand(n, n) #.astype("float64")
    max_wait = min(n // 2, n - 3)
    # waiting weeks can't go backwards
    for i in range(max_wait):
        for j in range(max_wait):
            if j != i + 1:
                a[i][j] = 0.
    # other states cannot go into waiting weeks
    for i in range(0, max_wait):
        for j in range(n - max_wait, n):
            a[j][i] = 0.
    # ... except the "Remain" group
    p_diagnosis = 1e-3
    p_leave = 1e-3
    a[n-1, :] = 0.
    a[:, n-1] = p_leave
    a[n-1, 0] = p_diagnosis
    a[n-1, n-1] = 1 - p_diagnosis
    row_sums = a.sum(axis=1)
    return a / row_sums[:, np.newaxis]
    # return a / a.sum(axis=1).sum(axis=2)


def markov(pos, neighbor, weights):
    n_iter = int(1e6)
    histo = {}
    n = len(neighbor)
    for i in range(n):
        histo[i] = 0
    for _ in range(n_iter):
        new_neighbour = random.randint(0, n - 1)
        new_pos = neighbor[pos][new_neighbour]
        new_weight = weights[new_neighbour]
        old_weight = weights[pos]
        if random.random() < new_weight / old_weight:
            pos = new_pos
        histo[pos] += 1
    return histo


def neighbours(x):
    n = len(x)
    neighbor = []

    for i in range(n):
        xs = []
        row = x[i]
        for j in range(n):
            if row[j] == 0:
                xs.append(i)
            else:
                xs.append(j)
        neighbor.append(xs)

    return neighbor


def do_markov(weights, n):
    print(f"weights = {weights}")
    ns = neighbours(original)
    histo = markov(n - 1, ns, weights)
    print(f"MCMC:\n{histo}")


if __name__ == "__main__":
    n = 6
    original = transitions(n)
    x = original

    print(f"initial matrix:\n{x}")

    eigen_vals, eigen_vecs_as_columns = np.linalg.eig(x)

    print(f"Eigenvectors:\n{eigen_vecs_as_columns}")  # unit length but not necessarily orthogonal
    print(f"Eigenvalues:\n{eigen_vals}")
    # print(f"Check orthonormal:\n{np.dot(np.transpose(eigen_vecs_as_columns), eigen_vecs_as_columns)}")

    for i, v in enumerate(eigen_vals):
        if math.isclose(v.real, 1.0):
            print(f"dominant eigenvalue at position {i}")
            dominant_index = i

    dominant_vec = eigen_vecs_as_columns[:, dominant_index]
    print(f"Dominant eigen vector = {dominant_vec}")
    dominant = np.zeros([n, n])
    dominant[dominant_index][dominant_index] = 1.0
    reconstituted = np.dot(np.dot(eigen_vecs_as_columns, dominant), np.linalg.inv(eigen_vecs_as_columns))
    print(f"Reconstituted:\n{reconstituted}")

    iterations = 50
    for _ in range(iterations):
        x = np.dot(x, x)

    print(f"After {iterations} the matrix looks like:\n{x}")

    do_markov([1] * n, n)
    weights = []
    for i in range(n):
        w = sum(original[:, i])
        weights.append(w)
    do_markov(weights, n)

