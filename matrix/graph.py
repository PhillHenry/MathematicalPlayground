import numpy as np
import random


np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)


def transitions(n=6):
    a = np.random.rand(n, n) #.astype("float64")
    max_wait = min(n // 2, n - 3)
    # waiting weeks can't go backwards
    for i in range(max_wait):
        for j in range(max_wait):
            if j <= i:
                a[i][j] = 0.
    # other states cannot go into waiting weeks
    for i in range(0, max_wait):
        for j in range(n - max_wait, n):
            a[j][i] = 0.
    # ... except the majority of the population
    p_diagnosis = 1e-5
    p_leave = 1e-3
    a[n-1, :] = 0.
    a[:, n-1] = p_leave
    a[n-1, 0] = p_diagnosis
    a[n-1, n-1] = 1 - p_diagnosis
    row_sums = a.sum(axis=1)
    return a / row_sums[:, np.newaxis]
    # return a / a.sum(axis=1).sum(axis=2)


def markov(pos, neighbor, weight):
    n_iter = 1000000
    histo = {}
    n = len(neighbor)
    for i in range(n):
        histo[i] = 0
    for _ in range(n_iter):
        new_neighbour = random.randint(0, n - 1)
        new_pos = neighbor[pos][new_neighbour]
        new_weight = sum(weight[:, new_neighbour])
        old_weight = sum(weight[:, pos])
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


if __name__ == "__main__":
    n = 6
    original = transitions(n)
    x = original

    print(f"initial matrix:\n{x}")

    eigen_vals, eigen_vecs = np.linalg.eig(x)

    x_vecs = np.dot(x, eigen_vecs)
    print(f"x_vecs:\n{x_vecs}")

    print(f"Eigenvectors:\n{eigen_vecs}")
    print(f"Eigenvalues:\n{eigen_vals}")

    print("x . v_i")
    for i in range(n):
        eigen_vec = np.transpose(eigen_vecs)[i]
        eigen_val = eigen_vals[i]
        v = eigen_vec
        multiplied = np.dot(x, v)
        if eigen_val != 0:
            multiplied = multiplied / eigen_val
        print(multiplied)

    iterations = 30
    for _ in range(iterations):
        x = np.dot(x, x)

    print(f"After {iterations} the matrix looks like:\n{x}")

    histo = markov(n - 1, neighbours(x), x)
    print(f"MCMC:\n{histo}")
