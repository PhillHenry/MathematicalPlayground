import numpy as np
import random
import math


np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)


def transition_matrix(n=6):
    """
    Create the stochastic matrix.
    First `max_wait` states are the waiting list for week_1, week_2, ..., week_max_wait.
    The last state is the "remain" pool of dead, cured or about to enter the system.
    All states can possibly transition to the "remain" state (via column `n` of the matrix)

    :param n: Number of states
    :return: an n x n stochastic matrix. Values of at [i,j] are the probabilities of transitioning
             from state j to state i.
    """
    a = np.random.rand(n, n)
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
    m = a / row_sums[:, np.newaxis]
    return np.transpose(m)


def metropolis(pos, neighbor, weights, m):
    """
    The Metropolis algorithm. Stolen from
    https://github.com/marcelom/smac-001/blob/master/programs_lecture_1/pebble_basic_inhomogeneous.py
    (Coursera's "Statistical Mechanics: Algorithms and Computations")

    :param pos: Start state
    :param neighbor: list of list of neighbours for a given state
    :param weights:
    :param m: stochastic matrix
    :return: Historgram of state to number of iterations spent in it
    """
    n_iter = int(1e6)
    n = len(neighbor)
    histo = np.zeros([n, n])
    for i in range(n):
        histo[i] = 0
    for _ in range(n_iter):
        new_pos = neighbor[pos][random.randint(0, n - 1)]
        if generalized_metropolis(weights, m, pos, new_pos):  # TODO m should be the a priori
            pos = new_pos
        histo[new_pos][pos] += 1
    return histo


def generalized_metropolis(weights, m, a, b):
    """
    Equation 1.18 from 'Statistical Mechanic: Algorithms and Computations", Krauth
    :param weights:
    :param m: An a priori distribution
    :return: boolean indication whether to move or not
    """
    p_b = weights[b]
    p_a = weights[a]
    p_b_to_a = m[a][b]
    p_a_to_b = m[b][a]
    if p_a == 0 or p_a_to_b == 0:
        return True
    else:
        return random.random() < min(1, (p_b * p_b_to_a) / (p_a * p_a_to_b))


def generate_neighbours_list(transition_matrix):
    """

    :param transition_matrix:
    :return: the neighbours of each state
    """
    n = len(transition_matrix)
    neighbor = []

    for i in range(n):
        xs = []
        col = transition_matrix[:, i]
        for j in range(n):
            if col[j] == 0:
                xs.append(i)
            else:
                xs.append(j)
        neighbor.append(xs)

    return neighbor


def do_metropolis(weights, matrix):
    print(f"weights = {weights}")
    ns = generate_neighbours_list(matrix)
    histo = metropolis(0, ns, weights, matrix)
    print(f"MCMC:\n{histo}")


def inspect_eigenvectors(x):
    """
    :param x: The stochastic matrix
    :return: The S A S^-1 matrix where A is a matrix that contains only the dominant eigenvector.
             This should equal x^t as t -> infinity.
    """
    num_states = len(x)
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
    dominant = np.zeros([num_states, num_states])
    dominant[dominant_index][dominant_index] = 1.0
    reconstituted = np.dot(np.dot(eigen_vecs_as_columns, dominant), np.linalg.inv(eigen_vecs_as_columns))
    return reconstituted


def explore_stochastic_matrix():
    num_states = 6
    original = transition_matrix(num_states)
    matrix = original

    print(f"initial matrix:\n{matrix}")

    reconstituted = inspect_eigenvectors(matrix)
    print(f"Reconstituted:\n{reconstituted}")  # should be pretty close to matrix^t below

    iterations = 50
    for _ in range(iterations):
        matrix = np.dot(matrix, matrix)

    print(f"After {iterations} the matrix looks like:\n{matrix}")

    # PH - no idea what to use for the weights!
    # Try uniform values:
    do_metropolis([1] * num_states, original)

    # Try total of incoming probabilities
    weights = []
    for i in range(num_states):
        w = sum(original[i, :])
        weights.append(w)
    do_metropolis(weights, original)


if __name__ == "__main__":
    explore_stochastic_matrix()

