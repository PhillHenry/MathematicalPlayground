from graph import transition_matrix, generate_neighbours_list, metropolis
import numpy as np

# "The Metropolis- Hastings algorithm finds a transition matrix with the given
# limiting vector"
# http://web2.uwindsor.ca/math/hlynka/MHObservations.pdf


if __name__ == "__main__":
    n = 6
    matrix = transition_matrix(n)
    neighbours = generate_neighbours_list(matrix)
    weights = [1] * n
    m = np.zeros([n, n])
    for i in range(n):
        xs = neighbours[i]
        for j in xs:
            m[j][i] += 1/n
    print(m)
    print(metropolis(0, neighbours, weights, m))
