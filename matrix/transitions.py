import numpy as np
from data.queues import Board


if __name__ == "__main__":
    np.set_printoptions(precision=3)
    np.set_printoptions(suppress=True)
    n_counters = 20
    board = Board(5, n_counters)

    for _ in range(150):
        board = board.next_move()

    print("\nTransitions:")
    eigen_vals, eigen_vecs_as_columns = np.linalg.eig(board.transitions)
    print(f"Eigen values = {np.sort(eigen_vals)}")
    print(f"Eigen Vectors:\n{eigen_vecs_as_columns}")