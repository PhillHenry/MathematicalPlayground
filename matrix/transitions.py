import numpy as np
from data.queues import Board


def print_sorted_eigens(eigens):
    sorted_vals, sorted_vecs = zip(*eigens)
    print(f"Eigen values = {sorted_vals}")
    # print(f"Eigen Vectors:")
    # for vec in sorted_vecs:
    #     print(vec)


def play_game() -> Board:
    n_counters = 40
    board = Board(5, n_counters, n_counters//20)
    for _ in range(1000):
        board = board.next_move()
    return board


def transition_eigens(board: Board):
    print("\nTransitions:")
    eigen_vals, eigen_vecs_as_columns = np.linalg.eig(board.probability_matrix())
    eigens = sorted(zip(eigen_vals, eigen_vecs_as_columns), key=lambda x: -x[0])
    print_sorted_eigens(eigens)


if __name__ == "__main__":
    np.set_printoptions(precision=3)
    np.set_printoptions(suppress=True)
    board = play_game()
    transition_eigens(board)
