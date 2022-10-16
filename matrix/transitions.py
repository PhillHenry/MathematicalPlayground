import numpy as np
from data.queues import Board


def print_sorted_eigens(eigens):
    sorted_vals, sorted_vecs = zip(*eigens)
    print(f"Unique eigen values = {set(sorted_vals)}")
    # print(f"Eigen Vectors:")
    # for vec in sorted_vecs:
    #     print(vec)


def play_game() -> Board:
    n_counters = 800
    board = Board(20, n_counters, 4)
    for _ in range(10000):
        board = board.next_move()
    return board


def transition_eigens(board: Board):
    print("\nTransitions:")
    m = board.laplacian()
    eigen_vals, eigen_vecs_as_columns = np.linalg.eig(m)
    eigens = sorted(zip(eigen_vals, eigen_vecs_as_columns), key=lambda x: -x[0])
    print_sorted_eigens(eigens)


if __name__ == "__main__":
    np.set_printoptions(precision=4)
    np.set_printoptions(suppress=True)
    board = play_game()
    transition_eigens(board)
