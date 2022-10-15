import numpy as np
import random as random


class Board:

    def __init__(self, board_size, n_counters, n_max=None):
        self.board_size = board_size
        self.n_counters = n_counters
        if n_max is None:
            self.n_max = n_counters
        else:
            self.n_max = n_max
        self.initial_dispositions()

    def initial_dispositions(self):
        self.board = np.zeros((self.board_size, self.board_size), dtype=int)
        self.board[0][0] = self.n_counters
        self.transitions = np.zeros((self.board_size ** 2, self.board_size ** 2), dtype=int)

    def next_move(self):
        width = self.board.shape[0]
        height = self.board.shape[1]
        for y in range(width):
            for x in range(height):
                for n in range(self.board[y][x]):
                    if self.heads():
                        count = self.board[y][x]
                        new_x = y
                        new_y = x
                        if self.heads() and y < width - 1 and self.board[y + 1][x] <= self.n_max:
                            new_x = y + 1
                            self.board[new_x][x] = self.board[new_x][x] + 1
                            self.board[y][x] = count - 1
                        elif self.heads() and x < height - 1 and self.board[y][x + 1] <= self.n_max:
                            new_y = x + 1
                            self.board[y][new_y] = self.board[y][new_y] + 1
                            self.board[y][x] = count - 1
                        old_index = (width * y) + x
                        new_index = (width * new_x) + new_y
                        self.transitions[old_index][new_index] = self.transitions[old_index][
                                                                     new_index] + 1
        self.handle_end_states(height, width)
        return self

    def handle_end_states(self, height, width):
        num_end = self.board[width - 1][height - 1]
        if num_end > 0:
            self.board[width - 1][height - 1] = num_end - 1
            self.board[0][0] = self.board[0][0] + 1

    def heads(self):
        return random.random() > 0.5

    def probability_matrix(self):
        totals = self.transitions.sum(axis=0)
        return self.transitions / totals


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    np.set_printoptions(precision=3)
    np.set_printoptions(suppress=True)
    n_counters = 20
    board = Board(10, n_counters)

    fig, ax = plt.subplots()
    ln = ax.imshow(board.board, cmap="hot", interpolation='nearest')


    def init():
        print("init")
        board.initial_dispositions()
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        return ln,


    def update(frame):
        m = board.next_move().board
        eigen_vals, eigen_vecs_as_columns = np.linalg.eig(m)
        print(f"{frame} eigen values = {np.sort(eigen_vals)}")
        # print("EigenVectors:\n{eigen_vecs_as_columns}")
        ln = ax.imshow(m, cmap="hot", interpolation='nearest', vmin=0, vmax=n_counters // 4)
        return ln,


    ani = FuncAnimation(fig, update, frames=200, init_func=init, blit=True)

    plt.show()

    print("\nTransitions:")
    eigen_vals, eigen_vecs_as_columns = np.linalg.eig(board.transitions)
    print(f"Eigen values = {np.sort(eigen_vals)}")
    print(f"Eigen Vectors:\n{eigen_vecs_as_columns}")
