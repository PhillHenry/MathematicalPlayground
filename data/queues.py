import numpy as np
import random as random


class Board:

    def __init__(self, n_states, n_counters):
        self.board = np.zeros((n_states, n_states), dtype=int)
        self.n_counters = n_counters
        self.initial_dispositions()

    def initial_dispositions(self):
        self.board[0][0] = self.n_counters

    def next_move(self):
        width = self.board.shape[0]
        height = self.board.shape[1]
        for x in range(width):
            for y in range(height):
                for n in range(self.board[x][y]):
                    if self.heads():
                        count = self.board[x][y]
                        if self.heads() and x < width - 1:
                            self.board[x + 1][y] = self.board[x + 1][y] + 1
                            self.board[x][y] = count - 1
                        elif self.heads() and y < height - 1:
                            self.board[x][y + 1] = self.board[x][y + 1] + 1
                            self.board[x][y] = count - 1
        return self

    def heads(self):
        return random.random() > 0.5


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    np.set_printoptions(precision=3)
    np.set_printoptions(suppress=True)
    board = Board(5, 20)
    n_plots = 5
    fig, plots = plt.subplots(n_plots)
    for plot in plots:
        m = board.next_move().board
        eigen_vals, eigen_vecs_as_columns = np.linalg.eig(m)
        print(f"eigen values = {eigen_vals}\nEigenVectors:\n{eigen_vecs_as_columns}")
        plot.imshow(m, cmap="hot", interpolation='nearest')
        plot.set_yticklabels([])
        plot.set_xticklabels([])
    plt.show()
