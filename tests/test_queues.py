from data.queues import Board
import numpy as np


class BoardIgnoringEndStates(Board):
    def handle_end_states(self, height, width):
        pass


def total_number_of_counters_on_board(board: Board) -> int:
    return np.sum(board.board, axis=(0, 1))


def check_total_probabilities(board: Board):
    m = board.probability_matrix()
    n_cols = m.shape[1]
    totals = np.zeros([n_cols])
    for i in range(m.shape[0]):
        row = m[i, :]
        totals += row
    expected = np.zeros(n_cols)
    expected[0:] = 1.
    print()
    print(expected)
    print(totals)
    assert np.isclose(totals, expected).all()


def test_counters_reach_end():
    n_states = 5
    n_counters = 20
    board = BoardIgnoringEndStates(n_states, n_counters)
    board = run_game(board)
    assert board.board[n_states - 1][n_states - 1] == n_counters


def test_probability_matrix():
    n_states = 5
    n_counters = 50
    board = Board(n_states, n_counters)
    board = run_game(board)
    check_total_probabilities(board)


def run_game(board):
    for _ in range(1000):
        board = board.next_move()
        assert total_number_of_counters_on_board(board) == board.n_counters
    return board

