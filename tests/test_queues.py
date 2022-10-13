from data.queues import Board
import numpy as np


class BoardIgnoringEndStates(Board):
    def handle_end_states(self, height, width):
        pass


def total_number_of_counters_on_board(board: Board) -> int:
    return np.sum(board.board, axis=(0, 1))


def test_counters_reach_end():
    n_states = 5
    n_counters = 20
    board = BoardIgnoringEndStates(n_states, n_counters)
    for _ in range(1000):
        board = board.next_move()
        assert total_number_of_counters_on_board(board) == n_counters
    assert board.board[n_states - 1][n_states - 1] == n_counters
