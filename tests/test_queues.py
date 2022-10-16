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


def check_laplacian_invariants(board: Board):
    m = board.laplacian()
    height, width = m.shape
    for i in range(height):
        for j in range(width):
            element = m[i][j]
            if i != j:
                assert element == 0 or element == -1
            else:
                assert element >= 0


def test_laplacian():
    board = create_board()
    check_laplacian_invariants(board)


def test_probability_matrix_of_zeros():
    board = create_board()
    p = board.probability_matrix()
    assert p.all() == False


def test_degrees():
    d = board_after_transitions().degrees()
    for i in range(d.shape[0]):
        for j in range(d.shape[1]):
            if i != j:
                assert d[i][j] == 0
            else:
                assert d[i][j] > 0


def test_adjacency_matrix():
    a = board_after_transitions().adjacency_matrix()
    elements = [x for xs in a.tolist() for x in xs]
    assert set(elements) == set([1, 0])


def test_counters_reach_end():
    board_size = 5
    n_counters = 20
    board = BoardIgnoringEndStates(board_size, n_counters)
    board = run_game(board)
    assert board.board[board_size - 1][board_size - 1] == n_counters


def test_probability_matrix():
    board = board_after_transitions()
    check_total_probabilities(board)


def board_after_transitions():
    board = create_board()
    board = run_game(board)
    return board


def create_board():
    board_size = 5
    n_counters = 50
    board = Board(board_size, n_counters)
    return board


def run_game(board):
    for _ in range(1000):
        board = board.next_move()
        assert total_number_of_counters_on_board(board) == board.n_counters
    return board

