from data.queues import Board


def test_counters_reach_end():
    n_states = 5
    n_counters = 20
    board = Board(n_states, n_counters)
    for _ in range(1000):
        board = board.next_move()
    assert board.board[n_states - 1][n_states - 1] == n_counters
