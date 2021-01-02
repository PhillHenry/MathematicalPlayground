import pytest
import distributions.bootstrap as to_test

xs = list(range(40))


def test_confidence_interval():
    first, last = to_test.confidence_interval_of(xs, 10)
    assert first == min(xs) + 2
    assert last == max(xs) - 2
    print('first = %d, last = %d'.format(first, last))

