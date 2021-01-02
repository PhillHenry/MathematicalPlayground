import pytest
import distributions.bootstrap as to_test

xs = list(range(1000))


def test_confidence_interval():
    first, last = to_test.confidence_interval_of(xs, 90)
    assert first == min(xs) + 50
    assert last == max(xs) - 50
    print('first = %d, last = %d'.format(first, last))

