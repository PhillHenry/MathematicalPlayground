from scipy import signal
import numpy as np
from numpy.random import default_rng


real_lag = 100


def print_sample(xs: list):
    first = xs[:10]
    last = xs[-10:]
    mid_start = real_lag - 5
    mid_end = real_lag + 5
    mid = xs[mid_start:mid_end]
    print()
    print(first)
    print(mid)
    print(last)


rng = default_rng()
x = rng.standard_normal(1000)
noise = rng.standard_normal(1000)
y = np.concatenate([rng.standard_normal(real_lag), x + noise])

correlation = signal.correlate(x, y, mode="full")
print_sample(correlation)
max_correlation = np.argmax(correlation)
print(f"max_correlation = {max_correlation}")

lag_indices = signal.correlation_lags(x.size, y.size, mode="full")
print_sample(lag_indices)

lag = lag_indices[max_correlation]
print(f"lag = {lag}")
assert lag == -real_lag
