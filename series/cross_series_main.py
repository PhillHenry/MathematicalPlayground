import pandas as pd
import numpy as np
from scipy import signal
from series.cross_series_correlation import print_sample


if __name__ == '__main__':
    inpatients = pd.read_csv("/home/henryp/Documents/CandF/sample_RD130_ecds_oct_nov_21.tsv", sep="\t")
    ecds = pd.read_csv("/home/henryp/Documents/CandF/sample_RD130_inpatient_oct_nov_21.tsv", sep="\t")
    x = inpatients.running_total
    y = ecds.running_total

    correlation = signal.correlate(x, y, mode="full")
    print_sample(correlation)
    max_correlation = np.argmax(correlation)
    print(f"max_correlation = {max_correlation}")

    lag_indices = signal.correlation_lags(x.size, y.size, mode="full")
    print_sample(lag_indices)

    lag = lag_indices[max_correlation]
    print(f"lag = {lag}")
