from sklearn import linear_model
from sklearn.linear_model._coordinate_descent import Lasso
import numpy as np
from models.lin_reg_utils import train, check, check_row, test
from data.one_hot_encodings import make_fake_1hot_encodings, make_y, make_target, drop_last
import random


def p_values(intercept_, coef_, n, X, y):
    '''
    From https://stackoverflow.com/questions/27928275/find-p-value-significance-in-scikit-learn-linearregression
    '''
    beta_hat = [intercept_] + coef_.tolist()

    # compute the p-values
    from scipy.stats import t
    # add ones column
    X1 = np.column_stack((np.ones(n), X))
    # standard deviation of the noise.
    sigma_hat = np.sqrt(np.sum(np.square(y - X1@beta_hat)) / (n - X1.shape[1]))
    # estimate the covariance matrix for beta
    beta_cov = np.linalg.inv(X1.T@X1)
    # the t-test statistic for each variable from the formula from above figure
    t_vals = beta_hat / (sigma_hat * np.sqrt(np.diagonal(beta_cov)))
    # compute 2-sided p-values.
    p_vals = t.sf(np.abs(t_vals), n-X1.shape[1])*2
    print(f"t-values = {t_vals}")
    print(f"p-values = {p_vals}")


def train_and_check(x, ys, n_rows, model):
    n_train = int(n_rows * 0.8)
    m, coeffs, intercept = train(model, n_train, x, ys)
    error, r2 = test(m, n_train, x, ys)
    # p_values(intercept, coeffs, n_rows, x, ys)  # causes numpy.linalg.LinAlgError: Singular matrix
    return error, coeffs, r2


def deltas(xs):
    return [b - a for a, b in zip(xs, xs[1:])]


def skip_every(n, xs):
    return [x for i, x in enumerate(xs) if (i + 1) % n != 0]


def num_non_increasing(xs, skip):
    ds = deltas(xs)
    skipped = skip_every(skip, ds)
    return len([x for x in skipped if x < 0])


def compare_1hot_vs_dummy(model):
    n_rows = 1000
    n_categories = 4
    n_cardinality = 5
    m = make_fake_1hot_encodings(drop_last=False, n_rows=n_rows, n_categories=n_categories, n_cardinality=n_cardinality)
    m_dropped = drop_last(m, n_categories, n_cardinality)
    for noise in [0, 10, 100, 1000]:
        ys = make_y(m, error=noise)
        print("+" * 50)
        print("Noise level = %.4f," % noise)
        print("One hot encoding")
        test_error, m_coeffs, r2 = train_and_check(m, ys, n_rows, model)
        print('\tMean squared error: %.4f' % test_error)
        print('\tCoefficient of determination: %.2f' % r2)
        print("Dummy variable encoding")
        dummy_error, m_dropped_coeffs, r2 = train_and_check(m_dropped, ys, n_rows, model)
        print('\tMean squared error: %.4f' % dummy_error)
        print('\tCoefficient of determination: %.2f' % r2)
        delta_error = test_error - dummy_error
        print("Difference in error %.4f" % delta_error)
        print(f"non increasing coefficients: {num_non_increasing(m_coeffs, n_cardinality)}")
        print(f"non increasing dropped coefficients: {num_non_increasing(m_dropped_coeffs, n_cardinality - 1)}")
        print("-" * 50)
        print("")


def compare_one_hot_to_dummy(noise):
    n_rows = 1000
    n_categories = 4
    n_cardinality = 20
    m = make_fake_1hot_encodings(drop_last=False, n_rows=n_rows, n_categories=n_categories, n_cardinality=n_cardinality)
    m_dropped = drop_last(m, n_categories, n_cardinality)
    ys = make_y(m, error=noise)
    model = linear_model.LinearRegression(fit_intercept=True)
    one_hot_error, _, _ = train_and_check(m, ys, n_rows, model)
    dummy_error, _, _ = train_and_check(m_dropped, ys, n_rows, model)
    return one_hot_error, dummy_error


def hypothesis_test(xs: list, ys: list):
    l = len(xs)
    n_samples = l * 10
    assert l == len(ys)
    results = []
    for _ in range(n_samples):
        x = xs[random.randint(0, l - 1)]
        y = ys[random.randint(0, l - 1)]
        results.append(x - y)
    return results


def compare_errors(noise=100):
    num_trials = 1000
    results = []
    one_hot_errors = []
    dummy_errors = []
    for _ in range(num_trials):
        one_hot_error, dummy_error = compare_one_hot_to_dummy(noise)
        results.append(one_hot_error - dummy_error)
        one_hot_errors.append(one_hot_error)
        dummy_errors.append(dummy_error)
    print("num trials = %d, mean = %.4f, std_dev = %.4f" % (num_trials, np.mean(results), np.std(results)))
    results = hypothesis_test(one_hot_errors, dummy_errors)
    p_value = len([x for x in results if x > 0]) / len(results)
    print(f"p-value dummy variables better than one-hot encoding: {p_value}")


if __name__ == "__main__":
    np.set_printoptions(precision=3)

    print("\n========== Linear Regression ===========\n")
    for intercept in [True, False]:
        print("Intercept = %s" % intercept)
        model = linear_model.LinearRegression(fit_intercept=intercept)
        compare_1hot_vs_dummy(model)

    print("\n========== Lasso ===========\n")
    compare_1hot_vs_dummy(Lasso())

    #compare_errors(noise=10)  # expensive and shows no significant difference between 1-hot and dummy vars

