import numpy as np


np.set_printoptions(precision=3)
# np.set_printoptions(suppress=True)


def transitions(n=5):
    a = np.random.rand(n, n)
    max_wait = 3
    for i in range(max_wait):
        for j in range(max_wait):
            if j <= i:
                a[i][j] = 0.
    row_sums = a.sum(axis=1)
    return a / row_sums[:, np.newaxis]


if __name__ == "__main__":
    x = transitions()

    print(x)

    eigen_vals, eigen_vecs = np.linalg.eig(x)
    print(eigen_vecs)
    print(eigen_vals)

    for _ in range(30):
        x = np.dot(x, x)

    print(x)

    n = len(eigen_vecs)
    m = np.zeros([n, n])
    m[:, 0] = eigen_vecs[0]
    Q = eigen_vecs
    lambda_inf = np.zeros([n, n])
    lambda_inf[0][0] = 1.
    print(np.dot(Q, np.dot(lambda_inf, np.linalg.inv(Q))))
