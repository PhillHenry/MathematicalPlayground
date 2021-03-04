import sys
from sklearn.linear_model import LinearRegression
from data_loader import load_svm_file

if __name__ == "__main__":
    file = sys.argv[1]
    xs, ys = load_svm_file(file)
    assert(len(xs) == len(ys))
    print("Loaded {} lines with actual values that look like {} and observations like:\n{}"
          .format(len(xs), ys[0], xs[0]))
    reg = LinearRegression().fit(xs, ys)
    print(reg.score(xs, ys))







