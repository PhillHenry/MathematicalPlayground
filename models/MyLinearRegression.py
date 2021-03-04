import sys
from sklearn.linear_model import LinearRegression


def parse_line(xs):
    elements = []
    for x in xs:
        pos, val = x.split(":")
        elements.append(float(val))
    return elements


def load_svm_file(file):
    print("Loading {}".format(file))
    actual = []
    obs = []
    with open(file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            elements = line.strip().split(" ")
            actual.append(float(elements[0]))
            obs.append(parse_line(elements[1:]))
        f.close()
    return obs, actual


if __name__ == "__main__":
    file = sys.argv[1]
    xs, ys = load_svm_file(file)
    assert(len(xs) == len(ys))
    print("Loaded {} lines with actual values that look like {} and observations like:\n{}"
          .format(len(xs), ys[0], xs[0]))
    reg = LinearRegression().fit(xs, ys)
    print(reg.score(xs, ys))







