
def parse_line(xs):
    elements = []
    for x in xs:
        pos, val = x.split(":")
        elements.append(float(val))
    return elements


# Apparently, sklearn.datasets.load_svmlight_file will do this
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
