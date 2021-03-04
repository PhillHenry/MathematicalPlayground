from hyperopt import hp
from data_loader import load_svm_file
import sys
from hyperopt import fmin, tpe, space_eval
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import ElasticNet


class LGTuner(object):
    def __init__(self, xs, ys):
        self.xs = xs
        self.ys = ys

    def objective(self, args):
        alpha = args
        print("alpha = {}".format(alpha))
        reg = ElasticNet(alpha=alpha).fit(xs, ys)
        return reg.score(xs, ys)

    def tune(self):
        space = hp.choice('a',
                          [
                              hp.uniform('alpha', 0, 1.0)
                          ])
        best = fmin(self.objective, space, algo=tpe.suggest, max_evals=100)
        return best


if __name__ == "__main__":
    file = sys.argv[1]
    xs, ys = load_svm_file(file)
    tuner = LGTuner(xs, ys)
    print(tuner.tune())
