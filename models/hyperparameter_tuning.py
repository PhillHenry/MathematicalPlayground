from hyperopt import hp
from data_loader import load_svm_file
import sys
from hyperopt import fmin, tpe, space_eval
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import ElasticNet
import numpy as np


class LGTuner(object):
    def __init__(self, xs, ys):
        self.xs = xs
        self.ys = ys

    def objective(self, args):
        l1_ratio, alpha = args
        # print("alpha = {}".format(alpha))
        reg = ElasticNet(alpha=alpha, l1_ratio=l1_ratio).fit(xs, ys)
        return reg.score(xs, ys)

    def tune(self):
        space = hp.choice('a',
                          [
                              (hp.uniform('l1_ratio', 0, 1.0), hp.loguniform('alpha', np.log(0.01), np.log(1.0)))
                          ])
        best = fmin(self.objective, space, algo=tpe.suggest, max_evals=100)
        return best


if __name__ == "__main__":
    """
    Run against SVM data such as:
    /home/henryp/Code/Scala/MyCode/spark/data/mllib/sample_linear_regression_data.txt
    """
    file = sys.argv[1]
    xs, ys = load_svm_file(file)
    tuner = LGTuner(xs, ys)
    print(tuner.tune())
