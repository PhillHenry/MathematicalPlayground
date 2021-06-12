import numpy as np
from sklearn import linear_model
from models.lin_reg_utils import alpha_intercepts, no_drop_last


if __name__ == "__main__":
    np.set_printoptions(precision=3)
    for intercept in [True, False]:
        for drop in [True, False]:
            print(f"LinearRegression: fit_interecpt = {intercept}, drop_last = {drop}")
            no_drop_last(linear_model.LinearRegression(fit_intercept=intercept), drop_last=drop)
    alpha_intercepts(linear_model.Ridge)
    alpha_intercepts(linear_model.Lasso)
