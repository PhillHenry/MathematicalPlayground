from lin_reg_1hot_vs_dummy import compare_1hot_vs_dummy
from sklearn import linear_model
from sklearn.linear_model._coordinate_descent import Lasso
import matplotlib.pyplot as plt
import math as math


def add_pvalue_bar_chart(p_values, ax, color, offset):
    xs = []
    indices = []
    for i, x in enumerate(p_values):
        if x != 0 and i != 0:  # ignore the intercept p-value
            xs.append(-math.log(x))
            indices.append(i + offset)
    rects1 = ax.barh(indices, xs, color=color, height=0.5)
    ax.set_xlabel('-log(p-value)')
    ax.set_title(f"noise = {noise}")
    # ax.set_xticks(indices)


if __name__ == "__main__":
    '''
    When do we ignore the intercept?
    "The shortest answer: never, unless you are sure that your linear approximation of the data 
    generating process (linear regression model) either by some theoretical or any other reasons is 
    forced to go through the origin."
    https://stats.stackexchange.com/questions/7948/when-is-it-ok-to-remove-the-intercept-in-a-linear-regression-model
    '''
    model = Lasso()  #linear_model.LinearRegression(fit_intercept=True)
    noise_to_results = compare_1hot_vs_dummy(model, [8,16,32,64,128,256,512])
    noisy_keys = [x for x in noise_to_results.keys() if x != 0]
    n = len(noisy_keys)
    colours = ["red", "blue", "green"]
    fig, ax = plt.subplots()
    for index, noise in enumerate(sorted(noisy_keys)):
        # ax = plt.subplot((n * 100) + 10 + 1)
        one_hot_vs_dummy = noise_to_results[noise]
        add_pvalue_bar_chart(one_hot_vs_dummy.dummy_results.p_values, ax, None, index * 0.1)
    plt.show()
