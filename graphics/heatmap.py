import matplotlib.pyplot as plt


def add_heatmap(m, title):
    heatmap = plt.imshow(m, cmap='hot', interpolation='nearest')
    plt.colorbar(heatmap)
    plt.title(title)