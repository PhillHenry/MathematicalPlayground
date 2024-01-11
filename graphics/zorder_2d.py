import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sorting.zorder import interleave
from utils.rolling_stats import print_stats_for

def draw_curve(ax, num_bits):
  z_to_loc = {}
  for i in range(2 ** num_bits):
    for j in range(2 ** num_bits):
      z = interleave(i, j)
      z_to_loc[z] = [i, j]

  indices = sorted(np.random.choice(list(z_to_loc.keys()), len(z_to_loc)//2, replace=False))
  locs = [z_to_loc[z] for z in sorted(indices)]

  # Choose pretty colors.
  cmap = matplotlib.cm.get_cmap('rainbow')

  # Draw. This may be a little slow.
  length = len(locs)
  for ii in range(length-1):
    x1 = locs[ii][0]
    x2 = locs[ii + 1][0]
    y1 = locs[ii][1]
    y2 = locs[ii + 1][1]
    ax.plot([x1, x2],
            [y1, y2],
            '-', color=cmap(((ii * 5)//length) * 0.2))
  print_stats_for(locs)
  for x, y in locs:
    plt.plot(x, y, "bx")
  for i in range(int(2 ** num_bits)):
    for j in range(int(2 ** num_bits)):
      plt.plot(i, j, "y+")
  ax.set_aspect('equal')
  ax.set_title("Sparse Z-Ordering")
  ax.set_xlabel('dimension 1')
  ax.set_ylabel('dimension 2')

# see https://github.com/PrincetonLIPS/numpy-hilbert-curve/blob/main/README.md
draw_curve(plt.axes(), 5)
plt.savefig('/home/henryp/Pictures/zorder_dense_2d.png', bbox_inches='tight')
plt.show()
