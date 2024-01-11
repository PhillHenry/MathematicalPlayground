import numpy             as np
import matplotlib.pyplot as plt
import matplotlib

from hilbert import decode
from utils.rolling_stats import print_stats_for

num_dims = 2

def draw_curve(ax, num_bits):

  # The maximum Hilbert integer.
  max_h = 2**(num_bits*num_dims)

  # Generate a sequence of Hilbert integers.
  hilberts = np.arange(max_h)
  hilberts = sorted(np.random.choice(hilberts, max_h//2, replace=False))

  # Compute the 2-dimensional locations.
  locs = decode(hilberts, num_dims, num_bits)

  # Choose pretty colors.
  cmap = matplotlib.cm.get_cmap('rainbow')

  # Draw. This may be a little slow.
  length = len(hilberts)
  for ii in range(length-1):
    # Note the hilbert library returns *unsigned* ints so turn them to normal ints so we can calculate distance
    x1 = int(locs[ii, 0])
    x2 = int(locs[ii + 1, 0])
    y1 = int(locs[ii, 1])
    y2 = int(locs[ii + 1, 1])
    file_index = (ii * 5) // length
    ax.plot([x1, x2],
            [y1, y2],
            '-', color=cmap(file_index * 0.2))
    d = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
  print_stats_for(list(map(lambda loc: [int(loc[0]), int(loc[1])], locs)))
  for x, y in locs:
    plt.plot(x, y, "bx")
  for i in range(int(2 ** num_bits)):
    for j in range(int(2 ** num_bits)):
      plt.plot(i, j, "y+")
  ax.set_aspect('equal')
  ax.set_title("Sparse ")
  ax.set_xlabel('dimension 1')
  ax.set_ylabel('dimension 2')

# see https://github.com/PrincetonLIPS/numpy-hilbert-curve/blob/main/README.md
draw_curve(plt.axes(), 5)
plt.savefig('/home/henryp/Pictures/hilbert_2d.png', bbox_inches='tight')
plt.show()
