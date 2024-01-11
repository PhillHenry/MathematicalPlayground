import numpy             as np
import matplotlib.pyplot as plt
import matplotlib

from hilbert import decode

num_dims = 2

def draw_curve(ax, num_bits):

  # The maximum Hilbert integer.
  max_h = 2**(num_bits*num_dims)

  # Generate a sequence of Hilbert integers.
  hilberts = np.arange(max_h)
  hilberts = sorted(np.random.choice(hilberts, max_h//2, replace=False))

  # Compute the 2-dimensional locations.
  locs = decode(hilberts, num_dims, num_bits)
  for loc, point in zip(locs, hilberts):
    print("%20s -> %d" % (loc, point))

  # Choose pretty colors.
  cmap = matplotlib.cm.get_cmap('rainbow')

  # Draw. This may be a little slow.
  length = len(hilberts)
  for ii in range(length-1):
    ax.plot([locs[ii, 0], locs[ii+1, 0]],
            [locs[ii, 1], locs[ii+1, 1]],
            '-', color=cmap(((ii * 5)//length) * 0.2))
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
plt.savefig('/home/henryp/Pictures/example_2d_color.png', bbox_inches='tight')
plt.show()
