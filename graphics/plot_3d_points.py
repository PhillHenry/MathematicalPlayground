import matplotlib.pyplot as plt
import sys as sys
from pathlib import Path


def parse_line(line: str) -> (int, int, int):
    pts = line.split(",")
    return int(pts[0].strip()), int(pts[1].strip()), int(pts[2].strip())


if __name__ == "__main__":
    fig = plt.figure()
    ax = plt.axes(projection="3d")

    colours = ["green", "red", "blue", "grey"]
    for file, colour in zip(sys.argv[1:], colours):
        with open(file, "r") as f:
            xs = []
            ys = []
            zs = []
            print(f"Opening {file}")
            lines = f.readlines()
            for line in [x for x in lines if x.strip() != ""]:
                x, y, z = parse_line(line)
                print(f"{x}, {y}, {z}")
                xs.append(x)
                ys.append(y)
                zs.append(z)
            ax.scatter3D(xs, ys, zs, color=colour, s=1)
            print(f"{min(xs)},{max(xs)}")
            print(f"{min(ys)},{max(ys)}")
            print(f"{min(zs)},{max(zs)}")

    ax.set_xlabel('c1')
    ax.set_ylabel('c2')
    ax.set_zlabel('c3')
    plt.title("Z-Ordering of 4 files")
    plt.legend(["file1", "file2", "file3", "file4"], loc="lower right")
    ax.view_init(elev=10., azim=45)
    plt.savefig(f"{Path.home()}/Pictures/z_ordering.png", bbox_inches='tight', dpi=100)

    plt.show()
