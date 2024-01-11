import math


def update_min(results: dict, key: int, *xs):
    for x in xs:
        results[key] = min(x, results.get(key, math.inf))


def update_max(results: dict, key: int, *xs):
    for x in xs:
        results[key] = max(x, results.get(key, -math.inf))


def print_stats_for(locs: [], num_files=5):
    distances = []
    length = len(locs)
    min_x = {}
    min_y = {}
    max_x = {}
    max_y = {}
    for ii in range(length - 1):
        # Note the hilbert library returns *unsigned* ints so turn them to normal ints so we can calculate distance
        x1 = locs[ii][0]
        x2 = locs[ii + 1][0]
        y1 = locs[ii][1]
        y2 = locs[ii + 1][1]
        file_index = (ii * num_files) // length
        d = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
        distances.append(d)
        update_min(min_x, file_index, x1, x2)
        update_min(min_y, file_index, y1, y2)
        update_max(max_x, file_index, x1, x2)
        update_max(max_y, file_index, y1, y2)

    print(f"Average steps between adjacent points: {sum(distances)/len(distances)}")
    spread_areas = []
    for i in range(num_files):
        spread_x = max_x[i] - min_x[i]
        spread_y = max_y[i] - min_y[i]
        print(f"file {i}: spread_x = {spread_x}, spread_y = {spread_y}")
        spread_areas.append(spread_x * spread_y)
    print(f"total area = {sum(spread_areas)}")

