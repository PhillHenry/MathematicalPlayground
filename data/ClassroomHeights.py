import numpy as np
import random as random


class ClassroomHeights:

    def __init__(self, num_measurements, num_students):
        rows = []
        for i in range(num_students):
            measurements = np.random.normal(random.gauss(150, 10), 5, num_measurements)
            rows.append(measurements)
        self.m = np.asmatrix(np.stack(rows))

