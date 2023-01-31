import numpy as np
from matplotlib import pyplot as plt
import math


def estimate_T(g_x, g_y, x, y, window_size):
    T = np.zeros((2, 2))

    # define the window
    x = math.floor(x - window_size[1] / 2)
    y = math.floor(y - window_size[0] / 2)
    x1 = x + window_size[1]
    y1 = y + window_size[0]

    dx = g_x[y:y1, x:x1]
    dy = g_y[y:y1, x:x1]

    T[0, 0] = np.sum(dx*dx)
    T[1, 0] = np.sum(dx*dy)
    T[0, 1] = T[1, 0]
    T[1, 1] = np.sum(dy*dy)

    return T
