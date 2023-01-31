import math
import numpy as np


def estimate_e(I, J, g_x, g_y, x, y, window_size):
    e = np.zeros((2, 1))

    # window
    x = math.floor(x - window_size[1] / 2)
    y = math.floor(y - window_size[0] / 2)
    x1 = x + window_size[1]
    y1 = y + window_size[0]

    I_reg = I[y:y1, x:x1]
    J_reg = J[y:y1, x:x1]
    g_x_reg = g_x[y:y1, x:x1]
    g_y_reg = g_y[y:y1, x:x1]

    e[0] = np.sum(np.multiply(I_reg-J_reg, g_x_reg))
    e[1] = np.sum(np.multiply(I_reg-J_reg, g_y_reg))
    return e
