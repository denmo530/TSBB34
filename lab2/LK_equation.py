import numpy as np
from regularized_values import regularized_values
from estimate_T import estimate_T
from estimate_e import estimate_e
from interpolation import interpolation


def LK_equation(I, J, x, y, window_size, iterations):
    d_tot = np.zeros((2, 1))

    g_x, g_y, _ = regularized_values(J, 10, 1)

    J_org = J
    g_x_org = g_x
    g_y_org = g_y

    i = 0
    while i < iterations:
        T = estimate_T(g_x, g_y, x, y, window_size)
        e = estimate_e(I, J, g_x, g_y, x, y, window_size)
        d = np.linalg.solve(T, e)
        d_tot += d

        J = interpolation(J_org, d_tot)
        g_x = interpolation(g_x_org, d_tot)
        g_y = interpolation(g_y_org, d_tot)
        i += 1

    return d_tot
