from LK_equation import LK_equation
import numpy as np
from interpolation import interpolation


def LK_equation_multi(I, J, x, y, window_size, max_iterations, min_error):
    d_tot = np.zeros((2, 1))
    Jnew = J
    g_x = J
    g_y = J

    for n in range(max_iterations, 0, -1):

        # sc = 2 ** (n-1)
        d, Jg_interpol, Jgdx_interpol, Jgdy_interpol = LK_equation(
            I, Jnew, g_x, g_y, x, y, window_size, max_iterations, min_error)

        d_tot += d
        J = Jg_interpol
        g_x = Jgdx_interpol
        g_y = Jgdy_interpol

    return d_tot
