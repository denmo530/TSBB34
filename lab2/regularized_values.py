import numpy as np
from scipy.signal import fftconvolve

# Get gradients for J


def regularized_values(J, ksize, sigma):
    # gets the gradient images
    lp = np.atleast_2d(
        np.exp(-0.5 * (np.arange(-ksize, ksize + 1) / sigma) ** 2))
    lp = lp / np.sum(lp)  # normalize the filter
    df = np.atleast_2d(-1.0 / np.square(sigma) *
                       np.arange(-ksize, ksize + 1) * lp)

    g_x = fftconvolve(J, df, mode="same")
    g_y = fftconvolve(J, df.T, mode="same")
    return g_x, g_y, lp
