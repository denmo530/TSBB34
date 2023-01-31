import numpy as np
from scipy.signal import convolve2d as conv2
from scipy.signal import fftconvolve
from regularized_values import regularized_values


def orientation_tensor(J, k_size, sigma):
    # returns img with tensors in 3rd dim, T11, T12, T22 respectively

    g_x, g_y, lp = regularized_values(J, k_size, sigma)

    T_field = np.zeros((np.shape(J)[0], np.shape(J)[1], 3))
    T_field[:, :, 0] = g_x * g_x
    T_field[:, :, 1] = g_x * g_y
    T_field[:, :, 2] = g_y * g_y

    # First
    T_field[:, :, 0] = fftconvolve(T_field[:, :, 0], lp, mode='same')
    T_field[:, :, 1] = fftconvolve(T_field[:, :, 1], lp, mode='same')
    T_field[:, :, 2] = fftconvolve(T_field[:, :, 2], lp, mode='same')

    # Second
    T_field[:, :, 0] = fftconvolve(T_field[:, :, 0], lp.T, mode='same')
    T_field[:, :, 1] = fftconvolve(T_field[:, :, 1], lp.T, mode='same')
    T_field[:, :, 2] = fftconvolve(T_field[:, :, 2], lp.T, mode='same')

    return T_field
