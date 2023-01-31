import numpy as np


def harris(T_field, kappa):

    H_field = np.zeros((np.shape(T_field)[0], np.shape(T_field)[1]))
    for row in range(np.shape(T_field)[0]):
        for col in range(np.shape(T_field)[1]):
            T11 = T_field[row, col, 0]
            T12 = T_field[row, col, 1]
            T22 = T_field[row, col, 2]
            # CH = detT - k(traceT)^2
            detT = T11 * T22 - T12 * T12
            H_field[row, col] = detT - kappa * (T11 + T22)**2

    return H_field
