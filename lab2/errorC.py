import numpy as np
from interpolation import interpolation


def errorC(I, J, J_org, d):

    Jv = interpolation(J, d)

    # Calculates the error
    error = np.linalg.norm(J_org-I)
    diff = np.linalg.norm(Jv-I)

    return error, diff, Jv
