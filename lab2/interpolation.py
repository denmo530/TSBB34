import scipy
import numpy as np


def interpolation(img, dtot):
    # creates interpolated object for in_im over area x:x+width, y:y+height
    x = 0
    y = 0
    [width, height] = [np.shape(img
                                )[1], np.shape(img)[0]]
    x_coords = np.arange(x, x+width)
    y_coords = np.arange(y, y+height)

    img_interpol = scipy.interpolate.RectBivariateSpline(
        y_coords, x_coords, img[y:y+height, x:x+width])

    img_v = img_interpol(
        np.arange(dtot[1], height + dtot[1]), np.arange(dtot[0], width + dtot[0]))

    return img_v
