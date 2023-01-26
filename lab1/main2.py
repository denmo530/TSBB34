import lab1
import numpy as np
from scipy.signal import convolve2d
import scipy
import matplotlib.pyplot as plt
from scipy.interpolate import RectBivariateSpline
from PIL import Image

# Implement Single scale Lucas - kanade function


def LK_equation(I, J, J_org, k_size, sigma, w, iterations):

    M = I.shape[0]
    N = I.shape[1]

    T = np.zeros((len(I), len(I[0]), 2, 2))
    e = np.zeros((len(I), len(I[0]), 2, 1))

    lowpass = np.atleast_2d(
        np.exp(-0.5*(np.arange(-(k_size//2), k_size//2+1, 1)/sigma)**2))
    lowpass = lowpass/np.sum(lowpass)
    derivative_filt = np.atleast_2d(-1.0/np.square(sigma)
                                    * np.arange(-(k_size//2), k_size//2+1, 1) * lowpass)

    # Gradients
    g_x = convolve2d(convolve2d(J, derivative_filt,
                     mode='same'), lowpass.T, mode='same')
    g_y = convolve2d(convolve2d(J, lowpass, mode='same'),
                     derivative_filt.T, mode='same')

    # 21x21 filter
    w_kern = np.ones((w, w))

    # Beräkna x², y², xy
    g_x2 = np.multiply(g_x, g_x)
    g_y2 = np.multiply(g_y, g_y)
    g_xy = np.multiply(g_x, g_y)

    # Falta alla tre för sig med neighbourhood 21x21 filter med 1or
    con_x2 = convolve2d(g_x2, w_kern, mode="same")
    con_y2 = convolve2d(g_y2, w_kern, mode="same")
    con_xy = convolve2d(g_xy, w_kern, mode="same")

    # Kombinera för T
    T = np.array([[con_x2, con_xy], [
                 con_xy, con_y2]])
    T = T.transpose((2, 3, 0, 1))
    # print(f"T: {T.shape}")

    # Beräkna e
    # [I(x) - J(x)]*gradienten J(x), där x är positonen för varje pixel
    IJ = I - J
    resx = np.multiply(IJ, g_x)
    resy = np.multiply(IJ, g_y)
    conx = convolve2d(resx, w_kern, mode="same")
    cony = convolve2d(resy, w_kern, mode="same")
    e = np.array([conx, cony])
    e = e.transpose((1, 2, 0))
    # print(f"e: {e.shape}")

    # print(T.shape)
    # print(e.shape)
    d = np.linalg.solve(T, e)
    # print(d.shape)
    # print(f"d: {d.shape}")

    if iterations > 0:
        Jnew = J + 0.5*g_x + 0.5*g_y  # J(x+dTot) by interpolating
        return LK_equation(I, Jnew, J_org, k_size, sigma, w, iterations-1)

    else:
        d = (np.linalg.solve(T, e))
        error, diff, Jv = errorC(I, J_org, d)
        print("Error: ", error)
        print("Difference: ", diff)
        return d, Jv


def errorC(I, J, d):

    # Get width and height, creates array coordinates for the image
    # print(np.ndim(J))
    [width, height] = [np.shape(I)[1], np.shape(I)[0]]
    # height = 480, width = 512
    [x_c, y_c] = [np.arange(0, width), np.arange(0, height)]

    # Creates interpolation function
    J_interpol = scipy.interpolate.RectBivariateSpline(y_c, x_c, J)

    # Reshapes into a 2D array
    mesh_grid_a = np.array(np.meshgrid(x_c, y_c)).reshape(2, -1)

    # Creates a new set of coordinates with the motion field
    Jc = np.array([mesh_grid_a[0] + d[..., 0].flatten()/2,
                   mesh_grid_a[1] + d[..., 1].flatten()/2])

    # Applies the interpolation
    Jv = J_interpol(Jc[1], Jc[0], grid=False).reshape(np.shape(J))

    # Calculates the error
    error = np.linalg.norm(J-I)
    diff = np.linalg.norm(Jv-I)

    # Shows the two images side by side
    # lab1.image_grid({"I": I, "J": J}, share_all=True,
    #                 imshow_opts={'cmap': 'gray'})
    # plt.figure("Diff"), plt.imshow(np.abs(Jv-I), cmap='gray')

    # print(f"Error: {error} \nDifference: {diff}")
    # plt.imshow(Jv)
    lab1.gopimage(d)
    plt.show()

    return error, diff, Jv


def LK_equation_multi(I, J, numScales):
    Vtot = 0
    Jn = J
    for n in range(numScales, 2, -1):
        # print('n = ', n)
        sc = 2 ** (n-1)
        # def LK_equation(I, J, J_org, k_size, sigma, w, iterations):
        V, Jv = LK_equation(I, Jn, J, k_size=sc*2,
                            sigma=sc * 0.1, w=21, iterations=0)

        Vtot += V
        Jn = Jv
        lab1.gopimage(Vtot)
        plt.show()
    return 0


if __name__ == "__main__":
    I = lab1.load_image_grayscale("./forwardL/forwardL0.png")
    J = lab1.load_image_grayscale("./forwardL/forwardL1.png")

    # def LK_equation(I, J, J_org, k_size, sigma, w, iterations):
    k_size = 3
    sigma = 1
    w = 21
    iterations = 1

    d, Jv = LK_equation(I, J, J, k_size, sigma, w, iterations)
    error, diff, Jv = errorC(I, J, d)

    # def LK_equation_multi(I, J, numScales, *args):
    numScales = 5
    LK_equation_multi(I, J, numScales)
