import lab1
import numpy as np
from scipy.signal import convolve2d
import scipy
import matplotlib.pyplot as plt
from scipy.interpolate import RectBivariateSpline
from PIL import Image

# Implement Single scale Lucas - kanade function


def LK_equation(I, J, w):

    # Find gradient using Sobel kernel
    sobelx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobely = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

    # Schar kernel, more accurate according to a source
    # sobelx = np.array([[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]])
    # sobely = np.array([[3, 10, 3], [0, 0, 0], [-3, -10, -3]])

    # Gradients
    g_x = convolve2d(J, sobelx, mode="same")
    g_y = convolve2d(J, sobely, mode="same")

    # Beräkna x², y², xy
    g_x2 = np.multiply(g_x, g_x)
    g_y2 = np.multiply(g_y, g_y)
    g_xy = np.multiply(g_x, g_y)

    # Falta alla tre för sig med neighbourhood 21x21 filter med 1or
    con_x2 = convolve2d(g_x2, w, mode="same")
    con_y2 = convolve2d(g_y2, w, mode="same")
    con_xy = convolve2d(g_xy, w, mode="same")

    # Kombinera för T
    T = np.array([[con_x2, con_xy], [
                 con_xy, con_y2]])
    T = T.transpose((2, 3, 0, 1))
    print(f"T: {T.shape}")

    # Beräkna e
    # [I(x) - J(x)]*gradienten J(x), där x är positonen för varje pixel
    IJ = I - J
    resx = np.multiply(IJ, g_x)
    resy = np.multiply(IJ, g_y)
    conx = convolve2d(resx, w, mode="same")
    cony = convolve2d(resy, w, mode="same")
    e = np.array([conx, cony])
    e = e.transpose((1, 2, 0))
    print(f"e: {e.shape}")
    # print(T.shape)
    # print(e.shape)
    d = np.linalg.solve(T, e)
    # print(d.shape)
    print(f"d: {d.shape}")

    return d


def error(I, J, V):
    # Get width and height, creates array coordinates for the image
    [width, height] = [np.shape(J)[1], np.shape(J)[0]]
    [x_c, y_c] = [np.arange(0, width), np.arange(0, height)]

    # Creates interpolation function
    J_interpol = scipy.interpolate.RectBivariateSpline(y_c, x_c, J)

    # Reshapes into a 2D array
    mesh_grid = np.array(np.meshgrid(x_c, y_c)).reshape(2, -1)

    # Creates a new set of coordinates with the motion field
    J_c = np.array([mesh_grid[0] + V[..., 0].flatten()/2,
                    mesh_grid[1] + V[..., 1].flatten()/2])

    # Applies the interpolation
    J_v = J_interpol(J_c[1], J_c[0], grid=False).reshape(np.shape(J))

    # Calculates the error
    error = np.linalg.norm(J-I)
    diff = np.linalg.norm(J_v-I)

    # Shows the two images side by side
    # lab1.image_grid({"I": I, "J": J}, share_all=True,
    #                 imshow_opts={'cmap': 'gray'})
    # plt.figure("Diff"), plt.imshow(np.abs(J_v-I), cmap='gray')

    print(f"Error: {error} \nDifference: {diff}")
    lab1.gopimage(V)
    plt.show()

    return J_v


def LK_equation_multi(I, J, numScales):
    Vtot = 0
    Jn = J
    for n in range(numScales, 0, -1):
        print('n= ', n)
        sc = 2 ** (n-1)
        V = LK_equation(I, Jn, w * sc)
        Vtot = Vtot + V

        lab1.gopimage(Vtot, scale=3)
        plt.show()
    return Vtot


if __name__ == "__main__":
    I = lab1.load_image_grayscale("./forwardL/forwardL0.png")
    J = lab1.load_image_grayscale("./forwardL/forwardL1.png")
    # 21x21 filter
    w = np.ones((21, 21))

    V = LK_equation(I, J, w)
    J_v = error(I, J, V)

    numScales = 4
    # V_tot = LK_equation_multi(I, J, numScales)
