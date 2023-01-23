import lab1
import numpy as np
from scipy.signal import convolve2d
import cv2
import matplotlib.pyplot as plt
from scipy.ndimage.interpolation import map_coordinates

# Implement Single scale Lucas - kanade function


def LK_equation(I, J, w):
    # Fins rows and cols for J
    rows, cols = J.shape[:2]
    # print(f"rows: {rows} \n cols: {cols}")
    # Find gradient using sobel filter
    # sobelx = cv2.Sobel(J, cv2.CV_16S, 1, 0, ksize=5)
    # sobely = cv2.Sobel(J, cv2.CV_16S, 0, 1, ksize=5)
    sobelx = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    sobely = np.array([[-1, 0, 1], [-2, 0, 1], [-1, 0, 1]])

    # Convert back to unsigned int 8-bit
    # sobelx = cv2.convertScaleAbs(sobelx)
    # sobely = cv2.convertScaleAbs(sobely)

    # Gradients
    g_x = convolve2d(J, sobelx, mode="same")
    g_y = convolve2d(J, sobely, mode="same")

    # Beräkna x², y², xy
    g_x2 = np.multiply(g_x, g_x)
    g_y2 = np.multiply(g_y, g_y)
    g_xy = np.multiply(g_x, g_y)

    # Falta alla tre för sig med neighbourhood 15x15 filter med 1or
    con_x2 = convolve2d(g_x2, w, mode="same")
    con_y2 = convolve2d(g_y2, w, mode="same")
    con_xy = convolve2d(g_xy, w, mode="same")

    # Kombinera för T
    T = np.array([[con_x2, con_xy], [
                 con_xy, con_y2]])
    T = T.transpose((2, 3, 0, 1))
    print(f"T: {T.shape}")

    # Beräkna e
    # [I(x) - J(x)]*gradienten J(x), där x är varje pixel
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
    print(d)
    print(f"d: {d.shape}")
    # print(d.shape)

    return d


if __name__ == "__main__":
    I = lab1.load_image_grayscale("./forwardL/forwardL0.png")
    J = lab1.load_image_grayscale("./forwardL/forwardL5.png")
    # 21x21 filter
    w = np.ones((21, 21))

    V = LK_equation(I, J, w)

    # Calc error
    error = I - J
    error = np.linalg.norm(error)

    # Calc difference
    x, y = np.meshgrid(np.arange(J.shape[1]), np.arange(J.shape[0]))
    x_v = x + V[..., 0]
    y_v = y + V[..., 1]
    J_xv = map_coordinates(J, [x_v, y_v], order=1)
    diff = I - J_xv
    diff = np.linalg.norm(diff)
    print(f"Error: {error} \n Difference: {diff} ")
    lab1.gopimage(V)
    plt.show()
