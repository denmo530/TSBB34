import lab1
import numpy as np
from scipy.signal import convolve2d
import cv2

# Implement Single scale Lucas - kanade function
def LK_equation(I, J, w):
    # Find gradient using sobel filter
    sobelx = cv2.Sobel(J, cv2.CV_16S, 1, 0, ksize=3)
    sobely = cv2.Sobel(J, cv2.CV_16S, 0, 1, ksize=3)
    
    # Convert back to unsigned int 8-bit
    sobelx = cv2.convertScaleAbs(sobelx)
    sobely = cv2.convertScaleAbs(sobely)

    # Gradients 
    g_x = convolve2d(J, sobelx, mode="same")
    g_y = convolve2d(J, sobely, mode="same")

    # Beräkna x², y², xy
    g_x2 = g_x ** 2
    g_y2 = g_y ** 2
    g_xy = g_x*g_y

    # Falta alla tre för sig med neighbourhood 15x15 filter med 1or
    con_x2 = convolve2d(g_x2, w, mode="same")
    con_y2 = convolve2d(g_y2, w, mode="same")
    con_xy = convolve2d(g_xy, w, mode="same")

    # Kombinera för T 
    T = np.array([[con_x2, con_xy], [con_xy, con_y2]]).reshape([100, 100, 2, 2])

    # Beräkna e
    # [I(x) - J(x)]*gradienten J(x), där x är varje pixel
    IJ = I - J
    resx = IJ*g_x
    resy = IJ*g_y
    conx = convolve2d(resx, w, mode="same")
    cony = convolve2d(resy, w, mode="same")
    e = np.array([[conx], [cony]]).reshape([100, 100, 2, 1])
    print(e.shape)

    d = np.linalg.solve(T, e)

    print(d)
    print(d.shape)
    
    return 0

if __name__ == "__main__":
    I = lab1.load_image_grayscale("./forwardL/forwardL0.png")
    J = lab1.load_image_grayscale("./forwardL/forwardL1.png")
    # 21x21 filter
    w = np.ones((21, 21))
    
    LK_equation(I, J, w)