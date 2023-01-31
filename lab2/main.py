import cv2
import numpy as np
import scipy
from matplotlib.patches import Circle
from matplotlib import pyplot as plt
import lab2
from LK_equation import LK_equation
from orientation_tensor import orientation_tensor
from harris import harris


k_size = 3
sigma = 1
window = 21
iterations = 10
kappa = 0.05

I, J, dTrue = lab2.get_cameraman()

dtot = LK_equation(I, J, 120, 85, (40, 70), iterations=5)

print('dTrue = ', dTrue)
print('dtot = ', (dtot[0][0], dtot[1][0]))

I = cv2.imread(
    './images/chessboard/img1.png', cv2.IMREAD_GRAYSCALE)


T_field = orientation_tensor(I, k_size, sigma)
# T = estimate_T()

# # xx
# plt.figure(1)
# plt.imshow(T_field[:, :, 0])
# # xy
# plt.figure(2)
# plt.imshow(T_field[:, :, 1])
# # yy
# plt.figure(3)
# plt.imshow(T_field[:, :, 2])
# plt.show()

H_field = harris(T_field, kappa)
# plt.figure(4)
# plt.imshow(H_field)
# plt.show()


thresh = np.amax(H_field) * 0.8
H_mask = H_field > thresh
H_thresh = H_field * H_mask
# plt.figure(5)
# plt.imshow(H_thresh)
# plt.show()

# remove dilation
img_max = scipy.ndimage.maximum_filter(H_thresh, size=3)
[row, col] = np.nonzero((H_thresh == img_max) * H_mask)
# print(len(row))

# # save the K best ones
K = 5
best = np.zeros((len(row), 3))
for i in range(len(row)):
    best[i, 0] = H_thresh[row[i], col[i]]
    best[i, 1] = row[i]
    best[i, 2] = col[i]
best_sorted = best[best[:, 0].argsort()]

# # get K feat coords
best_coords = np.zeros((K, 2))
for i in range(K):
    end = np.shape(best_sorted)[0] - 1
    best_coords[i, 0] = best_sorted[end - i, 1]
    best_coords[i, 1] = best_sorted[end - i, 2]


# # ignore first value
_, ax = plt.subplots(1, num="1")
ax.imshow(I, cmap='gray')
for i in range(K):  # draw circles
    ax.add_patch(Circle((best_coords[i, 1], best_coords[i, 0]),
                        10, fill=False, edgecolor='red', linewidth=1))


# images
for im in range(1, 11):
    # print('Processing image ', im)
    J = cv2.imread("./images/chessboard/img%d.png" % im, cv2.IMREAD_GRAYSCALE)
    # sub plot
    _, ax = plt.subplots(1, num="%d" % im)
    ax.imshow(J, cmap='gray')
    # cv2.imwrite("./images/chessboard/marked/img%d.png" %
    #             im, J)
    # Best feature points
    for i in range(K):
        # prev image points
        x = best_coords[i, 1]
        y = best_coords[i, 0]
        d = LK_equation(I, J, x, y, (window, window), iterations)
        # pos update
        best_coords[i, 1] += d[0]
        best_coords[i, 0] += d[1]
        # print(best_coords[i])
        # update displacement
        ax.add_patch(Circle(
            (best_coords[i, 1], best_coords[i, 0]), 10, fill=False, edgecolor='red', linewidth=1))
        plt.savefig("./images/chessboard/marked/img%d.png" % im)
    I = J


plt.show()
