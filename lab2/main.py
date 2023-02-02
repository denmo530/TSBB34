import cv2
import numpy as np
import scipy
from matplotlib import pyplot as plt
from matplotlib import animation
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
    './images/chessboard/img1.png', cv2.IMREAD_GRAYSCALE).astype(float)

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

# # save the K best tracking points
K = 5
best = np.zeros((len(row), 3))
for i in range(len(row)):
    best[i, 0] = H_thresh[row[i], col[i]]
    best[i, 1] = row[i]
    best[i, 2] = col[i]
best_sorted = best[best[:, 0].argsort()]

# # get K best coords
best_coords = np.zeros((K, 2))
for i in range(K):
    end = np.shape(best_sorted)[0] - 1
    best_coords[i, 0] = best_sorted[end - i, 1]
    best_coords[i, 1] = best_sorted[end - i, 2]

anim_fig, ax = plt.subplots()
patches = []
frames = []

first_im = ax.imshow(I, animated=True, cmap='gray')

# first image
# _, ax = plt.subplots(1, num="1")
# ax.imshow(I, cmap='gray')

# Harris detector
# for i in range(K):  # draw circles
#     marker = plt.scatter(
#         best_coords[i, 1], best_coords[i, 0], s=40, marker='o', edgecolors='#FF420F', clip_on=False)
#     marker.set_facecolor("none")
#     patches.append(ax.add_artist(marker))
# frames.append([first_im, *patches])


# Combine with LK tracker
for i in range(1, 11):
    print('Image:', i, 'of 10')
    J = cv2.imread("./images/chessboard/img%d.png" %
                   i, cv2.IMREAD_GRAYSCALE).astype(float)
    # sub plot
    # fig, ax = plt.subplots(1, num="%d" % i)
    # ax.imshow(J, cmap='gray')

    im = ax.imshow(J, animated=True, cmap='gray')
    patches = []

    # Best feature points
    for j in range(K):
        # prev image points
        x = best_coords[j, 1]
        y = best_coords[j, 0]
        d = LK_equation(I, J, x, y, (window, window), iterations)
        # pos update
        best_coords[j, 1] += d[0]
        best_coords[j, 0] += d[1]
        # print(best_coords[i])
        # update displacement

        marker = plt.scatter(
            best_coords[j, 1], best_coords[j, 0], s=40, marker='o', edgecolors='#FF420F', clip_on=False)
        marker.set_facecolor("none")

        # plt.savefig("./images/chessboard/marked/img%d.png" % im)

        patches.append(ax.add_artist(marker))

    I = J
    frames.append([im, *patches])

ani = animation.ArtistAnimation(
    anim_fig, frames, interval=100, blit=True, repeat_delay=0)


plt.show()
print('Done')
