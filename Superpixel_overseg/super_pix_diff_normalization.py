# import the necessary packages
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
import matplotlib.pyplot as plt
import dippykit as dip
import cv2
import numpy as np

# This does almost the same thing as the super_pix.py, except it changes some parameters and performs normalization of
# the RGB vector mu_super_pix and position vector ctr_super_pix. I'm doing this in order to get some better results.
# The paper does not mention anything about normalization, but the results are really bad if we just implement what they
# wrote. Also note that it takes quite a lot of time to execute.

# Function for normalizing an array along its columns
def norm_array(X, x_min = 0, x_max = 1):
    nom = (X-X.min(axis=0))*(x_max-x_min)
    denom = X.max(axis=0) - X.min(axis=0)
    return x_min + nom/denom

#Read the input image
X_f = dip.imread('tiger.jpg')

dip.figure(1)
# display the input image
dip.imshow(X_f)
plt.savefig('p1.png',dpi=300,bbox_inches='tight',pad_inches=0.1)

# Calculate the superpixels. The output of slic is an image with the pixel values as superpixel indices. The size
# of the image is same as the input image.
super_pixels = slic(X_f, n_segments = 500, sigma = 5)

# This is to visualize the boundaries of superpixels
dip.figure(2)
dip.imshow(mark_boundaries(X_f, super_pixels))
plt.savefig('p2.png',dpi=300,bbox_inches='tight',pad_inches=0.1)

# Get the number of superpixels
num_super_pix = len(np.unique(super_pixels))

# We will calculate the centre of each superpixel and its corresponding mean RGB value. They will be stored in arrays
# ctr_super_pix and mu_super_pix.
ctr_super_pix = np.zeros((num_super_pix, 2))
mu_super_pix = np.zeros((num_super_pix, 3))

# Recover the shape of the input image
(size_M, size_N, num_channels) = X_f.shape

for n in range(num_super_pix):
    temp_pix_x = []
    temp_pix_y = []

    temp_S_r = []
    temp_S_g = []
    temp_S_b = []

    for i in range(size_M):
        for j in range(size_N):
            if super_pixels[i][j] == n:
                temp_pix_x.append(i)
                temp_pix_y.append(j)

                temp_S_r.append(X_f[i][j][0])
                temp_S_g.append(X_f[i][j][1])
                temp_S_b.append(X_f[i][j][2])

    ctr_super_pix[n, 0] = np.mean(np.asarray(temp_pix_x))
    ctr_super_pix[n, 1] = np.mean(np.asarray(temp_pix_y))

    mu_super_pix[n, 0] = np.mean(np.asarray(temp_S_r))
    mu_super_pix[n, 1] = np.mean(np.asarray(temp_S_g))
    mu_super_pix[n, 2] = np.mean(np.asarray(temp_S_b))

# Normalize these arrays using norm_array function
mu_super_pix = norm_array(mu_super_pix)
ctr_super_pix = norm_array(ctr_super_pix, 0, 10)

# Calculate the color contrast prior, G_s[i]
G_s = np.zeros((num_super_pix, 1))

gamma_i = 1
delta = 0.5

for i in range(num_super_pix):
    for j in range(num_super_pix):
        diff_mu = np.subtract(mu_super_pix[i][:3], mu_super_pix[j][:3])
        diff_p = np.subtract(ctr_super_pix[i][:2], ctr_super_pix[j][:2])

        temp_S = np.square(np.linalg.norm(diff_mu)) * np.exp(-(1/(2 * np.square(delta))) * np.linalg.norm(diff_p))

        G_s[i] = G_s[i] + temp_S

    G_s[i] = (1/gamma_i) * G_s[i]

# Normalize G_s
G_s = norm_array(G_s)

# Visualize G_s as an image ( just for display purposes)
G_s_img = np.zeros((size_M, size_N))

for n in range(num_super_pix):
    for i in range(size_M):
        for j in range(size_N):
            if super_pixels[i][j] == n:
                G_s_img[i][j] = G_s[n]
dip.figure(3)
dip.imshow(G_s_img, 'gray')
plt.savefig('p3.png',dpi=300,bbox_inches='tight',pad_inches=0.1)

# Get the binary version of G_s according to the threshold thresh_G
G_s_bin = np.zeros((num_super_pix, 1))
thresh_G = 0.5

for i in range(num_super_pix):
    if G_s[i] < thresh_G:
        G_s_bin[i] = 0
    else:
        G_s_bin[i] = 1

# Map these binary values to a binary version of the original image, using the array super_pixels. Let this be
# img_G_s_bin
img_G_s_bin = np.zeros((size_M, size_N))

for n in range(num_super_pix):
    for i in range(size_M):
        for j in range(size_N):
            if super_pixels[i][j] == n:
                img_G_s_bin[i][j] = G_s_bin[n]
dip.figure(4)
dip.imshow(img_G_s_bin, 'gray')
# dip.show()
plt.savefig('p4.png',dpi=300,bbox_inches='tight',pad_inches=0.1)
# We need to follow a similar process to find I_s[i] and O_s[i]. All we need to do is replace the mu_super_pix with a
# different feature, such as intensity in case of I_s[i]. For U_s[i], we need to just penalize the distance of from the
# superpixel from centre of the image, so something like e^{-w[P'_s[i] - P_c]^2} should work.