# import the necessary packages
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
import matplotlib.pyplot as plt
import dippykit as dip
import cv2
import numpy as np

# Function for normalizing an array along its columns
def norm_array(X, x_min = 0, x_max = 1):
    nom = (X-X.min(axis=0))*(x_max-x_min)
    denom = X.max(axis=0) - X.min(axis=0)
    return x_min + nom/denom

#Read the input image
X_f = dip.imread('tiger.jpg')

# display the input image
dip.imshow(X_f)
dip.show()

# Calculate the superpixels. The output of slic is an image with the pixel values as superpixel indices. The size
# of the image is same as the input image.
super_pixels = slic(X_f, n_segments = 300, sigma = 5)

# This is to visualize the boundaries of superpixels
dip.imshow(mark_boundaries(X_f, super_pixels))
dip.show()

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

dip.imshow(G_s_img, 'gray')
dip.show()

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

dip.imshow(img_G_s_bin, 'gray')
dip.show()

# I found out that we need to normalize the RGB vectors and position vectors in order to get some better results.
# Without this, we just get G_s[i] as all zeros, because the term inside the exponent is very large.