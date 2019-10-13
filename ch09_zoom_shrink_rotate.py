import dippykit as dip
import numpy as np
from math import cos, sin, pi, radians

im = dip.im_read('cameraman.tif')

# Part (b) 
row_scale = 1/2.5 # enlarge the image by 2.5 along rows
column_scale = 1.7 # shrink the image by 1.7 along columns
ang = 27.5 # angle of rotation

M1 = np.array([[row_scale, 0], [0, column_scale]]) # scaling matrix
M2 = np.array([[cos(radians(ang)), -sin(radians(ang))], [sin(radians(ang)), cos(radians(ang))]]) # rotation matrix
M = np.matmul(M1, M2)

# Scaling and rotating the image using bicubic interpolation
im_interp = dip.resample(im, M, interp='bicubic')

# Part (c) 
# Inverse the original transform matrix to undo the change
M_inv = np.linalg.inv(M)

# Reverse the transform using bicubic interpolation: output shape = original image shape
im_recov = dip.resample(im_interp, M_inv, interp='bicubic', crop=True, crop_size=im.shape)

# Part (d)
# The difference between the original image and image obtained in (c)
diff = abs(im - im_recov)

## Plotting
dip.subplot(2,2,1)
dip.imshow(im, 'gray')
dip.title('(a) Original image', fontsize='x-small')

dip.subplot(2,2,2)
dip.imshow(im_interp, 'gray')
dip.title('(b) Image after transformation', fontsize='x-small')

dip.subplot(2,2,3)
dip.imshow(im_recov, 'gray')
dip.title('(c) Image after inverse transformation of (b)', fontsize='x-small')

dip.subplot(2,2,4)
dip.imshow(diff, 'gray')
dip.title('(d) Difference between (a) and (c)', fontsize='x-small')
dip.show()
