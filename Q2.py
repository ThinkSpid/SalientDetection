import dippykit as dip
import numpy as np
from math import pi
## see this for change
X = dip.im_read('cameraman.tif') # Read image
X = dip.im_to_float(X)

# Display original Image
dip.imshow(X, 'gray')
dip.title('Original Cameraman image')
dip.show()

zoom_row = 1/2.5 # zoom by 2.5 units along the rows
shrink_col = 1.7 # shrink by 1.7 units along the columns
rot_ang = 27.5 * (pi/180) # convert to radians

S1 = np.array([[zoom_row, 0], [0, shrink_col]]) # scaling matrix
R2 = np.array([[np.cos(rot_ang), np.sin(rot_ang)], [-np.sin(rot_ang), np.cos(rot_ang)]]) # rotation matrix
C1 = np.matmul(S1, R2) # Composition of rotation and scaling

# Scaling and rotating the image with bicubic interpolation
X_trans = dip.resample(X, C1, interp='bicubic')

# Display transformed image
dip.imshow(X_trans, 'gray')
dip.title('Image after scaling and rotation')
dip.show()

C1_inv = np.linalg.inv(C1) # Inverse of the composite transformation

# Reverse the transform using bicubic interpolation
X_trans_inv = dip.resample(X_trans, C1_inv, interp='bicubic',
                           crop=True, crop_size=X.shape)

# Display Image after inverse transform
dip.imshow(X_trans_inv, 'gray')
dip.title('Image after inverse transformation')
dip.show()

# The difference between the original image and image obtained in (c)
X_sub = np.abs(X - X_trans_inv)
dip.imshow(X_sub, 'gray')
dip.title('Difference between (a) and (c)')
dip.show()

PSNR_trans_X = dip.PSNR(X_trans_inv , X, 1)
print('PSNR between inverse transformed image and original image is:', PSNR_trans_X)

