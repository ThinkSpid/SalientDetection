import dippykit as dip
import numpy as np

X=dip.im_read("/Users/binbinren/Downloads/images/cameraman.tif")
X=dip.im_to_float(X)

X *=255
Y=X+75

Y=dip.float_to_im(Y/255)

dip.im_write(Y,"/Users/binbinren/Downloads/images/problemset/cameraman_add.png")

Z=X**2
Z=dip.float_to_im(Z/255)
dip.im_write(Z,"/Users/binbinren/Downloads/images/problemset/cameraman_square.png")

fX=dip.fft2(X)
fX=dip.fftshift(fX)
fX=np.log(np.abs(fX))

dip.imshow(fX)
dip.show()
