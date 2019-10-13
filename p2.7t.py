import dippykit as dip
import numpy as np

# Define a length signifying half the width of the domain of the square 2D
# signal
length = 64

# Create ogrids for the input and output domains
# Make the input domain twice as wide as the output domain to minimize
# convolution artifacts in the
x_in, y_in = np.ogrid[(-2*length):(2*length+1), (-2*length):(2*length+1)]
x_out, y_out = np.ogrid[(-length):(length+1), (-length):(length+1)]

# Create the function defined in the homework (f)
f = np.zeros((x_in*y_in).shape)

f=0.5+0.5*np.cos(2*np.pi*((x_in/8)+(y_in/32)))

# Convolve the signal with itself and retain the signal domain size
result_conv = dip.convolve2d(f, f, mode='same')

# Extract the center of the convolved signal
result_conv = result_conv[length:(3*length+1), length:(3*length+1)]

# Plot the convolved signal
dip.surf(x_out, y_out, result_conv, cmap='summer')
dip.xlabel('m')
dip.ylabel('n')
dip.title('$f[m,n]*f[m,n]$')
dip.show()
