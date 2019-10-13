import dippykit as dip
import numpy as np

#define a length
length = 6

#create orgids
x_in, y_in = np.ogrid[(-2*length):(2*length+1), (-2*length):(2*length+1)]
x_out, y_out = np.ogrid[(-length):(length+1), (-length):(length+1)]

#define the function from the homework
f = np.zeros((x_in*y_in).shape)
f[(0 <= x_in) & (x_in <= y_in)] = 1

#do the convolve
result_conv = dip.convolve2d(f, f, mode='same')

#get the center part
result_conv = result_conv[length:(3*length+1), length:(3*length+1)]

#plot the result
dip.surf(x_out, y_out, result_conv, cmap='summer')
dip.xlabel('m')
dip.ylabel('n')
dip.title('$f[m,n]*f[m,n]$')
dip.show()
