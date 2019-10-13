import dippykit as dip
import numpy as np
#define a length
length = 64


x_in, y_in = np.ogrid[(-2*length):(2*length+1), (-2*length):(2*length+1)]
x_out, y_out = np.ogrid[(-length):(length+1), (-length):(length+1)]

f=0.5+0.5*np.cos(2*np.pi*((x_in/8)+(y_in/32)))

dip.surf(x_out, y_out, f, cmap='summer')
dip.xlabel('m')
dip.ylabel('n')
dip.title('$cosine wave$')
dip.show()