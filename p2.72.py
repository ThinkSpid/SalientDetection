import numpy as np
import matplotlib.pyplot as plot

x=np.arange(0,63,1)
y=np.arange(0,63,1)

f=0.5+0.5*np.cos(2*np.pi*((x/8)+(y/32)))


time = x

# Amplitude of the cosine wave is cosine of a variable like time

amplitude = f

# Plot a cosine wave using time and amplitude obtained for the cosine wave

plot.plot(time,amplitude)

# Give a title for the cosine wave plot

plot.title('Cosine wave')

# Give x axis label for the cosine wave plot

plot.xlabel('Time')

# Give y axis label for the cosine wave plot

plot.ylabel('Amplitude = cosine(time)')

# Draw the grid for the graph

plot.grid(True, which='both')

plot.axhline(y=0, color='b')

# Display the cosine wave plot

plot.show()