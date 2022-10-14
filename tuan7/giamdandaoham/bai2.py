import numpy as np
from mpl_toolkits import mplot3d
from matplotlib import pyplot as plt

ax = plt.axes(projection="3d")

x = np.linspace(-2, 2, 11)
y = np.linspace(-2, 2, 11)

X, Y = np.meshgrid(x,y)
Z = X**2 + Y**2

#ax.plot_surface(X, Y, Z)
ax.plot_wireframe(X, Y, Z)
plt.show()
