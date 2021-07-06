# func3d.py
import numpy as np
from matplotlib import pyplot as plt

x, y = np.mgrid[1:50000:100j, 1:42:48j]

z = x * y * 0.08 / 360

fig = plt.figure(figsize=(12, 9))
ax = fig.gca(projection='3d')
ax.plot_surface(x, y, z, cmap=plt.get_cmap('rainbow'))

ax.set_xlabel('\nloan amount(x)')
ax.set_ylabel('\nfirst loan days(y)')
ax.set_zlabel('fee(z)')

plt.show()
