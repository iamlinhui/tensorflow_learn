# func3d.py
import numpy as np
from matplotlib import pyplot as plt

x, y = np.mgrid[1:50000:100j, 1:36:48j]

z = (x * 0.08 / 12 * np.power((0.08 / 12) + 1, y)) / (np.power((0.08 / 12) + 1, y) - 1)

fig = plt.figure(figsize=(12, 9))
ax = fig.gca(projection='3d')
ax.plot_surface(x, y, z, cmap=plt.get_cmap('rainbow'))

ax.set_xlabel('\nloan amount(x)')
ax.set_ylabel('\nLoan term(y)')
ax.set_zlabel('pmt(z)')

plt.show()
