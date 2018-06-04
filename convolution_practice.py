import numpy as np
from scipy.ndimage.filters import convolve
from utils import convolve_horizontal, convolve_vertical


hx = np.array([[1, 0, -1]])
hy = np.array([[-1], [0], [1]])

h1 = hy.transpose().reshape(3)

a = np.array(
    [[1, 2, 0, 0],
     [5, 3, 4, 0],
     [0, 0, 0, 7],
     [9, 3, 0, 0]])

print(convolve(a, hx, mode='constant', cval=0.0))

print()

print(convolve_horizontal(a, hx))

print()
print()


print(convolve(a, hy, mode='constant', cval=0.0))

print()

print(convolve_vertical(a, hy))
