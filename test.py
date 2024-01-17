from util import *
from hw1 import *
import numpy as np
import matplotlib.pyplot as plt
import math


image = load_image('data/edge_img/easy/003.jpg')
mag, suppressed, edges = canny(image, [1,2])

plt.figure(); plt.imshow(edges, cmap='gray')
plt.figure(); plt.imshow(mag, cmap='gray')
plt.figure(); plt.imshow(suppressed, cmap='gray')

plt.show()
