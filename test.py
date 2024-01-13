from util import *
from hw1 import *
import numpy as np
import matplotlib.pyplot as plt
import math


image = load_image('data/edge_img/easy/002.jpg')

image = denoise_gaussian(image)
mag, theta = get_mag_theta(image)
suppressed = nonmax_suppress(mag, theta)
plt.figure(); plt.imshow(suppressed, cmap='gray')
plt.figure(); plt.imshow(theta, cmap='gray')
plt.figure(); plt.imshow(image, cmap='gray')
plt.show()
#plt.show()
#print(gaussian_1d(5))
#plt.show()
# print(gaussian(3, 1))
