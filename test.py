from util import *
from hw1 import *
import numpy as np
import matplotlib.pyplot as plt

image = load_image('data/69015.jpg')
#box = np.array([[1, 1, 1, 1, 1],[1, 1, 1, 1, 1],[1, 1, 1, 1, 1],[1, 1, 1, 1, 1],[1, 1, 1, 1, 1]])
#img = conv_2d(image,box)
#trimmed_image = trim_border(image, wx=0, wy=50)
#plt.figure(); plt.imshow(image, cmap='gray')
#plt.figure(); plt.imshow(trimmed_image, cmap='gray')
#filter = np.array([[1,1,1,1,1,1,1,1,1],[1,1,1,1,1,1,1,1,1],[1,1,1,1,1,1,1,1,1]])
#convoluted_img = conv_2d(image, filter, 'mirror')
gaussian_image = denoise_gaussian(image, sigma=4)
plt.figure(); plt.imshow(gaussian_image, cmap='gray')

plt.show()
#print(gaussian_1d(5))
#plt.show()