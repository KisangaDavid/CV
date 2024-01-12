from util import *
from hw1 import *
import numpy as np
import matplotlib.pyplot as plt
import math


# def gaussian(x,sigma):
#     return (1.0/(2*np.pi*(sigma**2)))*np.exp(-(x**2)/(2*(sigma**2)))
ones = np.ones((3,3))
# denoise_gaussian(25.5)
#image = load_image('data/69015.jpg')
#box = np.array([[1, 1, 1, 1, 1],[1, 1, 1, 1, 1],[1, 1, 1, 1, 1],[1, 1, 1, 1, 1],[1, 1, 1, 1, 1]])
#img = conv_2d(image,box)
#trimmed_image = trim_border(image, wx=0, wy=50)
#plt.figure(); plt.imshow(image, cmap='gray')
#plt.figure(); plt.imshow(trimmed_image, cmap='gray')
#filter = np.array([[1,1,1,1,1,1,1,1,1],[1,1,1,1,1,1,1,1,1],[1,1,1,1,1,1,1,1,1]])
#convoluted_img = conv_2d(image, filter, 'mirror')
#gaussian_image = denoise_gaussian(image, sigma=4)
#plt.figure(); plt.imshow(gaussian_image, cmap='gray')
image = load_image('data/295087.jpg')
bilateral_denoised = denoise_bilateral(image, 3, 25)
#filter = np.array([[1,1,1,1,1,1,1,1,1],[1,1,1,1,1,1,1,1,1],[1,1,1,1,1,1,1,1,1]])
#convoluted_img = conv_2d(image, filter, 'mirror')
#print(gaussian(0, 25.5))
plt.figure(); plt.imshow(bilateral_denoised, cmap='gray')
plt.show()
#plt.show()
#print(gaussian_1d(5))
#plt.show()
# print(gaussian(3, 1))
