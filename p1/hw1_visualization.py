## Homework 1
##
## For this assignment, you will implement basic convolution, denoising,
## downsampling, and edge detection operations.
## Your implementation should be restricted to using low-level primitives
## in numpy (e.g. you may not call a Python library routine for convolution in
## the implementation of your convolution function).
##
## This notebook provides examples for testing your code.
## See hw1.py for detailed descriptions of the functions you must implement.

import numpy as np
import matplotlib.pyplot as plt

from util import *
from hw1 import *

# Problem 1 - Convolution (10 Points)
#
# Implement the conv_2d() function as described in hw1.py.
#
# The example below tests your implementation by convolving with a box filter.

image = load_image('data/69015.jpg')
box = np.array([[1, 1, 1, 1, 1],[1, 1, 1, 1, 1],[1, 1, 1, 1, 1],[1, 1, 1, 1, 1],[1, 1, 1, 1, 1]])
img = conv_2d(image,box)

plt.figure(); plt.imshow(image, cmap='gray')
plt.figure(); plt.imshow(img, cmap='gray')
plt.show()

## Problem 2
# (a) Denoising with Gaussian filtering (5 Points)
##
## Implement denoise_gaussian() as described in hw1.py.
##
## The example below tests your implementation.

image = load_image('data/148089_noisy.png')
imgA  = denoise_gaussian(image, 1.0)
imgB  = denoise_gaussian(image, 2.5)

plt.figure(); plt.imshow(image, cmap='gray')
plt.figure(); plt.imshow(imgA, cmap='gray')
plt.figure(); plt.imshow(imgB, cmap='gray')
plt.show()

# (b) Denoising with bilateral filtering (5 Points)
##
## Implement denoise_bilateral() as described in hw1.py.
##
## The example below tests your implementation.

image = load_image('data/148089_noisy.png')
imgA  = denoise_bilateral(image, sigma_s = 1.0, sigma_r=25.5)
imgB  = denoise_bilateral(image, sigma_s = 3.0, sigma_r=25.5)

plt.figure(); plt.imshow(image, cmap='gray')
plt.figure(); plt.imshow(imgA, cmap='gray')
plt.figure(); plt.imshow(imgB, cmap='gray')
plt.show()

## Problem 3 - Smoothing and downsampling (5 Points)
##
## Implement smooth_and_downsample() as described in hw1.py.
##
## The example below tests your implementation.

image  = load_image('data/69015.jpg')
ds_image = smooth_and_downsample(image, downsample_factor = 2)

plt.figure(); plt.imshow(image)
plt.figure(); plt.imshow(ds_image)
plt.show()

## Problem 4 - Bilinear upsampling (5 Points)
##
## Implement bilinear_upsampling() as described in hw1.py.
##
## The example below tests your implementation.

image  = load_image('data/69015.jpg')
us_image = bilinear_upsampling(image, upsample_factor = 2)

plt.figure(); plt.imshow(image)
plt.figure(); plt.imshow(us_image)
plt.show()

## Problem 5 - Sobel gradient operator (5 Points)
##
## Implement sobel_gradients() as described in hw1.py.
##
## The example below tests your implementation.

image  = load_image('data/69015.jpg')
dx, dy = sobel_gradients(image)

plt.figure(); plt.imshow(image, cmap='gray')
plt.figure(); plt.imshow(dx, cmap='gray')
plt.figure(); plt.imshow(dy, cmap='gray')
plt.show()

# Problem 6 -  (a) Nonmax suppression (10 Points)
#              (b) Edge linking and hysteresis thresholding (10 Points)
#              (c) Canny edge detection (5 Points)
#
# Implement nonmax_suppress(), hysteresis_edge_linking(), canny() as described in hw1.py
#
# The examples below test your implementation

image  = load_image('data/edge_img/easy/002.jpg')
mag, nonmax, edge = canny(image)
plt.figure(); plt.imshow(mag, cmap='gray')
plt.figure(); plt.imshow(nonmax, cmap='gray')
plt.figure(); plt.imshow(edge, cmap='gray')
plt.show()

# Extra Credits:
# (a) Improve Edge detection image quality (5 Points)
