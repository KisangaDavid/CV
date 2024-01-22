# THIS is reference code using scipy tools
# Only for self check usage
import numpy as np
from scipy import ndimage
import scipy.signal
import scipy.io as sio
from util import *
import matplotlib.pyplot as plt
from hw1 import sobel_gradients

def conv_2d(image, filt, mode='zero'):
   assert image.ndim == 2, 'image should be grayscale'
   filt = np.atleast_2d(filt)
   if mode =='zero':
       result = scipy.signal.convolve2d(image, filt,
                                  mode='same', boundary='fill', fillvalue=0)
   elif mode == 'mirror':
       result = scipy.signal.convolve2d(image, filt,
                                  mode='same', boundary='symm')
   else:
       raise NotImplementedError
   return result


def denoise_gaussian(image, sigma = 1.0):
   img = ndimage.gaussian_filter(image, sigma)
   return img

def sobel_gradients_(img):
    gx = np.array([[1, 0, -1],
                    [2, 0, -2],
                    [1, 0, -1]])
    gy = np.array([[1, 2, 1],
                    [0, 0, 0],
                    [-1, -2, -1]])
    dx = conv_2d(img, gx, mode='mirror')
    dy = conv_2d(img, gy, mode='mirror')
    return dx, dy

image = load_image('data/78004.jpg')
#filter = np.array([[1,1,1,1,1,1,1,1,1],[1,1,1,1,1,1,1,1,1],[1,1,1,1,1,1,1,1,1]])
#convoluted_img = conv_2d(image, filter, 'mirror')
dx_img, dy_img = sobel_gradients_(image)
custom_dx_img, custom_dy_img = sobel_gradients(image)
plt.figure(); plt.imshow(dx_img)
plt.figure(); plt.imshow(dy_img)
plt.figure(); plt.imshow(custom_dx_img)
plt.figure(); plt.imshow(custom_dy_img)
plt.show()