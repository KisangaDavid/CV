from util import *
from hw1 import *
import numpy as np
import matplotlib.pyplot as plt
import math


image = load_image('data/edge_img/easy/001.jpg')
#image = denoise_gaussian(image,math.sqrt(2))
# dx, dy = sobel_gradients(image)
# mag, theta = get_mag_theta(dx, dy)
# downsampled_theta = smooth_and_downsample(theta)
# upsampled_theta = bilinear_upsampling(downsampled_theta)
# upsampled_theta = upsampled_theta[0:theta.shape[0], 0:theta.shape[1]]
# average_theta = np.mean([upsampled_theta, theta], axis = 0)
mag, suppressed, edges = canny(image, [1,2])
#suppressed = nonmax_suppress(mag, theta)
#edgelinked = hysteresis_edge_linking(suppressed, theta)


plt.figure(); plt.imshow(image, cmap='gray')
plt.figure(); plt.imshow(mag, cmap='gray')
plt.figure(); plt.imshow(suppressed, cmap='gray')
plt.figure(); plt.imshow(edges, cmap='gray')
# plt.figure(); plt.imshow(upsampled_theta, cmap='gray')
# plt.figure(); plt.imshow(average_theta, cmap='gray')

plt.show()
