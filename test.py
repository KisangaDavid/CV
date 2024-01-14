from util import *
from hw1 import *
import numpy as np
import matplotlib.pyplot as plt
import math


image = load_image('data/edge_img/medium/003.jpg')
image = denoise_gaussian(image,math.sqrt(2))
mag, theta = get_mag_theta(image)
suppressed = nonmax_suppress(mag, theta)
edgelinked = hysteresis_edge_linking(suppressed, theta)

plt.figure(); plt.imshow(edgelinked, cmap='gray')

plt.show()
