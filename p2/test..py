from hw2 import *
from util import *
from visualize import *

img = load_image("./data/easy/001.jpg")
xs, ys, scores = find_interest_points(img)
plot_interest_points(img, xs, ys, scores)
