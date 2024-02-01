from hw2 import *
from util import *
from visualize import *
import numpy as np
import matplotlib.pyplot as plt
import PIL, pickle, time
import scipy.io as sio
from util import *
from visualize import *
from hw2 import *
import cv2 as cv

easy_img = load_image("./data/easy/003.jpg")
# xs, ys, scores = find_interest_points(img)

# extract_features(img, xs, ys)
# plot_interest_points(img, xs, ys, scores)


## Homework 2
##
## For this assignment, you will design and implement interest point detection,
## feature descriptor extraction, feature matching, and apply those functions
## to the task of object detection.  After matching descriptors between a
## template image of an object and a test image containing an example of the
## same object category, your system should predict the location (bounding box)
## of the object in the test image.  This latter step will require fitting a
## transformation to the set of candidate matches.
##
## As interest points, feature descriptors, matching criterion, and object
## detection all involve design choices, there is flexibility in the
## implementation details, and many reasonable solutions. For object detection,
## you will first implement a system that considers translation only, and
## then subsequently implement a version that accounts for the possibility of
## scale change between template and candidate detections.
##
## As was the case for homework 1, your implementation should be restricted to
## using low-level primitives in numpy.
##
## See hw2.py for detailed descriptions of the functions you must implement.
##
## In addition to submitting your implementation in hw2.py, you will also need
## to submit a writeup in the form of a PDF (hw2.pdf) which briefly explains
## the design choices you made and any observations you might have on their
## effectiveness.
##
## Resources:
##
##  - You may want to build your feature descriptor based on Canny edges.
##    You may import your code from homework 1 or define additional functions
##    in hw2.py as needed.
##
##  - Some functions for visualizing interest point detections and feature
##    matches are provided (visualize.py).
##
## Submit:
##
##    hw2.py   - your implementation of homework 2
##    hw2.pdf  - your writeup describing your design choices
##    *.py     - any additional code (e.g., hw1.py if using for Canny edges)


## Examples.
##
## Each data file (*.mat) contains a set of categorical specificed template
## images, masks, test images and object bounding box targets.  By matching the
## features between template and test image, you should localize the object of
## interest in test images by estimating the object box center and box shape.
## Feel free to experiment with different examples.
##
## Data are collected from the Caltech 101 Data Set.

def parse_data(file_name):
    data = sio.loadmat(file_name)
    template_images = data['template_images'][0].tolist()
    template_masks = data['template_masks'][0].tolist()
    test_images = data['test_images']
    if len(test_images.shape) == 2:
        test_images = test_images[0]
    test_images_target = data['test_images_target']
    template_masks = [data > 0  for data in template_masks]
    return template_images, template_masks, test_images, test_images_target

template_images, template_masks, test_images, test_images_targets = parse_data('data_cup.mat')

img0 = template_images[1]
img0_mask = template_masks[1]
img1 = test_images[6]
img1_target = test_images_targets[6]

# cv_img = cv.cornerHarris(img0,2,3,0.06)
# plt.figure(); plt.imshow(img0, cmap='gray')
# img0 = bilinear_upsampling(img0, 10)
# downsampled = smooth_and_downsample(img0, 10)
# plt.figure(); plt.imshow(img0, cmap='gray')
# plt.figure(); plt.imshow(downsampled, cmap = 'gray')
# plt.show()
#N = 100

# easy_xs, easy_ys, easy_scores = find_interest_points(easy_img, N, 1.0)
# plot_interest_points(easy_img, easy_xs, easy_ys, easy_scores)

#xs0, ys0, scores0 = find_interest_points(img0, N, 1.0, img0_mask)
#xs1, ys1, scores1 = find_interest_points(img1, N, 1.0)
# plot_interest_points(img0, xs0, ys0, scores0)
# plot_interest_points(img1, xs1, ys1, scores1)

#feats0 = extract_features(img0, xs0, ys0, 1.0)
#feats1 = extract_features(img1, xs1, ys1, 1.0)

#matches, match_scores = match_features(feats0, feats1, scores0, scores1)

#threshold = 1.1
 # adjust this for your match scoring system
#plot_matches(img0, img1, xs0, ys0, xs1, ys1, matches, match_scores, threshold)
#tx, ty, votes = hough_votes(xs0, ys0, xs1, ys1, matches, match_scores)
#plt.show(block = False)
#plot_interest_points(img0, xs0, ys0, scores0)
#plot_interest_points(img1, xs1, ys1, scores1)
# pred_bbox = object_detection(template_images, template_masks, img1)
# display_bbox(img1, pred_bbox, img1_target)


## Problem 1 - Interest Point Operator
##             (12 Points Implementation + 3 Points Write-up)
##
## (A) Implement find_interest_points() as described in hw2.py    (12 Points)
## (B) Include (in hw2.pdf) a brief description of the design      (3 Points)
##     choices you made and their effectiveness.

# N = 100
# xs0, ys0, scores0 = find_interest_points(img0, N, 1.0, mask = img0_mask)
# xs1, ys1, scores1 = find_interest_points(img1, N, 1.0)

# plot_interest_points(img0, xs0, ys0, scores0)
# plot_interest_points(img1, xs1, ys1, scores1)
# plt.show(block = False)

# ## Problem 2 - Feature Descriptor Extraction
# ##             (12 Points Implementation + 3 Points Write-up)
# ##
# ## (A) Implement extract_features() as described in hw2.py        (12 Points)
# ## (B) Include (in hw2.pdf) a brief description of the design      (3 Points)
# ##     choices you made and their effectiveness.

# feats0 = extract_features(img0, xs0, ys0, 1.0)
# feats1 = extract_features(img1, xs1, ys1, 1.0)
xs1, ys1, scores1 = find_interest_points(img0, 200, 1.0, img0_mask)
plot_interest_points(img0, xs1, ys1, scores1)
plt.show()
# pred_bbox = object_detection(template_images, template_masks, img1, multi_scale = False)
# display_bbox(img1, pred_bbox, img1_target)
# #plt.show(block = False)
# plt.show()
# print(gaussian_1d(2.1
#                   ))

# for data_name in ['data_car']:
#     template_images, template_masks, test_images, test_images_targets = \
#                                         parse_data(data_name + '.mat')
#     iou_list = []
#     time_list = []
#     for i, (test_img, test_img_target_box) in enumerate(zip(test_images, test_images_targets)):
#         t1 = time.time()
#         pred_bbox = object_detection(template_images, template_masks, test_img)
#         t2 = time.time()
#         iou = compute_iou(pred_bbox, test_img_target_box)
#         print('{}th {} image IOU {}'.format(i, data_name, iou))
#         iou_list.append(iou)
#         time_list.append(t2 - t1)
#     print('class {}, average IOU {}, total running time {}s'.format(data_name, np.array(iou_list).mean(), np.array(time_list).sum()))