import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

"""
   Visualize interest point detections on an image.

   Plot N detected interest points as red circles overlaid on the image.

   As a visual indicator of interest point score, scale red color intensity
   with relative score.

   Arguments:
      image       - a grayscale image in the form of a 2D numpy
      xs          - numpy array of shape (N,) containing x-coordinates
      ys          - numpy array of shape (N,) containing y-coordinates
      scores      - numpy array of shape (N,) containing scores
"""
def plot_interest_points(image, xs, ys, scores):
   assert image.ndim == 2, 'image should be grayscale'
   # determine color scale
   s_rank = np.argsort(scores)
   N = s_rank.size
   colors = np.zeros((N,3))
   colors[:,0] = 0.95 * (s_rank / N) + 0.05
   # display points
   plt.figure()
   plt.imshow(image, cmap='gray')
   plt.scatter(xs,ys,c=colors)
   plt.show()

"""
   Visualize feature matches.

   Draw lines from feature locations in the first image to matching locations
   in the second.

   Only display matches with scores above a specified threshold (th).

   Reasonable values for the threshold are dependent on your scheme for
   scoring matches.  Varying the threshold to display only the best matches
   can be a useful debugging tool.

   Arguments:
      image0  - a grayscale image in the form of a 2D numpy (first image)
      image1  - a grayscale image in the form of a 2D numpy (second image)
      xs0     - numpy array of shape (N0,) containing x-coordinates of the
                interest points for features in the first image
      ys0     - numpy array of shape (N0,) containing y-coordinates of the
                interest points for features in the first image
      xs1     - numpy array of shape (N1,) containing x-coordinates of the
                interest points for features in the second image
      ys1     - numpy array of shape (N1,) containing y-coordinates of the
                interest points for features in the second image
      matches - a numpy array of shape (N0,) containing, for each feature in
                the first image, the index of the best match in the second
      scores  - a numpy array of shape (N0,) containing a real-valued score
                for each pair of matched features
      th      - threshold; only display matches with scores above threshold
"""
def plot_matches(image0, image1, xs0, ys0, xs1, ys1, matches, scores, th):
   assert image0.ndim == 2, 'image should be grayscale'
   assert image1.ndim == 2, 'image should be grayscale'
   # combine images
   sy0, sx0 = image0.shape
   sy1, sx1 = image1.shape
   sy = sy0 + sy1
   sx = max(sx0, sx1)
   image = np.zeros((sy, sx))
   image[0:sy0,0:sx0]       = image0;
   image[sy0:sy0+sy1,0:sx1] = image1;
   # get coordinates of matches
   xm = xs1[matches]
   ym = ys1[matches]
   # draw correspondence
   plt.figure()
   plt.imshow(image, cmap='gray')
   X = np.zeros((2))
   Y = np.zeros((2))
   N = matches.size
   for n in range(N):
      if (scores[n] > th):
         X[0] = xs0[n]
         X[1] = xm[n]
         Y[0] = ys0[n]
         Y[1] = ym[n] + sy0
         plt.plot(X,Y,'b-')
         plt.plot(X[0],Y[0],'ro')
         plt.plot(X[1],Y[1],'ro')

"""
   Visualize the predicted bounding box (shown in blue) and ground truth
   bounding box (shown in red) on top of the image

   Arguments:
      image      - a grayscale image in the form of a 2D numpy
      pred_box   - a numpy array of shape (4,) specifying the pred bounding box
                   in the format (x_min, y_min, x_max, y_max)
      target_box - a numpy array of shape (4,) specifying the target bounding box
                   in the format (x_min, y_min, x_max, y_max)
"""
def display_bbox(image, pred_box, target_box):
   assert image.ndim == 2, 'image should be grayscale'
   plt.figure()
   fig,ax = plt.subplots(1)
   ax.imshow(image, cmap='gray')
   rect = patches.Rectangle((pred_box[0], pred_box[1]), pred_box[2] - pred_box[0],
        pred_box[3] - pred_box[1],linewidth=1,edgecolor='blue',facecolor='none')
   ax.add_patch(rect)
   rect = patches.Rectangle((target_box[0], target_box[1]), target_box[2] - target_box[0],
    target_box[3] - target_box[1],linewidth=1,edgecolor='red',facecolor='none')
   ax.add_patch(rect)
