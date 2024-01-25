import numpy as np
import imageio

"""
   Convert an RGB image to grayscale.

   This function applies a fixed weighting of the color channels to form the
   resulting intensity image.

   Arguments:
      rgb   - a 3D numpy array of shape (sx, sy, 3) storing an RGB image

   Returns:
      gray  - a 2D numpy array of shape (sx, sy) storing the corresponding
              grayscale image
"""
def rgb2gray(rgb):
    gray = np.dot(rgb[...,:3],[0.29894, 0.58704, 0.11402])
    return gray

"""
   Load an image and convert it to grayscale.

   Arguments:
      filename - image file to load

   Returns:
      image    - a 2D numpy array containing a grayscale image
"""
def load_image(filename):
   image = imageio.imread(filename)
   image = image / 255
   if (image.ndim == 3):
      image = rgb2gray(image)
   return image

"""
   Compute IOU (intersection of union) between two bounding boxes

   Arguments:
       boxA - a numpy array of shape (4,) specifying the bounding box
              in the format (x_min, y_min, x_max, y_max)
       boxB - a numpy array of shape (4,) specifying the bounding box
              in the format (x_min, y_min, x_max, y_max)

   Returns:
        iou - float number, intersection of union
"""
def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou
