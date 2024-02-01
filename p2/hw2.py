import numpy as np
import math

# Helper functions from hw1
def mirror_border(image, wx = 1, wy = 1):
   assert image.ndim == 2, 'image should be grayscale'
   sx, sy = image.shape
   # mirror top/bottom
   top    = image[:wx:,:]
   bottom = image[(sx-wx):,:]
   img = np.concatenate( \
      (top[::-1,:], image, bottom[::-1,:]), \
      axis=0 \
   )
   # mirror left/right
   left  = img[:,:wy]
   right = img[:,(sy-wy):]
   img = np.concatenate( \
      (left[:,::-1], img, right[:,::-1]), \
      axis=1 \
   )
   return img

def pad_border(image, wx = 1, wy = 1):
   assert image.ndim == 2, 'image should be grayscale'
   sx, sy = image.shape
   img = np.zeros((sx+2*wx, sy+2*wy))
   img[wx:(sx+wx),wy:(sy+wy)] = image
   return img

def gaussian_1d(sigma = 1.0):
   width = np.ceil(3.0 * sigma)
   x = np.arange(-width, width + 1)
   g = np.exp(-(x * x) / (2 * sigma * sigma))
   g = g / np.sum(g)          # normalize filter to sum to 1 ( equivalent
   g = np.atleast_2d(g)       # to multiplication by 1 / sqrt(2*pi*sigma^2) )
   return g

def conv_2d(image, filt, mode='zero'):
   assert image.ndim == 2, 'image should be grayscale'
   filt = np.atleast_2d(filt)
   assert mode == 'zero' or mode == 'mirror', 'mode must be zero or mirror'

   h_padding = filt.shape[1] // 2
   v_padding =  filt.shape[0] // 2
   new_image = np.zeros(image.shape)

   if mode == 'zero':
      padded_image = pad_border(image, wx=v_padding, wy=h_padding)
   else: 
      padded_image = mirror_border(image, wx=v_padding, wy=h_padding)

   for x_idx in range(new_image.shape[1]):
      x_idx = x_idx + h_padding
      for y_idx in range(new_image.shape[0]):
         y_idx = y_idx + v_padding
         sub_img = padded_image[y_idx - v_padding: y_idx + v_padding + 1,
                                x_idx - h_padding: x_idx + h_padding + 1]
         filtered_pixel = np.sum((sub_img * filt))
         new_image[y_idx - v_padding, x_idx - h_padding] = filtered_pixel
   return new_image

def denoise_gaussian(image, sigma = 1.0):
   horiz_gaussian = gaussian_1d(sigma)
   vert_gaussian = horiz_gaussian.T
   img = conv_2d(image, horiz_gaussian, 'mirror')
   img = conv_2d(img, vert_gaussian, 'mirror')
   return img

def sobel_gradients(image):
   component1 = np.array([1, 2, 1]).reshape((1,3))
   component2 = np.array([-1,0,1]).reshape((1,3))
   dx = conv_2d(image, component1.T, 'mirror')
   dx = conv_2d(dx, component2, 'mirror')

   dy = conv_2d(image, component2.T, 'mirror')
   dy = conv_2d(dy, component1, 'mirror')

   return dx, dy

def get_mag_theta(dx, dy):
   mag = np.sqrt(dx*dx +  dy*dy)
   theta = np.arctan2(dy,dx)
   return mag, theta
"""
   INTEREST POINT OPERATOR (12 Points Implementation + 3 Points Write-up)

   Implement an interest point operator of your choice.

   Your operator could be:

   (A) The Harris corner detector (Szeliski 7.1.1)

               OR

   (B) The Difference-of-Gaussians (DoG) operator defined in:
       Lowe, "Distinctive Image Features from Scale-Invariant Keypoints", 2004.
       https://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf

               OR

   (C) Any of the alternative interest point operators appearing in
       publications referenced in Szeliski or in lecture

              OR

   (D) A custom operator of your own design

   You implementation should return locations of the interest points in the
   form of (x,y) pixel coordinates, as well as a real-valued score for each
   interest point.  Greater scores indicate a stronger detector response.

   In addition, be sure to apply some form of spatial non-maximum suppression
   prior to returning interest points.

   Whichever of these options you choose, there is flexibility in the exact
   implementation, notably in regard to:

   (1) Scale

       At what scale (e.g. over what size of local patch) do you operate?

       You may optionally vary this according to an input scale argument.

       We will test your implementation at the default scale = 1.0, so you
       should make a reasonable choice for how to translate scale value 1.0
       into a size measured in pixels.

   (2) Nonmaximum suppression

       What strategy do you use for nonmaximum suppression?

       A simple (and sufficient) choice is to apply nonmaximum suppression
       over a local region.  In this case, over how large of a local region do
       you suppress?  How does that tie into the scale of your operator?

   For making these, and any other design choices, keep in mind a target of
   obtaining a few hundred interest points on the examples included with
   this assignment, with enough repeatability to have a large number of
   reliable matches between different views.

   If you detect more interest points than the requested maximum (given by
   the max_points argument), return only the max_points highest scoring ones.

   In addition to your implementation, include a brief write-up (in hw2.pdf)
   of your design choices.

   Arguments:
      image       - a grayscale image in the form of a 2D numpy array
      max_points  - maximum number of interest points to return
      scale       - (optional, for your use only) scale factor at which to
                    detect interest points
      mask        - (optional, for your use only) foreground mask constraining
                    the regions to extract interest points
   Returns:
      xs          - numpy array of shape (N,) containing x-coordinates of the
                    N detected interest points (N <= max_points)
      ys          - numpy array of shape (N,) containing y-coordinates
      scores      - numpy array of shape (N,) containing a real-valued
                    measurement of the relative strength of each interest point
                    (e.g. corner detector criterion OR DoG operator magnitude)
"""
def find_interest_points(image, max_points = 200, scale = 1.0, mask = None):
   # check that image is grayscale
  # denoise_gaussian(image, 1)
   assert image.ndim == 2, 'image should be grayscale'
   if mask is not None:
      image = image * mask
   dx, dy = sobel_gradients(image)
   dx2 = denoise_gaussian(dx * dx, 1)
   dy2 = denoise_gaussian(dy * dy, 1)
   dxdy = denoise_gaussian(dx * dy, 1)

   window_radius = 3
   bin_width = 5
   zero_length = bin_width + bin_width // 2
   # R values seem funky
   R = (dx2*dy2 - ((dxdy * dxdy))) - (0.06) * ((dx2 + dy2) * (dx2 + dy2))
   # Set firs / last window_radius columns to 0 (do not want to find features where we can't calculate their descriptors)
   R[:,:zero_length] = 0
   R[:,-zero_length:] = 0
   R[:zero_length,:] = 0
   R[-zero_length:,:] = 0

   # Spacial non-max suppression using the given window radius
   for cur_y in range(window_radius, image.shape[0] - window_radius, window_radius * 2 + 1):
      for cur_x in range(window_radius, image.shape[1] - window_radius, window_radius * 2 + 1):
         curWindow = R[cur_y - window_radius: cur_y + window_radius + 1, 
                      cur_x - window_radius: cur_x + window_radius + 1]
         maxIdx = np.unravel_index(curWindow.argmax(), curWindow.shape)
         newWindow = np.zeros(curWindow.shape)
         newWindow[maxIdx] = curWindow[maxIdx]
         R[cur_y - window_radius: cur_y + window_radius + 1, 
           cur_x - window_radius: cur_x + window_radius + 1] = newWindow

   # lower_limit is the max_points + 1 highest R value
   lower_limit = np.partition(R.flatten(), (R.flatten().size - max_points - 1))[R.flatten().size - max_points - 1]
  # print(R[R > 0].shape)
  # print(lower_limit)
   corner_idxs = np.nonzero((R > 0) & (R > lower_limit))
   regularized_scores = R[corner_idxs] / np.max(R[corner_idxs])
 #  print(regularized_scores.shape)
   ys = corner_idxs[0]
   xs = corner_idxs[1]
   scores = regularized_scores.flatten()

   return xs, ys, scores

"""
   FEATURE DESCRIPTOR (12 Points Implementation + 3 Points Write-up)

   Implement a SIFT-like feature descriptor by binning orientation energy
   in spatial cells surrounding an interest point.

   Unlike SIFT, you do not need to build-in rotation or scale invariance.

   A reasonable default design is to consider a 3 x 3 spatial grid consisting
   of cell of a set width (see below) surrounding an interest point, marked
   by () in the diagram below.  Using 8 orientation bins, spaced evenly in
   [-pi,pi), yields a feature vector with 3 * 3 * 8 = 72 dimensions.

             ____ ____ ____
            |    |    |    |
            |    |    |    |
            |____|____|____|
            |    |    |    |
            |    | () |    |
            |____|____|____|
            |    |    |    |
            |    |    |    |
            |____|____|____|

                 |----|
                  width

   You will need to decide on a default spatial width.  Optionally, this can
   be a multiple of a scale factor, passed as an argument.  We will only test
   your code by calling it with scale = 1.0.

   In addition to your implementation, include a brief write-up (in hw2.pdf)
   of your design choices.

  Arguments:
      image    - a grayscale image in the form of a 2D numpy
      xs       - numpy array of shape (N,) containing x-coordinates
      ys       - numpy array of shape (N,) containing y-coordinates
      scale    - scale factor

   Returns:
      feats    - a numpy array of shape (N,K), containing K-dimensional
                 feature descriptors at each of the N input locations
                 (using the default scheme suggested above, K = 72)
"""
def extract_features(image, xs, ys, scale = 1.0):
   # check that image is grayscale
   bin_width = 5

   assert image.ndim == 2, 'image should be grayscale'
   image = denoise_gaussian(image)
   dx, dy = sobel_gradients(image)
   mag, theta = get_mag_theta(dx, dy)
   feats = np.zeros((xs.shape[0], 3 * 3 * 8))
   for idx, x in enumerate(xs):
      # - pi = idx 0, pi = idx 7
      histogram = np.zeros((3,3,8))
      y = ys[idx]
      # TODO catch edge cases on edge of image
      partition_mag = mag[y - bin_width - bin_width //2: y + bin_width + bin_width //2 + 1, 
            x - bin_width - bin_width // 2: x + bin_width + bin_width //2 + 1]
      # Code to convolve with gaussian  
      # horiz_gaussian = gaussian_1d(2.1)
      # vert_gaussian = horiz_gaussian.T
      # window_gaussian = vert_gaussian@horiz_gaussian
      # partition_mag = partition_mag * window_gaussian
      partition_theta = theta[y - bin_width - bin_width //2: y + bin_width + bin_width //2 + 1, 
            x - bin_width - bin_width // 2: x + bin_width + bin_width //2 + 1]
      for partition_y in range(partition_theta.shape[0]):
         for partition_x in range(partition_theta.shape[1]):

            cur_theta = partition_theta[partition_y, partition_x]
            cur_mag = partition_mag[partition_y, partition_x]
            cur_theta += np.pi
            cur_dir = cur_theta / (np.pi / 4)
            # catch edge case if theta == 2 pi
            if cur_dir == 8.0:
               dir_idx = 0
            else:
               dir_idx = int(np.floor(cur_dir))
            histogram[partition_y // bin_width, partition_x // bin_width, dir_idx] += cur_mag * (1 - (cur_dir - np.floor(cur_dir)))
            if cur_dir > 7:
               histogram[partition_y // bin_width, partition_x // bin_width, 0] += cur_mag * (1 - (np.ceil(cur_dir) - cur_dir))
            else:
               histogram[partition_y // bin_width, partition_x // bin_width, int(np.ceil(cur_dir))] += cur_mag * (1 - (np.ceil(cur_dir) - cur_dir))
      feats_vector = histogram.flatten()
      feats_vector = feats_vector / np.linalg.norm(feats_vector)
      feats_vector[feats_vector > 0.2] = 0.2
      feats_vector = feats_vector / np.linalg.norm(feats_vector)
      feats[idx] = feats_vector
   return feats


"""
   FEATURE MATCHING (7 Points Implementation + 3 Points Write-up)

   Given two sets of feature descriptors, extracted from two different images,
   compute the best matching feature in the second set for each feature in the
   first set.

   Matching need not be (and generally will not be) one-to-one or symmetric.
   Calling this function with the order of the feature sets swapped may
   result in different returned correspondences.

   For each match, also return a real-valued score indicating the quality of
   the match.  This score could be based on a distance ratio test, in order
   to quantify distinctiveness of the closest match in relation to the second
   closest match.  It could optionally also incorporate scores of the interest
   points at which the matched features were extracted.  You are free to
   design your own criterion.

   In addition to your implementation, include a brief write-up (in hw2.pdf)
   of your design choices.

   Arguments:
      feats0   - a numpy array of shape (N0, K), containing N0 K-dimensional
                 feature descriptors (generated via extract_features())
      feats1   - a numpy array of shape (N1, K), containing N1 K-dimensional
                 feature descriptors (generated via extract_features())
      scores0  - a numpy array of shape (N0,) containing the scores for the
                 interest point locations at which feats0 was extracted
                 (generated via find_interest_point())
      scores1  - a numpy array of shape (N1,) containing the scores for the
                 interest point locations at which feats1 was extracted
                 (generated via find_interest_point())

   Returns:
      matches  - a numpy array of shape (N0,) containing, for each feature
                 in feats0, the index of the best matching feature in feats1
      scores   - a numpy array of shape (N0,) containing a real-valued score
                 for each match
"""
def match_features(feats0, feats1, scores0, scores1):
   scores = np.zeros(scores0.shape)
   matches = np.zeros(scores0.shape)
   for outer_idx, feat1 in enumerate(feats0):
      closest_feat_distance = np.inf
      second_closest_feat_distance = np.inf
      closest_feat_idx = None
     # closest_feat_score = 0
      for idx, feat2 in enumerate(feats1):
         cur_distance = np.linalg.norm(feat1 - feat2)
         if cur_distance < closest_feat_distance:
            second_closest_feat_distance = closest_feat_distance
            closest_feat_distance = cur_distance
            closest_feat_idx = idx
            continue
         if cur_distance < second_closest_feat_distance:
            second_closest_feat_distance = cur_distance
      if closest_feat_distance == 0:
         scores[outer_idx] = 1000
         print("DISTANCE 0!")
      else:
         scores[outer_idx] = second_closest_feat_distance / closest_feat_distance
      matches[outer_idx] = closest_feat_idx

   return matches.astype(int), scores

"""
   HOUGH TRANSFORM (7 Points Implementation + 3 Points Write-up)

   Assuming two images of the same scene are related primarily by
   translational motion, use a predicted feature correspondence to
   estimate the overall translation vector t = [tx ty].

   Your implementation should use a Hough transform that tallies votes for
   translation parameters.  Each pair of matched features votes with some
   weight dependant on the confidence of the match; you may want to use your
   estimated scores to determine the weight.

   In order to accumulate votes, you will need to decide how to discretize the
   translation parameter space into bins.

   In addition to your implementation, include a brief write-up (in hw2.pdf)
   of your design choices.

   Arguments:
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

   Returns:
      tx      - predicted translation in x-direction between images
      ty      - predicted translation in y-direction between images
      votes   - a matrix storing vote tallies; this output is provided for
                your own convenience and you are free to design its format
"""
def hough_votes(xs0, ys0, xs1, ys1, matches, scores):
   ##########################################################################
   bin_size = 3
   ys = np.concatenate([ys0, ys1])
   xs = np.concatenate([xs0, xs1])
   max_y_dim = max(ys)
   max_x_dim = max(xs)
   vote_y_dim = int(np.ceil(max_y_dim / bin_size) + 1)
   vote_x_dim = int(np.ceil(max_x_dim / bin_size) + 1)
   votes = np.zeros((vote_y_dim, vote_x_dim))
   for idx, x0 in enumerate(xs0):
      best_match_idx = matches[idx]
      y0 = ys0[idx]
      x0 = xs0[idx]
      x1 = xs1[best_match_idx]
      y1 = ys1[best_match_idx]
      x_translation = x1 - x0
      y_translation = y1 - y0
      # add to 4 different bins, can interpolate for better results
      # TODO can check if template fits on image
      # X translation or y translation < 0 is not possible, template image is considered top left
      if x_translation < 0 or y_translation < 0:
         continue
      # add to corresponding bin
      y_raw = y_translation /bin_size
      x_raw = x_translation /bin_size
      y_bucket = int(np.floor(y_raw))
      x_bucket = int(np.floor(x_raw))
      votes[y_bucket, x_bucket] += ((scores[idx] - 1))

      nearest_y_dir = int(2 * np.round(y_raw - y_bucket) - 1)
      nearest_x_dir = int(2 * np.round(x_raw - x_bucket) - 1)
      cur_score = scores[idx] - 1

      if x_bucket + nearest_x_dir > 0 and x_bucket + nearest_x_dir < vote_x_dim - 1:
         votes[y_bucket, x_bucket + nearest_x_dir] += cur_score 
      if y_bucket + nearest_y_dir > 0 and y_bucket + nearest_y_dir < vote_y_dim - 1:
         votes[y_bucket + nearest_y_dir, x_bucket] += cur_score
      if x_bucket + nearest_x_dir > 0 and x_bucket + nearest_x_dir < vote_x_dim - 1 \
         and y_bucket + nearest_y_dir > 0 and y_bucket + nearest_y_dir < vote_y_dim - 1:
         votes[y_bucket + nearest_y_dir, x_bucket + nearest_x_dir] += cur_score
      # add to closest surrounding buckets as well
      #
      
      # votes[y_bucket, x_bucket] += cur_score
      # if y_bucket != 0:
      #    votes[y_bucket - 1, x_bucket] += cur_score * .8
      # if x_bucket != 0:
      #    votes[y_bucket, x_bucket - 1] += cur_score * .8
      # if y_bucket != vote_y_dim - 1:
      #    votes[y_bucket + 1, x_bucket] += cur_score * .8
      # if x_bucket != vote_x_dim - 1:
      #    votes[y_bucket, x_bucket + 1] += cur_score * .8

      # add to surrounding 4 corners if not on edge:
      
      # if y_bucket !=  0 and x_bucket != 0:
      #    votes[y_bucket - 1, x_bucket - 1] += cur_score * .6
      # if y_bucket !=  0 and x_bucket != vote_x_dim - 1:
      #    votes[y_bucket - 1, x_bucket + 1] += cur_score * .6
      # if y_bucket !=  vote_y_dim - 1 and x_bucket != vote_x_dim - 1:
      #    votes[y_bucket + 1, x_bucket + 1] += cur_score * .6
      # if y_bucket !=  vote_y_dim - 1 and x_bucket != 0:
      #    votes[y_bucket + 1, x_bucket - 1] += cur_score * .6

      # if y_bucket != 0 and x_bucket != 0:
      #    votes[y_bucket - 1, x_bucket - 1] += ((scores[idx] - 1))
      
    #  votes[y_top, x_top] += ((scores[idx] - 1))

      # y_bottom = 
      # y_top = int(np.ceil(y_translation /bin_size))
      # x_bottom = 
      # x_top = int(np.ceil(x_translation /bin_size))
      
      # votes[y_bottom, x_top] += ((scores[idx] - 1))
      # votes[y_top, x_bottom] += ((scores[idx] - 1))
      
   ty, tx = np.unravel_index(votes.argmax(), votes.shape)
   ty = ty * bin_size
   tx = tx * bin_size
   return tx, ty, votes

"""
    OBJECT DETECTION (10 Points Implementation + 5 Points Write-up)

    Implement an object detection system which, given multiple object
    templates, localizes the object in the input (test) image by feature
    matching and hough voting.

    The first step is to match features between template images and test image.
    To prevent noisy matching from background, the template features should
    only be extracted from foreground regions.  The dense point-wise matching
    is then used to compute a bounding box by hough voting, where box center is
    derived from voting output and the box shape is simply the size of the
    template image.

    To detect potential objects with diversified shapes and scales, we provide
    multiple templates as input.  To further improve the performance and
    robustness, you are also REQUIRED to implement a multi-scale strategy
    either:
       (a) Implement multi-scale interest points and feature descriptors OR
       (b) Repeat a single-scale detection procedure over multiple image scales
           by resizing images.

    In addition to your implementation, include a brief write-up (in hw2.pdf)
    of your design choices on multi-scale implementaion and samples of
    detection results (please refer to display_bbox() function in visualize.py).

    Arguments:
        template_images - a list of gray scale images.  Each image is in the
                          form of a 2d numpy array which is cropped to tightly
                          cover the object.

        template_masks  - a list of binary masks having the same shape as the
                          template_image.  Each mask is in the form of 2d numpy
                          array specyfing the foreground mask of object in the
                          corresponding template image.

        test_img        - a gray scale test image in the form of 2d numpy array
                          containing the object category of interest.

    Returns:
         bbox           - a numpy array of shape (4,) specifying the detected
                          bounding box in the format of
                             (x_min, y_min, x_max, y_max)

"""
def object_detection(template_images, template_masks, test_img):
   scales = [0.8, 0.9, 1, 1.1, 1.2]
   max_points = 200
   xs1, ys1, scores1 = find_interest_points(test_img, max_points = max_points)
   feats1 = extract_features(test_img, xs1, ys1)
   # template_image = template_images[1]
   max_votes = 0
   best_template = None
   
   best_tx, best_ty = -1,-1
   for idx, template_image in enumerate(template_images):
      xs0, ys0, scores0 = find_interest_points(template_image, max_points = max_points, mask=template_masks[idx])
      feats0 = extract_features(template_image, xs0, ys0)
      matches, scores = match_features(feats0, feats1, scores0, scores1)
      tx, ty, votes = hough_votes(xs0, ys0, xs1, ys1, matches, scores)
      # template idx 7 has 10 mil
      if np.max(votes) > max_votes:
         best_tx = tx
         best_ty = ty
         best_template = template_image
         max_votes = np.max(votes)
   bbox = np.array([best_tx, best_ty, best_tx + best_template.shape[1], best_ty + best_template.shape[0]])
   return bbox
