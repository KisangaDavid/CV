import numpy as np
import math
"""
   Mirror an image about its border.

   Arguments:
      image - a 2D numpy array of shape (sx, sy)
      wx    - a scalar specifying width of the top/bottom border
      wy    - a scalar specifying width of the left/right border

   Returns:
      img   - a 2D numpy array of shape (sx + 2*wx, sy + 2*wy) containing
              the original image centered in its interior and a surrounding
              border of the specified width created by mirroring the interior
"""
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

"""
   Pad an image with zeros about its border.

   Arguments:
      image - a 2D numpy array of shape (sx, sy)
      wx    - a scalar specifying width of the top/bottom border
      wy    - a scalar specifying width of the left/right border

   Returns:
      img   - a 2D numpy array of shape (sx + 2*wx, sy + 2*wy) containing
              the original image centered in its interior and a surrounding
              border of zeros
"""
def pad_border(image, wx = 1, wy = 1):
   assert image.ndim == 2, 'image should be grayscale'
   sx, sy = image.shape
   img = np.zeros((sx+2*wx, sy+2*wy))
   img[wx:(sx+wx),wy:(sy+wy)] = image
   return img

"""
   Remove the border of an image.

   Arguments:
      image - a 2D numpy array of shape (sx, sy)
      wx    - a scalar specifying width of the top/bottom border
      wy    - a scalar specifying width of the left/right border

   Returns:
      img   - a 2D numpy array of shape (sx - 2*wx, sy - 2*wy), extracted by
              removing a border of the specified width from the sides of the
              input image
"""
def trim_border(image, wx = 1, wy = 1):
   assert image.ndim == 2, 'image should be grayscale'
   sx, sy = image.shape
   img = np.copy(image[wx:(sx-wx),wy:(sy-wy)])
   return img

"""
   Return an approximation of a 1-dimensional Gaussian filter.

   The returned filter approximates:

   g(x) = 1 / sqrt(2 * pi * sigma^2) * exp( -(x^2) / (2 * sigma^2) )

   for x in the range [-3*sigma, 3*sigma]
"""
def gaussian_1d(sigma = 1.0):
   width = np.ceil(3.0 * sigma)
   x = np.arange(-width, width + 1)
   g = np.exp(-(x * x) / (2 * sigma * sigma))
   g = g / np.sum(g)          # normalize filter to sum to 1 ( equivalent
   g = np.atleast_2d(g)       # to multiplication by 1 / sqrt(2*pi*sigma^2) )
   return g

def gaussian_for_color(intensity_dif, sigma = 1.0):
   g = np.exp(-(intensity_dif * intensity_dif) / (2 * sigma * sigma))
   return g
"""
   CONVOLUTION IMPLEMENTATION (10 Points)

   Convolve a 2D image with a 2D filter.

   Requirements:

   (1) Return a result the same size as the input image.

   (2) You may assume the filter has odd dimensions.

   (3) The result at location (x,y) in the output should correspond to
       aligning the center of the filter over location (x,y) in the input
       image.

   (4) When computing a product at locations where the filter extends beyond
       the defined image, treat missing terms as zero.  (Equivalently stated,3
       treat the image as being padded with zeros around its border).

   You must write the code for the nested loops of the convolutions yourself,
   using only basic loop constructs, array indexing, multiplication, and
   addition operators.  You may not call any Python library routines that
   implement convolution.

   Arguments:
      image  - a 2D numpy array
      filt   - a 1D or 2D numpy array, with odd dimensions
      mode   - 'zero': preprocess using pad_border or 'mirror': preprocess using mirror_border.

   Returns:
      result - a 2D numpy array of the same shape as image, containing the
               result of convolving the image with filt
"""
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

"""
   GAUSSIAN DENOISING (5 Points)

   Denoise an image by convolving it with a 2D Gaussian filter.

   Convolve the input image with a 2D filter G(x,y) defined by:

   G(x,y) = 1 / sqrt(2 * pi * sigma^2) * exp( -(x^2 + y^2) / (2 * sigma^2) )

   You may approximate the G(x,y) filter by computing it on a
   discrete grid for both x and y in the range [-3*sigma, 3*sigma].

   See the gaussian_1d function for reference.

   Note:
   (1) Remember that the Gaussian is a separable filter.
   (2) Denoising should not create artifacts along the border of the image.
       Make an appropriate assumption in order to obtain visually plausible
       results along the border.

   Arguments:
      image - a 2D numpy array
      sigma - standard deviation of the Gaussian

   Returns:
      img   - denoised image, a 2D numpy array of the same shape as the input
"""
def denoise_gaussian(image, sigma = 1.0):
   horiz_gaussian = gaussian_1d(sigma)
   vert_gaussian = horiz_gaussian.T
   img = conv_2d(image, horiz_gaussian, 'mirror')
   img = conv_2d(img, vert_gaussian, 'mirror')
   return img

"""
    BILATERAL DENOISING (5 Points)
    Denoise an image by applying a bilateral filter
    Note:
        Performs standard bilateral filtering of an input image.
        Reference link: https://en.wikipedia.org/wiki/Bilateral_filter

        Basically, the idea is adding an additional edge term to Guassian filter
        described above.

        The weighted average pixels:

        BF[I]_p = 1/(W_p)sum_{q in S}G_s(||p-q||)G_r(|I_p-I_q|)I_q

        In which, 1/(W_p) is normalize factor, G_s(||p-q||) is spatial Guassian
        term, G_r(|I_p-I_q|) is range Guassian term.

        We only require you to implement the grayscale version, which means I_p
        and I_q is image intensity.

    Arguments:
        image       - input image
        sigma_s     - spatial param (pixels), spatial extent of the kernel,
                       size of the considered neighborhood.
        sigma_r     - range param (not normalized, a propotion of 0-255),
                       denotes minimum amplitude of an edge
    Returns:
        img   - denoised image, a 2D numpy array of the same shape as the input
"""

def denoise_bilateral(image, sigma_s=1, sigma_r=25.5):
   assert image.ndim == 2, 'image should be grayscale'
   spacial = gaussian_1d(sigma_s / 3)
   spacial = spacial.T@spacial
   h_padding = sigma_s
   v_padding = h_padding
   new_image = np.zeros(image.shape)
   padded_image = mirror_border(image, wx=v_padding, wy=h_padding)
   wp = 0
   for x_idx in range(new_image.shape[1]):
      x_idx = x_idx + h_padding
      for y_idx in range(new_image.shape[0]):
         y_idx = y_idx + v_padding
         sub_img = padded_image[y_idx - v_padding: y_idx + v_padding + 1,
                                x_idx - h_padding: x_idx + h_padding + 1]
         differences = np.absolute(sub_img - padded_image[y_idx, x_idx])
         range_gaus = gaussian_for_color(differences, sigma_r)
         wp = np.sum(spacial * range_gaus)
         filtered_pixel = np.sum(sub_img * spacial * range_gaus) / wp
         new_image[y_idx - v_padding, x_idx - h_padding] = filtered_pixel
   return new_image

"""
   SMOOTHING AND DOWNSAMPLING (5 Points)

   Smooth an image by applying a gaussian filter, followed by downsampling with a factor k.

   Note:
      Image downsampling is generally implemented as two-step process:

        (1) Smooth images with low pass filter, e.g, gaussian filter, to remove
            the high frequency signal that would otherwise causes aliasing in
            the downsampled outcome.

        (2) Downsample smoothed images by keeping every kth samples.

      Make an appropriate choice of sigma to avoid insufficient or over smoothing.

         In principle, the sigma in gaussian filter should respect the cut-off frequency
         1 / (2 * k) with k being the downsample factor and the cut-off frequency of
         gaussian filter is 1 / (2 * pi * sigma).


   Arguments:
     image - a 2D numpy array
     downsample_factor - an integer specifying downsample rate

   Returns:
     result - downsampled image, a 2D numpy array with spatial dimension reduced
"""
def smooth_and_downsample(image, downsample_factor = 2):
   padded_image = mirror_border(image, downsample_factor, downsample_factor)
   smooth_image = denoise_gaussian(padded_image, downsample_factor / np.pi)
   new_image_h = math.ceil(image.shape[0] / downsample_factor)
   new_image_w = math.ceil(image.shape[1] / downsample_factor)
   new_image = np.zeros((new_image_h, new_image_w))
   for x_idx in range(new_image.shape[1]):
      padded_x = x_idx * downsample_factor + downsample_factor
      for y_idx in range(new_image.shape[0]):
         padded_y = y_idx * downsample_factor + downsample_factor
         sub_img = smooth_image[padded_y: padded_y + downsample_factor,
                                 padded_x: padded_x + downsample_factor]
         averaged_pixel = np.mean(sub_img)
         new_image[y_idx, x_idx] = averaged_pixel
   return new_image

  

"""
   BILINEAR UPSAMPLING (5 Points)

   Upsampling the input images with a factor of k with bilinear kernel

   Note:
      Image upsampling is generally implemented by mapping each output pixel
      (x_out,y_out) onto input images coordinates (x, y) = (x_out / k, y_out / k).
      Then, we use bilinear kernel to compute interpolated color at pixel
      (x,y), which requires to round (x, y) to find 4 neighboured pixels:

          P11 = (x1, y1)      P12 = (x1, y2)
          P21 = (x2, y1)      P22 = (x2, y2)

      where
          x1 = floor(x / k),  y1 = floor(y / k),
          x2 = ceil (x / k),  y2 = ceil (y / k)

      In practice, you can simplify the 2d interpolation formula by applying 1d
      interpolation along each axis:

          # interpolate along x axis
          C(x,y1) = (x2 - x)/(x2 - x1) * C(x1, y1) +  (x - x1)/(x2 - x1) * C(x2, y1)
          C(x,y2) = (x2 - x)/(x2 - x1) * C(x1, y2) +  (x - x1)/(x2 - x1) * C(x2, y2)

          # interpolate along y axis
          C(x,y) =  (y2 - y)/(y2 - y1) * C(x, y1)  +  (y - y1)/(y2 - y1) * C(x, y2)

      where C(x,y) denotes the pixel color at (x,y).

   Arguments:
     image - a 2D numpy array
     upsample_factor - an integer specifying upsample rate

   Returns:
     result - upsampled image, a 2D numpy array with spatial dimension increased
"""
def bilinear_upsampling(image, upsample_factor = 2):
    ##########################################################################
    # TODO: YOUR CODE HERE
    new_image_h = image.shape[0] * upsample_factor
    new_image_w = image.shape[1] * upsample_factor
    new_img = np.zeros((new_image_h, new_image_w))
    for y_idx in range(new_img.shape[0]):
       for x_idx in range(new_img.shape[1]):
         x = x_idx / upsample_factor
         y = y_idx / upsample_factor
         x1 = math.floor(x)
         y1 = math.floor(y)
         x2 = min(math.ceil(x), image.shape[1]-1)
         y2 = min(math.ceil(y), image.shape[0] - 1)
         if x1 == x2 and y1 == y2:
            new_img[y_idx,x_idx] = image[int(y1),int(x1)]
         elif x1 == x2:
            yfloor = image[y1,x1]
            yceil = image[y2,x1]
            new_img[y_idx,x_idx] = yfloor * (y2 - y) + yceil * (y - y1)
         elif y1 == y2:
            xfloor = image[y1,x1]
            xceil = image[y1,x2]
            new_img[y_idx,x_idx] = xfloor * (x2 - x) + xceil * (x - x1)
         else:
            cxy1 = (x2 - x)/(x2 - x1) * image[y1,x1] + (x-x1)/(x2-x1) * image[y1,x2]
            cxy2 = (x2 - x)/(x2 - x1) * image[y2,x1] + (x-x1)/(x2-x1) * image[y2,x2]
            new_img[y_idx,x_idx] = (y2 - y)/(y2 - y1) * cxy1  +  (y - y1)/(y2 - y1) * cxy2

    return new_img

"""
   SOBEL GRADIENT OPERATOR (5 Points)
   Compute an estimate of the horizontal and vertical gradients of an image
   by applying the Sobel operator.
   The Sobel operator estimates gradients dx(horizontal), dy(vertical), of
   an image I as:

         [ 1  0  -1 ]
   dx =  [ 2  0  -2 ] (*) I
         [ 1  0  -1 ]

         [  1  2  1 ]
   dy =  [  0  0  0 ] (*) I
         [ -1 -2 -1 ]

   where (*) denotes convolution.
   Note:
      (1) Your implementation should be as efficient as possible.
      (2) Avoid creating artifacts along the border of the image.
   Arguments:
      image - a 2D numpy array
   Returns:
      dx    - gradient in x-direction at each point
              (a 2D numpy array, the same shape as the input image)
      dy    - gradient in y-direction at each point
              (a 2D numpy array, the same shape as the input image)
"""
def sobel_gradients(image):
   ##########################################################################
   component1 = np.array([1, 2, 1]).reshape((1,3))
   component2 = np.array([-1,0,1]).reshape((1,3))
   dx = conv_2d(image, component1.T, 'mirror')
   dx = conv_2d(dx, component2, 'mirror')

   dy = conv_2d(image, component2.T, 'mirror')
   dy = conv_2d(dy, component1, 'mirror')

   return dx, dy
   ##########################################################################

"""
   NONMAXIMUM SUPPRESSION (10 Points)

   Nonmaximum suppression.

   Given an estimate of edge strength (mag) and direction (theta) at each
   pixel, suppress edge responses that are not a local maximum along the
   direction perpendicular to the edge.

   Equivalently stated, the input edge magnitude (mag) represents an edge map
   that is thick (strong response in the vicinity of an edge).  We want a
   thinned edge map as output, in which edges are only 1 pixel wide.  This is
   accomplished by suppressing (setting to 0) the strength of any pixel that
   is not a local maximum.

   Note that the local maximum check for location (x,y) should be performed
   not in a patch surrounding (x,y), but along a line through (x,y)
   perpendicular to the direction of the edge at (x,y).

   A simple, and sufficient strategy is to check if:
      ((mag[x,y] > mag[x + ox, y + oy]) and (mag[x,y] >= mag[x - ox, y - oy]))
   or
      ((mag[x,y] >= mag[x + ox, y + oy]) and (mag[x,y] > mag[x - ox, y - oy]))
   where:
      (ox, oy) is an offset vector to the neighboring pixel in the direction
      perpendicular to edge direction at location (x, y)

   Arguments:
      mag    - a 2D numpy array, containing edge strength (magnitude)
      theta  - a 2D numpy array, containing edge direction in [0, 2*pi)

   Returns:
      nonmax - a 2D numpy array, containing edge strength (magnitude), where
               pixels that are not a local maximum of strength along an
               edge have been suppressed (assigned a strength of zero)
"""
def get_mag_theta(image):
   dx, dy = sobel_gradients(image)
   mag = np.sqrt(dx*dx +  dy*dy)
   theta = np.arctan2(dy,dx)
   return mag, theta


def nonmax_suppress(mag, theta):
   theta[theta < 0] += np.pi
   suppressed_img = np.zeros((mag.shape))
   for y in range(mag.shape[0] - 1):
      for x in range(mag.shape[1] - 1):
         if theta[y,x] < np.pi / 8 or theta[y,x] >= 7 * np.pi / 8:
            lower = mag[y,x-1]
            higher = mag[y,x+1]
         elif theta[y,x] < 3 * np.pi / 8 and theta[y,x] >= np.pi / 8:
            lower = mag[y - 1, x + 1]
            higher = mag[y + 1, x - 1]
         elif theta[y,x] < 5 * np.pi / 8 and theta[y,x] >= 3* np.pi / 8:
            lower = mag[y+1, x]
            higher = mag[y-1, x]
         elif theta[y,x] < 7 * np.pi / 8 and theta[y,x] >= 5 * np.pi / 8:
            lower = mag[y + 1, x + 1]
            higher = mag[y - 1, x - 1]
         if(mag[y,x] >= higher and mag[y,x] > lower):
            suppressed_img[y,x] = mag[y,x]
   return suppressed_img


"""
   HYSTERESIS EDGE LINKING (10 Points)

   Hysteresis edge linking.

   Given an edge magnitude map (mag) which is thinned by nonmaximum suppression,
   first compute the low threshold and high threshold so that any pixel below
   low threshold will be thrown away, and any pixel above high threshold is
   a strong edge and will be preserved in the final edge map.  The pixels that
   fall in-between are considered as weak edges.  We then add weak edges to
   true edges if they connect to a strong edge along the gradient direction.

   Since the thresholds are highly dependent on the statistics of the edge
   magnitude distribution, we recommend to consider features like maximum edge
   magnitude or the edge magnitude histogram in order to compute the high
   threshold.  Heuristically, once the high threshod is fixed, you may set the
   low threshold to be propotional to the high threshold.

   Note that the thresholds critically determine the quality of the final edges.
   You need to carefully tuned your threshold strategy to get decent
   performance on real images.

   For the edge linking, the weak edges caused by true edges will connect up
   with a neighbouring strong edge pixel.  To track theses edges, we
   investigate the 8 neighbours of strong edges.  Once we find the weak edges,
   located along strong edges' gradient direction, we will mark them as strong
   edges.  You can adopt the same gradient checking strategy used in nonmaximum
   suppression.  This process repeats util we check all strong edges.

   In practice, we use a queue to implement edge linking.  In python, we could
   use a list and its fuction .append or .pop to enqueue or dequeue.

   Arguments:
     nonmax - a 2D numpy array, containing edge strength (magnitude) which is thined by nonmaximum suppression
     theta  - a 2D numpy array, containing edeg direction in [0, 2*pi)

   Returns:
     edge   - a 2D numpy array, containing edges map where the edge pixel is 1 and 0 otherwise.
"""

def hysteresis_edge_linking(nonmax, theta):
   ##########################################################################
   # TODO: YOUR CODE HERE
   raise NotImplementedError('hysteresis_edge_linking')
   ##########################################################################
   return edge

"""
   CANNY EDGE DETECTOR (5 Points)

   Canny edge detector.

   Given an input image:

   (1) Compute gradients in x- and y-directions at every location using the
       Sobel operator.  See sobel_gradients() above.

   (2) Estimate edge strength (gradient magnitude) and direction.

   (3) Run (1)(2) on downsampled images with multiple factors and
       then combine the results via upsampling to original resolution.

   (4) Perform nonmaximum suppression of the edge strength map, thinning it
       in the direction perpendicular to that of a local edge.
       See nonmax_suppress() above.

   (5) Compute the high threshold and low threshold of edge strength map
       to classify the pixels as strong edges, weak edges and non edges.
       Then link weak edges to strong edges

   Return the original edge strength estimate (max), the edge
   strength map after nonmaximum suppression (nonmax) and the edge map
   after edge linking (edge)

   Arguments:
      image             - a 2D numpy array
      downsample_factor - a list of interger

   Returns:
      mag      - a 2D numpy array, same shape as input, edge strength at each pixel
      nonmax   - a 2D numpy array, same shape as input, edge strength after nonmaximum suppression
      edge     - a 2D numpy array, same shape as input, edges map where edge pixel is 1 and 0 otherwise.
"""
def canny(image, downsample_factor = [1]):
   ##########################################################################
   # TODO: YOUR CODE HERE
   raise NotImplementedError('canny')
   ##########################################################################
   return mag, nonmax, edge
