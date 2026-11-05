#Example 7.27 Watershed Image Segmentation with using Scikit-image
import numpy as np
from skimage import data, util, filters, color
from skimage.segmentation import watershed
import matplotlib.pyplot as plt
import cv2
image = cv2.imread('skin.jpeg',0)
edges = filters.sobel(image)
grid = util.regular_grid(image.shape, n_points=200)
seeds = np.zeros(image.shape, dtype=int)
seeds[grid] = np.arange(seeds[grid].size).reshape(seeds[grid].shape) + 1
w0 = watershed(edges, seeds)
image0 = color.label2rgb(w0, image, bg_label=-1)
cv2.imshow('Classical watershed', image0)
cv2.waitKey(0)
cv2.destroyAllWindows()