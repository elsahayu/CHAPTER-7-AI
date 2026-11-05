#Example 7.25 K-means Clustering for Image Segmentation
import numpy as np
import matplotlib.pyplot as plt
import cv2
# Load image and Change RGB color
image = cv2.imread('strawberrys2.jpeg')
cv2.imshow('Original',image)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# Reshaping the image and Convert to float type
pixel_vals = np.float32(image.reshape((-1,3)))
# Criteria for the algorithm to stop: 100 iterations or 85% accuracy
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100,
0.85)

# K-means clustering with 3 clusters and random initial centres
k = 3
retval, labels, centers = cv2.kmeans(pixel_vals, k, None, criteria,
10, cv2.KMEANS_RANDOM_CENTERS)

# convert data into 8-bit values
centers = np.uint8(centers)
segmented_data = centers[labels.flatten()]

# reshape data into the original image dimensions
segmented_image = segmented_data.reshape((image.shape))
segmented_image = cv2.cvtColor(segmented_image, cv2.COLOR_RGB2BGR)
cv2.imshow('Segmented', segmented_image)

cv2.waitKey(0)
cv2.destroyAllWindows()