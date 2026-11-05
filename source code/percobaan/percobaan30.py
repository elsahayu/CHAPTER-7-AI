#Example 7.30 Image segmentation by using SNOW filter from Porespy
#https://www.researchgate.net/post/Watershed-segmentationimplementation-using-scikit-image
#pip install porespy
import porespy as ps
import os
import matplotlib.pyplot as plt
import cv2
import numpy as np
# Import image
file="blueberrys.jpeg"
gray= cv2.imread(file,0)
plt.imshow(gray)
plt.show()
# ret, thresh = cv2.threshold(gray, 164, 255, cv2.THRESH_BINARY)
thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
# cv2.imshow('Thresh', thresh)

plt.imshow(thresh)
plt.show()
# Perform SNOW on Binarized image
watershed = ps.filters.snow_partitioning(thresh > 0, randomize=True,return_all=True)
# Display and Load
plt.imshow(watershed.regions)
plt.show()
