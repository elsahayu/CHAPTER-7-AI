# Example 7.28 Scikit Watershed Webcam
import numpy as np
from skimage import data, util, filters, color
from skimage.segmentation import watershed
import matplotlib.pyplot as plt
import cv2

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
        
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    edges = filters.sobel(image)
    grid = util.regular_grid(image.shape, n_points=468)
    seeds = np.zeros(image.shape, dtype=int)
    seeds[grid] = np.arange(seeds[grid].size).reshape(seeds[grid].shape) + 1
    w0 = watershed(edges, seeds)
    image0 = color.label2rgb(w0, image, bg_label=-1)
    cv2.imshow('Classical watershed', image0)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()