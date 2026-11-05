# Example 7.26 Watershed Image Segmentation with OpenCV
# Modified based on:
# https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_watershed/py_watershed.html
import numpy as np
import cv2
from matplotlib import pyplot as plt
import urllib.request as urllib
# from urllib.request import Request, urlopen

# get image by url
# req = urllib.Request('https://opencv-python-tutroals.readthedocs.io/en/latest/_images/water_coins.jpg', headers={'User-Agent': 'Mozilla/5.0'})
# req = urllib.Request('https://images.wisegeek.com/skin-mole.jpg', headers={'User-Agent': 'Mozilla/5.0'})
# req = urllib.Request('http://www.anti-aging-skin-care-illusions.com/images/Dry-Skin.jpg', headers={'User-Agent': 'Mozilla/5.0'})
# req = urllib.Request('https://jooinn.com/images/human-skin-4.jpg', headers={'User-Agent': 'Mozilla/5.0'})
# req = urllib.Request('https://images.pexels.com/photos/4046561/pexels-photo-4046561.jpeg?auto=compress&cs=tinysrgb&dpr=2&h=650&w=940', headers={'User-Agent': 'Mozilla/5.0'})
req = urllib.Request('https://images.pexels.com/photos/4046567/pexels-photo-4046567.jpeg?auto=compress&cs=tinysrgb&dpr=2&h=650&w=940', headers={'User-Agent': 'Mozilla/5.0'})
# req = urllib.Request('https://images.pexels.com/photos/2709386/pexels-photo-2709386.jpeg?auto=compress&cs=tinysrgb&dpr=3&h=750&w=1260', headers={'User-Agent': 'Mozilla/5.0'})
# req = urllib.Request('https://images.pexels.com/photos/2683373/pexels-photo-2683373.jpeg?auto=compress&cs=tinysrgb&dpr=1&w=500', headers={'User-Agent': 'Mozilla/5.0'})
# req = urllib.Request('https://images.pexels.com/photos/952360/pexels-photo-952360.jpeg?auto=compress&cs=tinysrgb&dpr=2&h=650&w=940', headers={'User-Agent': 'Mozilla/5.0'})

resp = urllib.urlopen(req)
image = np.asarray(bytearray(resp.read()), dtype="uint8")
img = cv2.imdecode(image, cv2.IMREAD_COLOR)
img = cv2.resize(img, (520, 480), interpolation=cv2.INTER_AREA)
cv2.imshow('Original', img)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
# thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
# thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
cv2.imshow('Thresh', thresh)

# noise removal
kernel = np.ones((3, 3), np.uint8)
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

# sure background area
sure_bg = cv2.dilate(opening, kernel, iterations=3)
cv2.imshow('Sure BG', sure_bg)

# Finding sure foreground area
dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
cv2.imshow('Sure FG', sure_fg)

# Finding unknown region
sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg, sure_fg)

# Marker labelling
ret, markers = cv2.connectedComponents(sure_fg)

# Add one to all labels so that sure background is not 0, but 1
markers = markers + 1

# Now, mark the region of unknown with zero
markers[unknown == 255] = 0

markers = cv2.watershed(img, markers)
img[markers == -1] = [255, 0, 0]
cv2.imshow('Watershed', img)

cv2.waitKey(0)
cv2.destroyAllWindows()