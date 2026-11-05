#Example 7.2a Identifying and Drawing Contours with OpenCV
import cv2
import numpy as np

image = cv2.imread('shapes1.jpeg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
edged = cv2.Canny(gray, 30, 200)
contours, hierarchy = cv2.findContours(edged,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

print("Number of Contours found = " + str(len(contours)))

# contourIdx=-1signifies drawing all contours 3: thickness
cv2.drawContours(image=image, contours=contours, contourIdx=-1,color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
cv2.imshow('Contours', image)
cv2.waitKey(0)