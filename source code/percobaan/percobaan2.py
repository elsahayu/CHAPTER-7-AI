#Example 7.2 Shape Detection
import cv2
import numpy as np
img = cv2.imread("shapes1.jpeg")
width,height,c=img.shape
print(img.shape)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.medianBlur(gray, 5)
ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
cnts, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2
.CHAIN_APPROX_SIMPLE)
cnts = np.array(cnts,dtype=object)[[cv2.contourArea(c)>10 for c in cnts]]
for c in cnts:
    area = cv2.contourArea(c)
    if area > 10:
        cv2.drawContours(img, [c], -1, (0, 255, 0), 2)
        M = cv2.moments(c)
        if M["m00"] != 0:
            cX = int((M["m10"] / M["m00"]) )
            cY = int((M["m01"] / M["m00"]) )
            # Find the Center of a Blob (detected contour)
        cv2.circle(img, (cX,cY), 2, (255, 255,0), 2)
cv2.imshow("Output", img)
cv2.waitKey(0)