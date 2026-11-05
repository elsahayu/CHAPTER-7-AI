#Example 7.31 OpenCV Background Substractor
# importing libraries
import numpy as np
import cv2

# creating background subtractor object
fgbg = cv2.bgsegm.createBackgroundSubtractorMOG();
#fgbg = cv2.createBackgroundSubtractorMOG2();
#fgbg = cv2.bgsegm.createBackgroundSubtractorGMG();
#cap = cv2.VideoCapture(0);
cap = cv2.VideoCapture('vtest.avi')
while(1):
    # read frames
    ret, img = cap.read()
    if not ret:
        break
    fgmask = fgbg.apply(img)
    output = cv2.bitwise_and(img, img, mask=fgmask)


    cv2.imshow("Video", output)

    k = cv2.waitKey(30) & 0xff;
    if k == 27:
        break;

cap.release();
cv2.destroyAllWindows();