#Example 7.32 OpenCV Background Substractor
# importing libraries
import numpy as np
import cv2

# creating background subtractor object
fgbg = cv2.bgsegm.createBackgroundSubtractorMOG();
#fgbg = cv2.createBackgroundSubtractorMOG2();
#fgbg = cv2.bgsegm.createBackgroundSubtractorGMG();
# Kernel for denoise
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3));
#cap = cv2.VideoCapture(0);
cap = cv2.VideoCapture('vtest.avi')
bg=cv2.imread("londoneye.jpeg")
while(1):
    # read frames
    ret, img = cap.read()
    if not ret:
        break
    fgmask = fgbg.apply(img)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)

    output = cv2.bitwise_and(img, img, mask=fgmask)
    #Change the background=======================================
    height,width,depth = output.shape
    dim = (width, height)
    bg = cv2.resize(bg, dim, interpolation = cv2.INTER_AREA)
    fgmask1a = cv2.bitwise_not(fgmask)
    bg2 = cv2.bitwise_and(bg, bg, mask=fgmask1a)
    output = cv2.add(output,bg2)
    cv2.imshow("Video", output)

    k = cv2.waitKey(30) & 0xff;
    if k == 27:
        break;

cap.release();
cv2.destroyAllWindows();