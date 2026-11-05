# https://www.geeksforgeeks.org/background-subtraction-opencv/?ref=rp
# OpenCV Background Substraction
# Worked on TensorFlow 2.0 23/10/2020
# importing libraries
import numpy as np
import cv2

# creating object
fgbg1 = cv2.bgsegm.createBackgroundSubtractorMOG();
fgbg2 = cv2.createBackgroundSubtractorMOG2();
fgbg3 = cv2.bgsegm.createBackgroundSubtractorGMG();
# Kernel for denoise
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3));
#Create a tracker bars===========================================
def nothing(x):
 # any operation
    pass
cv2.namedWindow("Trackbars")
cv2.createTrackbar("Alpha", "Trackbars", 4, 10, nothing)

# capture frames from a camera or a video ======================
#cap = cv2.VideoCapture(0);
cap = cv2.VideoCapture('vtest.avi')
#load background image
bg = cv2.imread("londoneye.jpg")
#cv2.imshow("background", bg)
while(1):
    # read frames
    ret, img = cap.read()
    if not ret:
        break


    # apply mask for background subtraction =====================
    fgmask1 = fgbg1.apply(img);
    fgmask2 = fgbg2.apply(img);
    fgmask3 = fgbg3.apply(img);
    # apply transformation to remove noise ======================
    fgmask4 = cv2.morphologyEx(fgmask3, cv2.MORPH_OPEN, kernel);
    #Bitwise AND the image and mask
    output = cv2.bitwise_and(img, img, mask=fgmask1)
    #Change the background=======================================
    height,width,depth = output.shape
    dim = (width, height)
    # resize background image
    bg = cv2.resize(bg, dim, interpolation = cv2.INTER_AREA)
    fgmask1a = cv2.bitwise_not(fgmask1)
    bg2 = cv2.bitwise_and(bg, bg, mask=fgmask1a)

    alpha=cv2.getTrackbarPos("Alpha", "Trackbars")/10
    output2 = cv2.addWeighted(output,alpha,bg2,1-alpha,0)
    output3 = cv2.add(output,bg2)
    #Display the images ==========================================
    res1 = np.hstack((fgmask1, fgmask2,fgmask3, fgmask4))
    res2 = np.hstack((img, output,output2,output3))

    cv2.namedWindow("Mask", 0)
    cv2.resizeWindow("Mask", 640, 480)
    cv2.imshow("Mask", res1)

    cv2.namedWindow("Video", 0)
    cv2.resizeWindow("Video", 640, 480)
    cv2.imshow("Video", res2)

    k = cv2.waitKey(30) & 0xff;
    if k == 27:
        break;

cap.release();
cv2.destroyAllWindows(); 