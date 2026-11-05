#Example 7.1 Circle Detection
import cv2
img = cv2.imread("ball.jpeg")
width,height,c=img.shape
print(img.shape)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.medianBlur(gray, 5)
circles = cv2.HoughCircles(gray,cv2.HOUGH_GRADIENT,1,20,param1=50,param2=30,minRadius=35,maxRadius=50)
for co, i in enumerate(circles[0, :], start=1):
    cv2.circle(img,(int(i[0]),int(i[1])),int(i[2]),(0,255,0),2)
    cv2.circle(img,(int(i[0]),int(i[1])),2,(0,0,255),3)
print("Number of circles detected:", co)
cv2.imshow('Circle Detection',img)
