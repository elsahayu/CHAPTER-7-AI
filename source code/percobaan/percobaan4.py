#Example 7.4 Haar Cascade Code Detection
import time
import numpy as np
import cv2
classifier = cv2.CascadeClassifier('haarcascade_fullbody.xml')
cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Pass frame to the classifier
    bodys = classifier.detectMultiScale(gray, 1.5, 10)

    # Extract bounding boxes for any bodies identified
    for (x,y,w,h) in bodys:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
    cv2.imshow('Body', frame)
    if cv2.waitKey(1) == 13: #13 is the Enter Key
        break
cap.release()
cv2.destroyAllWindows()