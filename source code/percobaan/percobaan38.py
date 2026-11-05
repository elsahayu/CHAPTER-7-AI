# Example 7.38 Harry's Cloak (Invisible Cloak)
import cv2
import numpy as np
import time

print(cv2.__version__)

# Select Webcam or a video
# capture_video = cv2.VideoCapture("video1.mp4")
capture_video = cv2.VideoCapture(0)

# Warm up the Webcam
time.sleep(1)
count = 0
background = None

# Get the background
for i in range(30):
    return_val, background = capture_video.read()
    if not return_val:
        continue
    background = np.flip(background, axis=1)

# Start the Webcam
while capture_video.isOpened():
    return_val, img = capture_video.read()
    if not return_val:
        break

    count = count + 1
    img = np.flip(img, axis=1)

    # Convert the image from BGR to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Select the Red color range in HSV space
    lower_red = np.array([0, 120, 70])
    upper_red = np.array([10, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red, upper_red)

    # mask2 for Red color (wrapping around 180 in HSV)
    lower_red = np.array([170, 120, 70])
    upper_red = np.array([180, 255, 255])
    mask2 = cv2.inRange(hsv, lower_red, upper_red)

    # Apply the masks to image
    mask1 = mask1 + mask2
    mask1 = cv2.morphologyEx(mask1, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=2)
    mask1 = cv2.dilate(mask1, np.ones((3, 3), np.uint8), iterations=1)
    mask2 = cv2.bitwise_not(mask1)

    # Final Output
    res1 = cv2.bitwise_and(background, background, mask=mask1)
    res2 = cv2.bitwise_and(img, img, mask=mask2)
    final_output = cv2.addWeighted(res1, 1, res2, 1, 0)
    cv2.imshow("INVISIBLE CLOAK", final_output)

    k = cv2.waitKey(30) & 0xFF
    if k == 27:  # Press ESC to exit
        break

capture_video.release()
cv2.destroyAllWindows()
