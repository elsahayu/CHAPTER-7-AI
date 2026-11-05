# Example 7.17 webcam ObjectTracking ROI
# Modified based on:
# https://www.learnopencv.com/object-tracking-using-opencv-cpp-python/
import cv2
import sys

# OpenCV Trackers:
# tracker = cv2.TrackerBoosting_create()
tracker = cv2.TrackerMIL_create()
# tracker = cv2.TrackerKCF_create()
# tracker = cv2.TrackerTLD_create()
# tracker = cv2.TrackerMedianFlow_create()
# tracker = cv2.TrackerGOTURN_create()
# tracker = cv2.TrackerMOSSE_create()
# tracker = cv2.TrackerCSRT_create()

# Open Webcam and select a ROI box
video = cv2.VideoCapture(0)
ok, frame = video.read()
bbox = cv2.selectROI(frame, True)

# Initialize tracker with first frame and bounding box
ok = tracker.init(frame, bbox)

while True:
    ok, frame = video.read()

    # Start timer
    timer = cv2.getTickCount()

    if bbox is not None:
        ok, bbox = tracker.update(frame)

    # Calculate Frames per second (FPS)
    fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)

    # Draw bounding box
    if ok:
        p1 = (int(bbox[0]), int(bbox[1]))
        p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
        cv2.rectangle(frame, p1, p2, (0, 255, 0), 2, 1)
    else:
        cv2.putText(
            frame,
            "Tracking failure detected",
            (100, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 0, 255),
            2,
        )

    # Display FPS on frame
    cv2.putText(
        frame,
        "FPS : " + str(int(fps)),
        (100, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (50, 170, 50),
        2,
    )

    # Display result
    cv2.imshow("Tracking", frame)

    # Exit if ESC pressed
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break

video.release()
# close all windows
cv2.destroyAllWindows()