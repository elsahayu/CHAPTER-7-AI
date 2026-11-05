#Example 7.24 Pixellib Instance Webcam Feed
import time
import pixellib
from pixellib.instance import instance_segmentation
import cv2
segment_frame = instance_segmentation()
segment_frame.load_model("mask_rcnn_coco.h5")
capture = cv2.VideoCapture(0)
while True:
    start = time.time()
    ret, frame = capture.read()
    segmask, output =segment_frame.segmentFrame(frame)
    # Display result
    cv2.imshow("Webcam", output)
    # Exit if ESC pressed
    k = cv2.waitKey(1) & 0xff
    if k == 27 :
        break
    end = time.time()
    print(f"Inference Time: {end-start:.2f}seconds")