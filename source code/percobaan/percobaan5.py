# Example 7.5 YOLOv3 Detection
import numpy as np
import argparse
import cv2 as cv
import subprocess
import time
import os
from yolo_utils import infer_image, show_image
from types import SimpleNamespace

# Path folder yolov3-coco yang berada di ../yolov3-coco
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
YOLO_DIR = os.path.join(BASE_DIR, 'yolov3-coco')

d = {
    'confidence': 0.5,
    'threshold': 0.3,
    'weights': os.path.join(YOLO_DIR, 'yolov3.weights'),
    'config': os.path.join(YOLO_DIR, 'yolov3.cfg'),
    'show_time': False
}
FLAGS = SimpleNamespace(**d)
cocolabels = os.path.join(YOLO_DIR, 'coco-labels.txt')
labels = open(cocolabels).read().strip().split('\n')
print(FLAGS)

colors = np.random.randint(0, 255, size=(len(labels), 3), dtype='uint8')
net = cv.dnn.readNetFromDarknet(FLAGS.config, FLAGS.weights)
layer_names = net.getLayerNames()
layer_names = [layer_names[int(i) - 1] for i in net.getUnconnectedOutLayers()]

vid = cv.VideoCapture(0)
while True:
    _, frame = vid.read()
    if frame is None:
        break
    frame = cv.resize(frame, (640, 360), interpolation=cv.INTER_AREA)
    height, width = frame.shape[:2]
    frame, boxes, confidences, classids, idxs = infer_image(
        net, layer_names, height, width, frame, colors, labels, FLAGS
    )
    cv.imshow('webcam', frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

vid.release()
cv.destroyAllWindows()