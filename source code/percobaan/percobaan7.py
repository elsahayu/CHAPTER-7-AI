# Example 7.7 Object Detection with ImageAI
# https://github.com/OlafenwaMoses/ImageAI
# Only works for TensorFlow 1.X
# pip install imageai

from imageai.Detection import ObjectDetection
import cv2
import time

t0 = time.time()
detector = ObjectDetection()
detector.setModelTypeAsRetinaNet()
detector.setModelPath("resnet50_coco_best_v2.0.1.h5")
detector.loadModel()
detections = detector.detectObjectsFromImage("fruits.jpeg",output_image_path="imagenew.jpg")
print(detections)
t1 = time.time() - t0
print(t1)