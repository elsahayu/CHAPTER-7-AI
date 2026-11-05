# Example 7.10 Object Detection with MaskRCNN
# Modified based on:
# https://machinelearningmastery.com/how-to-perform-object-detectionin-photographs-with-mask-r-cnn-in-keras/
# Example of inference with a pre-trained coco model: download from:
# https://github.com/matterport/Mask_RCNN/releases/download/v2.0/mask_rcnn_coco.h5

import os
import cv2
import numpy as np
import imutils
import colorsys
import random
from matplotlib import pyplot
from matplotlib.patches import Rectangle
from keras.preprocessing.image import load_img, img_to_array

from mrcnn.config import Config
from mrcnn.model import MaskRCNN
from mrcnn import model as modellib
from mrcnn import visualize

# Daftar nama kelas COCO Dataset (80 kelas + 1 background)
CLASS_NAMES = [
    'BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
    'bus', 'train', 'truck', 'boat', 'traffic light',
    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
    'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
    'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
    'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard',
    'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
    'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
    'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
    'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
    'teddy bear', 'hair drier', 'toothbrush'
]

# Generate warna acak yang berbeda untuk setiap kelas
hsv = [(i / len(CLASS_NAMES), 1, 1.0) for i in range(len(CLASS_NAMES))]
COLORS = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
random.seed(42)
random.shuffle(COLORS)

# Konfigurasi pengujian
class TestConfig(Config):
    NAME = "test"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + 80

# Inisialisasi model Mask R-CNN
rcnn = MaskRCNN(mode='inference', model_dir='./', config=TestConfig())

# Memuat bobot model COCO
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(BASE_DIR, 'mrcnn', 'mask_rcnn_coco.h5') if os.path.exists(os.path.join(BASE_DIR, 'mrcnn', 'mask_rcnn_coco.h5')) else 'mask_rcnn_coco.h5'
rcnn.load_weights(model_path, by_name=True)

# Membaca dan mempersiapkan gambar
image = cv2.imread("./apple.jpeg")
if image is None:
    # Fallback jika apple.jpeg belum ada
    image = cv2.imread("./fruits.jpeg") if os.path.exists("./fruits.jpeg") else cv2.imread("./ball.jpeg")

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = imutils.resize(image, width=512)

# Melakukan inferensi/deteksi
r = rcnn.detect([image], verbose=1)[0]

# Visualisasi mask piksel objek
for i in range(0, r["rois"].shape[0]):
    classID = r["class_ids"][i]
    mask = r["masks"][:, :, i]
    color = COLORS[classID][::-1]
    image = visualize.apply_mask(image, mask, color, alpha=0.5)

image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

# Menggambar bounding box, label kelas, dan skor akurasi
for i in range(0, len(r["scores"])):
    (startY, startX, endY, endX) = r["rois"][i]
    classID = r["class_ids"][i]
    label = CLASS_NAMES[classID]
    score = r["scores"][i]
    color = [int(c) for c in np.array(COLORS[classID]) * 255]

    cv2.rectangle(image, (startX, startY), (endX, endY), color, 2)
    text = "{}: {:.3f}".format(label, score)
    y = startY - 10 if startY - 10 > 10 else startY + 10
    cv2.putText(
        image, text, (startX, y),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
    )

# Menampilkan hasil gambar
cv2.imshow('image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()