# Example 7.19 YOLOv4_DeepSORT on Google Colab
# Source: https://github.com/theAIGuysCode/yolov4-deepsort
# Fork: https://github.com/PerryXiao2015/yolov4-deepsort

# 1. Clone repository
# !git clone https://github.com/theAIGuysCode/yolov4-deepsort
!git clone https://github.com/PerryXiao2015/yolov4-deepsort
%cd yolov4-deepsort

# 2. Install dependencies (TensorFlow GPU)
!pip install -r requirements-gpu.txt

# 3. Download YOLOv4 full weights
!curl -L "https://drive.google.com/u/0/uc?export=download&confirm=T6-x&id=1cewMfusmPjYWbrnuJRuKhPMwRe_b9PaT" > yolov4.weights
!mv yolov4.weights ./data/

# 4. Convert darknet weights to tensorflow model
!python save_model.py --model yolov4

# 5. Run YOLOv4 DeepSORT object tracker on video
!python object_tracker.py --video ./data/video/car.mp4 --output ./outputs/car.avi --model yolov4

# Run YOLOv4 DeepSORT object tracker on webcam (set video flag to 0)
# !python object_tracker.py --video 0 --output ./outputs/webcam.avi --model yolov4

# 6. Download YOLOv4-Tiny weights
!wget https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.weights
!mv yolov4-tiny.weights ./data/

# 7. Environment & Colab display setup
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from google.colab.patches import cv2_imshow
# cv2_imshow(img)

# 8. Save YOLOv4-Tiny model
!python save_model.py --weights ./data/yolov4-tiny.weights --output ./checkpoints/yolov4-tiny-416 --model yolov4 --tiny

# 9. Run YOLOv4-Tiny object tracker
!python object_tracker.py --weights ./checkpoints/yolov4-tiny-416 --model yolov4 --video ./data/video/cars.mp4 --output ./outputs/cartiny.avi --tiny