#Example 7.21 Pixellib Semantic Segmentation – Webcam Feed
#Model download
#https://github.com/ayoolaolafenwa/PixelLib/releases/download/1.1/deeplabv3_xception_tf_dim_ordering_tf_kernels.h5
import pixellib
from pixellib.semantic import semantic_segmentation
import cv2
capture = cv2.VideoCapture(0)
segment_video = semantic_segmentation()
segment_video.load_pascalvoc_model("deeplabv3_xception_tf_dim_ordering_tf_kernels.h5")
#Webcam
segment_video.process_camera_pascalvoc(capture, overlay = True,
frames_per_second= 15, output_video_name="output_video.mp4", show_frames= True,frame_name= "frame", check_fps = True)
#Video
#segment_video.process_video("cityscene.mp4", show_bboxes = True,frames_per_second= 15, output_video_name="output_video.mp4")