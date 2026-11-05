#Example 7.23 Pixellib Instance Segment
#pip install pixellib
import pixellib
from pixellib.instance import instance_segmentation
import cv2
segment_frame = instance_segmentation()
segment_frame.load_model("mask_rcnn_coco.h5")
image = cv2.imread("strawberrys2.jpeg")
cv2.imshow("Original", image)
segmask, output = segment_frame.segmentFrame(image)
cv2.imshow("Instance Segment", output)
cv2.waitKey(0)
cv2.destroyAllWindows()