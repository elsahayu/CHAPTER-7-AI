#Example 7.20 Pixellib Semantic Segmentation
#pip install pixellib
import pixellib
from pixellib.semantic import semantic_segmentation
import cv2
segment_image = semantic_segmentation()
segment_image.load_pascalvoc_model("deeplabv3_xception_tf_dim_ordering_tf_kernels.h5")
f = "city0.jpeg"
segment_image.segmentAsPascalvoc(f, output_image_name = "output.jpg",
overlay = True)
image = cv2.imread(f)
cv2.imshow("Original", image)
output = cv2.imread("output.jpg")
cv2.imshow("Semantic Segment", output)
cv2.waitKey(0)
cv2.destroyAllWindows()