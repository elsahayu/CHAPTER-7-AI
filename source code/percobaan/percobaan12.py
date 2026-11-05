# Example 7.12 Object Detection from live webcam feed with Gluon
# Modified based on:
# https://cv.gluon.ai/build/examples_detection/demo_webcam.html
import gluoncv as gcv
import cv2
import mxnet as mx
# Load the model
net = gcv.model_zoo.get_model('ssd_512_mobilenet1.0_voc', pretrained=True)
# Compile the model for faster speed
net.hybridize()
# Load the webcam handler
cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = mx.nd.array(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).astype('uint8')
    rgb_nd, frame = gcv.data.transforms.presets.ssd.transform_test(frame, short=512, max_size=700)
    class_IDs, scores, bounding_boxes = net(rgb_nd)

    # Display the result
    img = gcv.utils.viz.cv_plot_bbox(
        frame, bounding_boxes[0], scores[0], class_IDs[0], class_names=net.classes
    )
    gcv.utils.viz.cv_plot_image(img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()