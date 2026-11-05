# Example 7.36 Webcam background blending with video
import cv2
import numpy as np
import time

# get captures
cap = cv2.VideoCapture(0)
background_capture = cv2.VideoCapture(r'./londoneye.mp4')
# background = cv2.imread('green.jpg')
counter = -1

while cap.isOpened():
    counter += 1
    start_time_extract_figure = time.time()
    
    # extract your figure
    ret_cap, frame = cap.read()
    if not ret_cap:
        break
        
    frame = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_AREA)
    cv2.imshow('original', frame)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mask = np.zeros(frame.shape[:2], np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    rect = (10, 10, 300, 400)
    
    start_time_grabCut = time.time()
    cv2.grabCut(frame, mask, rect, bgdModel, fgdModel, 1, cv2.GC_INIT_WITH_RECT)
    during_time_grabCut = time.time() - start_time_grabCut
    print('{}-th t_time: {}'.format(counter, during_time_grabCut))
    
    mask2 = np.where((mask == 2) | (mask == 0), (0,), (1,)).astype('uint8')
    frame = frame * mask2[:, :, np.newaxis]
    elapsed_time_extract_figure = time.time() - start_time_extract_figure
    print('{}-th extract_figure_time: {}'.format(counter, elapsed_time_extract_figure))
    
    # extract the background
    start_time_combination = time.time()
    ret, background = background_capture.read()
    if not ret:
        background_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, background = background_capture.read()
        if not ret:
            break
            
    background = cv2.resize(background, (640, 480), interpolation=cv2.INTER_AREA)
    
    # combine the figure and background using mask instead of iteration
    mask_1 = frame > 0
    mask_2 = frame <= 0
    combination = (cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) * mask_1 + background * mask_2)
    elapsed_time_combination = time.time() - start_time_combination
    print('{}-th combination_time: {}'.format(counter, elapsed_time_combination))
    
    cv2.imshow('combination', combination)
    
    k = cv2.waitKey(30) & 0xFF
    if k == 27:
        break

cap.release()
background_capture.release()
cv2.destroyAllWindows()