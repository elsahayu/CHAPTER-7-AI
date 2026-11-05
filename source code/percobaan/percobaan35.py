# Example 7.35 Webcam Feed BGRemoval with OpenCV
import keyboard  # using module keyboard
import cv2
import numpy as np
import imutils

# == Parameters
MASK_COLOR = (0.0, 0.0, 1.0)  # In BGR format


# -- Get the background mask
def getBackgroundMask(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # -- Edge detection
    edges = cv2.Canny(gray, 10, 100)
    edges = cv2.dilate(edges, None)
    edges = cv2.erode(edges, None)
    
    # -- Find contours in edges, sort by area
    contour_info = []
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    for c in contours:
        contour_info.append((
            c,
            cv2.isContourConvex(c),
            cv2.contourArea(c),
        ))
        
    if not contour_info:
        return np.zeros(img.shape, dtype=np.uint8)

    contour_info = sorted(contour_info, key=lambda c: c[2], reverse=True)
    max_contour = contour_info[0]

    # -- Create empty mask, draw filled polygon on it corresponding to largest contour
    # Mask is black, polygon is white
    mask = np.zeros(edges.shape)
    cv2.fillConvexPoly(mask, max_contour[0], (255))
    
    # -- Smooth mask, then blur it
    mask = cv2.dilate(mask, None, iterations=10)
    mask = cv2.erode(mask, None, iterations=10)
    mask = cv2.GaussianBlur(mask, (21, 21), 0)
    mask_stack = np.dstack([mask] * 3)  # Create 3-channel alpha mask
    return mask_stack


camera = cv2.VideoCapture(0)
bolMask = False

while camera.isOpened():
    ok, img = camera.read()
    if not ok:
        break

    # if key 't' is pressed, bolMask = not bolMask
    if keyboard.is_pressed('t'):
        bolMask = not bolMask
        cv2.waitKey(200)  # Debounce agar tidak berganti terlalu cepat

    if bolMask:
        mask_stack = getBackgroundMask(img)
        # -- Blend masked img into MASK_COLOR background
        mask_stack = mask_stack.astype('float32') / 255.0
        img = img.astype('float32') / 255.0
        masked = (mask_stack * img) + ((1 - mask_stack) * MASK_COLOR)
        img = (masked * 255).astype('uint8')

    cv2.imshow('video image', img)
    key = cv2.waitKey(30)
    if key == 27:  # press 'ESC' to quit
        break

camera.release()
cv2.destroyAllWindows()