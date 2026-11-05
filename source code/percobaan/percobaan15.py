# Example 7.15
# Use a standard A4 paper (210 x 297 mm), at 1040 mm distance,
# to calculate the Camera Focal Length
import cv2
import imutils

# Sideway A4 paper size in mm
paperHeight = 210
paperWidth = 297

# Distance of A4 paper to camera in mm
distance = 1040


def find_paper(image):
    x, y, w, h = -1, -1, -1, -1
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    mask = cv2.Canny(gray, 35, 125)
    cv2.imshow("Mask", mask)

    # ret, mask = cv2.threshold(gray, 127, 255, 0)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    if len(contours) > 0:
        c = max(contours, key=cv2.contourArea)
        cv2.drawContours(image, [c], 0, (0, 255, 0), 3)
        area = cv2.contourArea(c)
        # print(area)
        c = cv2.boundingRect(c)
        x, y, w, h = int(c[0]), int(c[1]), int(c[2]), int(c[3])

    return x, y, w, h


def getFocalLength(image, KNOWN_DISTANCE, KNOWN_WIDTH):
    focalLength = -1
    x, y, w, h = find_paper(image)
    focalLength = (w * KNOWN_DISTANCE) / KNOWN_WIDTH
    return x, y, w, h, focalLength


# Capture frame from a video
cap = cv2.VideoCapture(0)

while True:
    # Reads frame from webcam
    ret, frame = cap.read()
    if not ret or frame is None:
        print("Gagal membaca webcam.")
        break

    x, y, w, h, f = getFocalLength(frame, distance, paperWidth)

    if f > 0:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        text = f"Width: {w:0.1f} Height: {h:0.1f} pixels"
        cv2.putText(frame, text, (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        text = f"Focal Length: {f:0.1f}mm"
        cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Display frame in a window
    cv2.imshow("Webcam", frame)

    # Wait for Esc key to stop
    if cv2.waitKey(33) == 27:
        break

# Close the camera
cap.release()

# De-allocate any associated memory usage
cv2.destroyAllWindows()