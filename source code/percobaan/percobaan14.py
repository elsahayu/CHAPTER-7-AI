# Example 7.14 Training Haar Cascade Code – Testing the Trained Code
import cv2
import os

# Path model cascade
face_cascade = cv2.CascadeClassifier(r"./classifier/cascade.xml")

# Path gambar (gunakan Otter_6336.jpeg jika ada, atau fallback ke gambar di folder p/)
image_path = "Otter_6336.jpeg"
if not os.path.exists(image_path):
    fallback_path = os.path.join("p", "000002.jpg")
    if os.path.exists(fallback_path):
        print(f"[INFO] '{image_path}' tidak ditemukan, menggunakan fallback: '{fallback_path}'")
        image_path = fallback_path

img = cv2.imread(image_path)
if img is None:
    raise FileNotFoundError(f"Gambar tidak ditemukan di path: {image_path}. Pastikan file gambar tersedia.")
scale_percent = 60 # percent of original size
width = int(img.shape[1] * scale_percent / 100)
height = int(img.shape[0] * scale_percent / 100)
dim = (width, height)
resized = cv2.resize(img,dim)
gray=cv2.cvtColor(resized,cv2.COLOR_BGR2GRAY)
faces=face_cascade.detectMultiScale(gray,6.5,17)
for(x,y,w,h) in faces:
    resized=cv2.rectangle(resized,(x,y),(x+w,y+h),(0,255,0),2)
cv2.imshow('img',resized)
cv2.waitKey(0)
cv2.destroyAllWindows()