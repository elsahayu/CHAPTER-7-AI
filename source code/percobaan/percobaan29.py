#Example 7.29 plot_watershed with Scikit-image
#import necessary packages
import numpy as np
import urllib.request as urllib
import cv2
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
print("Mengunduh gambar...")
req = urllib.Request("https://jooinn.com/images/human-skin-4.jpg", headers={'User-Agent': 'Mozilla/5.0'})
resp = urllib.urlopen(req)
image = np.asarray(bytearray(resp.read()), dtype="uint8")
image = cv2.imdecode(image, cv2.IMREAD_COLOR)

# Resize gambar agar komputasi watershed cepat (resolusi asli 2462x3693 / 9 MP)
image = cv2.resize(image, (500, 500), interpolation=cv2.INTER_AREA)
print("Memproses segmentasi watershed...")


#show image
#cv2.imshow("Image", image)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
thresh = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,
cv2.THRESH_BINARY,11,2)
# Now we want to separate the two objects in image
# Generate the markers as local maxima of the distance to the background
distance = ndi.distance_transform_edt(thresh)
coords = peak_local_max(distance, footprint=np.ones((3, 3)),
labels=gray)
mask = np.zeros(distance.shape, dtype=bool)
mask[tuple(coords.T)] = True
markers, _ = ndi.label(mask)
labels = watershed(-distance, markers, mask=thresh)
fig, axes = plt.subplots(ncols=3, figsize=(9, 3), sharex=True,
sharey=True)
ax = axes.ravel()
ax[0].imshow(image_rgb, cmap=plt.cm.gray)
ax[0].set_title('Overlapping objects')
ax[1].imshow(-distance, cmap=plt.cm.gray)
ax[1].set_title('Distances')
ax[2].imshow(labels, cmap=plt.cm.nipy_spectral)
ax[2].set_title('Separated objects')
for a in ax:
    a.set_axis_off()
fig.tight_layout()
print("Selesai! Menampilkan plot...")
plt.show()