# Example 7.34 BGRemoval with OpenCV
import cv2 as cv


def remove_background(image, thresh, scale_factor=0.25, kernel_range=range(1, 15), border=None):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    im_bw = cv.threshold(gray, thresh, 255, cv.THRESH_BINARY)[1]
    im_bw_inv = cv.bitwise_not(im_bw)
    contour, _ = cv.findContours(im_bw_inv, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE)
    
    for cnt in contour:
        cv.drawContours(im_bw_inv, [cnt], 0, 255, -1)
        
    nt = cv.bitwise_not(im_bw)
    im_bw_inv = cv.bitwise_or(im_bw_inv, nt)
    border = border or kernel_range[-1]
    small = cv.resize(im_bw_inv, None, fx=scale_factor, fy=scale_factor)
    bordered = cv.copyMakeBorder(small, border, border, border, border, cv.BORDER_CONSTANT)
    
    for i in kernel_range:
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (2 * i + 1, 2 * i + 1))
        bordered = cv.morphologyEx(bordered, cv.MORPH_CLOSE, kernel)
        
    unbordered = bordered[border: -border, border: -border]
    mask = cv.resize(unbordered, (image.shape[1], image.shape[0]))
    fg = cv.bitwise_and(image, image, mask=mask)
    return fg


img = cv.imread('person_football.jpeg')
cv.imshow('image', img)
nb_img = remove_background(img, 100)
cv.imshow('bg rm', nb_img)

cv.waitKey(0)
cv.destroyAllWindows()