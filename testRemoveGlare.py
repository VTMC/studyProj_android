import cv2
import numpy as np

fn = 'C:/Users/owner/Downloads/raw_data(10)/traceurine_bmp/cr_20240327_151713.bmp'  # set tile filename
img = cv2.imread(fn)  # read tile into img.
# median blur. This seems to be better than gaussian for bright dots.
blr = cv2.medianBlur(img, 15)
# now grab brightness V of HSV here - but Gray is possibly as good
hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
val = hsv[:, :, 2]
# use ADAPTIVE_THRESH_GAUSSIAN to find spots. 
# I manually tweaked the values- these seem to work well with what I have.
at = cv2.adaptiveThreshold(np.array(255 - val), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 5, 17)
# Now invert the threshold, and run another for edges.
ia = np.array(255 - at)  # inversion of adaptiveThreshold of the value.
iv = cv2.adaptiveThreshold(ia, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 5, 9)
# ib = merged edges with the dots (as an invert mask).
ib = cv2.subtract(iv, ia)
# Turn this to a 3 channel mask.
bz = cv2.merge([ib, ib, ib])
# Use the blur where the mask is, otherwise use the image.
dsy = np.where(bz == (0, 0, 0), blr, img)
result = dsy

cv2.imshow('img', img)
cv2.imshow('result', result)
cv2.waitKey(0)
cv2.destroyAllWindows()