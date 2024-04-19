import cv2
import numpy as np

fn = 'C:/Users/owner/Downloads/Send Anywhere (2024-04-02 09-54-17)/20240402_094644.jpg'  # set tile filename
img = cv2.imread(fn)  # read tile into img.

rgb_planes = cv2.split(img)

result_planes = []
result_norm_planes = []
for plane in rgb_planes:
    dilated_img = cv2.dilate(plane, np.ones((7,7), np.uint8))
    bg_img = cv2.medianBlur(dilated_img, 21)
    diff_img = 255 - cv2.absdiff(plane, bg_img)
    norm_img = cv2.normalize(diff_img,None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    result_planes.append(diff_img)
    result_norm_planes.append(norm_img)
    
merged_plane = cv2.merge(result_planes)
merged_plane_norm = cv2.merge(result_norm_planes)

result = cv2.subtract(img, merged_plane)


cv2.namedWindow('img', cv2.WINDOW_NORMAL)
cv2.namedWindow('result', cv2.WINDOW_NORMAL)
cv2.namedWindow('result_norm', cv2.WINDOW_NORMAL)

cv2.resizeWindow(winname='img', width=500, height=1000)
cv2.resizeWindow(winname='result', width=500, height=1000)
cv2.resizeWindow(winname='result_norm', width=500, height=1000)

cv2.imshow('img', img)
cv2.imshow('result', result)
cv2.imshow('result_norm', merged_plane_norm)

cv2.waitKey(0)
cv2.destroyAllWindows()