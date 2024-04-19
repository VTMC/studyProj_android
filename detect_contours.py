import cv2 as cv
import numpy as np

def descendingOrderContours(contours):
    sortedContours = sorted(contours, key=lambda x: cv.contourArea(x), reverse=True)

    # contours.sort(key=lambda x: cv.contourArea(x), reverse=True)

    max_limiter = min(len(sortedContours), 7)
    sortedUpTo7SortedContours = []

    for i in range(max_limiter):
        area_value = cv.contourArea(sortedContours[i])
        print("[descendingOrderContours] contour[{}]\'s area value : {}".format(i, area_value))
        sortedUpTo7SortedContours.append(sortedContours[i])

    return sortedUpTo7SortedContours

def detect_contours(img):
    # Convert to gray scale
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Convert to adaptive threshold image
    adaptive_threshold = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 51, 2)

    # Find contours
    contours, hierarchy = cv.findContours(adaptive_threshold, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    # sort descending order contours
    max_limiter_contours = descendingOrderContours(contours)

    # Draw contours
    colors = [(255, 0, 0), (255, 128, 0), (255, 255, 0), (0, 255, 0), (0, 0, 255), (0, 0, 128), (255, 0, 255)]

    cv.drawContours(img, [max_limiter_contours], -1, (0, 255, 0), 2)

    cv.namedWindow('detected contours', cv.WINDOW_NORMAL)
    cv.imshow("detected contours", img)
    cv.resizeWindow("detected contours", 500, 800)

    # return max_limiter_contours

# def cropImgByContour(img, contour):

img = cv.imread("C:/Users/owner/Downloads/testingImages/t1_1.jpg")

detectedContours = detect_contours(img)

cv.waitKey(0)
cv.destroyAllWindows()