import cv2 as cv
import numpy as np
import cropContour
import checkHeight

def descendingOrderContours(contours):
    sortedContours = sorted(contours, key=lambda x: cv.contourArea(x), reverse=True)

    # contours.sort(key=lambda x: cv.contourArea(x), reverse=True)

    # max_limiter = min(len(sortedContours), 7)
    # sortedUpTo7SortedContours = []

    # for i in range(max_limiter):
    #     area_value = cv.contourArea(sortedContours[i])
    #     print("[descendingOrderContours] contour[{}]\'s area value : {}".format(i, area_value))
    #     sortedUpTo7SortedContours.append(sortedContours[i])

    # return sortedUpTo7SortedContours

    return sortedContours

def detect_contours(path ,img):
    #copy img to draw contours
    copy_img = img.copy()
    
    # Convert to gray scale
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Convert to adaptive threshold image
    adaptive_threshold = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 51, 2)

    # Find contours
    contours, hierarchy = cv.findContours(adaptive_threshold, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    # sort descending order contours
    sorted_contours = descendingOrderContours(contours)

    # Draw contours
    colors = [(255, 0, 0), (255, 128, 0), (255, 255, 0), (0, 255, 0), (0, 0, 255), (0, 0, 128), (255, 0, 255),
    (0, 0, 0), (128,128,128), (200, 200, 200)]

    resultContours = []

    for i in range(len(colors)):
        cv.drawContours(copy_img, sorted_contours, i, colors[i], 3, cv.LINE_8, hierarchy)
        print("sorted contours [%d]'s contours Area : %f" % (i, cv.contourArea(sorted_contours[i])))
        resultContours.append(sorted_contours[i])

    image_processed_res_img = cv.cvtColor(copy_img, cv.COLOR_BGR2RGB)

    cv.imwrite(path, image_processed_res_img)

    return resultContours

def getContourRect(img, contours, index):
    img_copy = img.copy()

    contourCornerPoints = cropContour.find_corner_points(contours[index])

    for point in contourCornerPoints:
        cv.circle(img_copy, point, 5, (255, 0, 0), -1)
        cv.putText(img_copy, str(point), point, cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    print("contourCornerPoints : ", contourCornerPoints)

    contourTopCenterPoint = (int((contourCornerPoints[1][0] + contourCornerPoints[2][0])/2), int((contourCornerPoints[1][1] + contourCornerPoints[2][1])/2))
    contourBottomCenterPoint = (int((contourCornerPoints[0][0] + contourCornerPoints[3][0])/2), int((contourCornerPoints[0][1] + contourCornerPoints[3][1])/2))

    print("contourTopCenterPoint : ", contourTopCenterPoint)
    print("contourBottomCenterPoint : ", contourBottomCenterPoint)

    cv.circle(img_copy, contourTopCenterPoint, 5, (0, 0, 255), -1)
    cv.putText(img_copy, str(contourTopCenterPoint), contourTopCenterPoint, cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv.circle(img_copy, contourBottomCenterPoint, 5, (0, 0, 255), -1)
    cv.putText(img_copy, str(contourBottomCenterPoint), contourBottomCenterPoint, cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    return img_copy

def drawCornerPoints(img):
    img_copy = img.copy()

    points = [
        (0,0),                         # leftTop
        (img.shape[1], 0),             #rightTop
        (img.shape[1], img.shape[0]),  #rightBottom
        (0, img.shape[0])              #leftBottom
    ]

    print("CornerPoints : ", points)

    for point in points:
        cv.circle(img_copy, point, 5, (255, 0, 0), -1)
        cv.putText(img_copy, str(point), point, cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    pointTopCenterPoint = (int((points[0][0] + points[1][0])/2), int((points[0][1] + points[1][1])/2))
    pointBottomCenterPoint = (int((points[2][0] + points[3][0])/2), int((points[2][1] + points[3][1])/2))

    cv.circle(img_copy, pointTopCenterPoint, 5, (0, 0, 255), -1)
    cv.putText(img_copy, str(pointTopCenterPoint), pointTopCenterPoint, cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv.circle(img_copy, pointBottomCenterPoint, 5, (0, 0, 255), -1)
    cv.putText(img_copy, str(pointBottomCenterPoint), pointBottomCenterPoint, cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    print("TopCenterPoint : ", pointTopCenterPoint)
    print("BottomCenterPoint : ", pointBottomCenterPoint)

    return img_copy
    