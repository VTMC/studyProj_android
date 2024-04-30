import cv2 as cv
import numpy as np

def rotate_contour(contour, angle):
    contour_points = contour.reshape(-1, 2)

    bound_contour_rect = cv.boundingRect(contour)
    center_x = (bound_contour_rect[0] + bound_contour_rect[2]) / 2
    center_y = (bound_contour_rect[1] + bound_contour_rect[3]) / 2

    radian = np.deg2rad(angle)

    rotated_points = np.zeros_like(contour_points)

    for i, point in enumerate(contour_points):
        rotated_x = int((point[0] - center_x) * np.cos(radian) - (point[1] - center_y) * np.sin(radian) + center_x)
        rotated_y = int((point[0] - center_x) * np.sin(radian) + (point[1] - center_y) * np.cos(radian) + center_y)

        rotated_points[i] = [rotated_x, rotated_y]

    return rotated_points

def find_corner_points(contour, angle=45):
    contour_points = contour.reshape(-1, 2)
    rotated_contour_points = rotate_contour(contour, angle)

    min_x_index = np.argmin(rotated_contour_points[:, 0])
    max_x_index = np.argmax(rotated_contour_points[:, 0])
    min_y_index = np.argmin(rotated_contour_points[:, 1])
    max_y_index = np.argmax(rotated_contour_points[:, 1])

    result_points = [
        tuple(contour_points[min_x_index]), #leftBottom
        tuple(contour_points[max_x_index]), #rightTop
        tuple(contour_points[min_y_index]), #rightBottom
        tuple(contour_points[max_y_index])  #leftTop
    ]

    return result_points

def crop_contour(img ,descendedOrderedContours, index, targetWidth, targetHeight):
    targetContour = descendedOrderedContours[index]

    targetContourPoints = find_corner_points(targetContour)

    print(str(targetContourPoints))

    targetPoints = np.array([
        [0, targetHeight],
        [targetWidth, 0],
        [0, 0],
        [targetWidth, targetHeight]
    ], dtype=np.float32)

    targetContourPointsMat = np.array(targetContourPoints, dtype=np.float32)
    targetPointsMat = np.array(targetPoints, dtype=np.float32)

    matrix = cv.getPerspectiveTransform(targetContourPointsMat, targetPointsMat)
    resultMat = cv.warpPerspective(img, matrix, (targetWidth, targetHeight))
    
    return resultMat



