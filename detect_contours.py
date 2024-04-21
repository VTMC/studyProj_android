import cv2 as cv
import numpy as np

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

def detect_contours(windowName ,img):
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

    for i in range(len(colors)):
        cv.drawContours(img, sorted_contours, i, colors[i], 3, cv.LINE_8, hierarchy)
        print("sorted contours [%d]'s contours Area : %f" % (i, cv.contourArea(sorted_contours[i])))

    image_processed_res_img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    # cv.namedWindow(windowName+'\'s detected contours', cv.WINDOW_NORMAL)
    # cv.imshow(windowName+'\'s detected contours', image_processed_res_img)
    # cv.resizeWindow(windowName+'\'s detected contours', 500, 800)

    return image_processed_res_img

# def cropImgByContour(img, contour):

path = "K:/python/traceUrine_NewAlgorithm/testingImages/"

img_t1_1 = cv.imread(path+"resized_1_1.jpg")
img_t3_4 = cv.imread(path+"resized_3_4.jpg")
img_t9_16 = cv.imread(path+"resized_9_16.jpg")
img_t_full = cv.imread(path+"resized_full.jpg")

# resized_t1_1 = cv.resize(img_t1_1, (1000, 2000))
# print("t1_1 resized")
# resized_t3_4 = cv.resize(img_t3_4, (1000,2000))
# print("t3_4 resized")
# resized_t9_16 = cv.resize(img_t9_16, (1000, 2000))
# print("t9_16 resized")
# resized_t_full = cv.resize(img_t_full, (1000, 2000))
# print("t_full resized")

print("t1_1 test...")
detectedContours_t1_1 = detect_contours("img_t1_1",img_t1_1)
print("\n\nt3_4 test...")
detectedContours_t3_4 = detect_contours("img_t3_4",img_t3_4)
print("\n\nt9_16 test...")
detectedContours_t9_16 = detect_contours("img_t9_16",img_t9_16)
print("\n\nt_full test...")
detectedContours_t_full = detect_contours("img_t_full",img_t_full)

# cv.waitKey(0)
# cv.destroyAllWindows()
cv.imwrite(path+"contour_1_1.jpg", detectedContours_t1_1)
cv.imwrite(path+"conotur_3_4.jpg", detectedContours_t3_4)
cv.imwrite(path+"contour_9_16.jpg", detectedContours_t9_16)
cv.imwrite(path+"contour_full.jpg", detectedContours_t_full)