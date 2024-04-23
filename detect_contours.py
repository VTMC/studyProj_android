import cv2 as cv
import numpy as np
import cropContour

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

    # cv.namedWindow(windowName+'\'s detected contours', cv.WINDOW_NORMAL)
    # cv.imshow(windowName+'\'s detected contours', image_processed_res_img)
    # cv.resizeWindow(windowName+'\'s detected contours', 500, 800)

    cv.imwrite(path, image_processed_res_img)

    return resultContours

def getContourRect(contours, index):
    contourCornerPoints = cropContour.find_corner_points(contours[index])

    print("contourCornerPoints : ", contourCornerPoints)

    contourTopCenterPoint = (((contourCornerPoints[1][0] + contourCornerPoints[3][0])/2), ((contourCornerPoints[1][1] + contourCornerPoints[3][1])/2))
    contourBottomCenterPoint = (((contourCornerPoints[0][0] + contourCornerPoints[2][0])/2), ((contourCornerPoints[0][1] + contourCornerPoints[2][1])/2))

    print("contourTopCenterPoint : ", contourTopCenterPoint)
    print("contourBottomCenterPoint : ", contourBottomCenterPoint)


path = "D:/studyProj_android/traceUrine_NewAlgorithm/testingImages/t/"
outputPath = "D:/studyProj_android/traceUrine_NewAlgorithm/testingImages/croppedImg/"

img_t1_1 = cv.imread(path+"t1_1.jpg")
img_t3_4 = cv.imread(path+"t3_4.jpg")
img_t9_16 = cv.imread(path+"t9_16.jpg")
img_t_full = cv.imread(path+"t_full.jpg")

print("t1_1 test...")
detectedContours_t1_1 = detect_contours(outputPath+"contoursImg_t1_1.jpg",img_t1_1)
cropped_t1_1 = cropContour.crop_contour(img_t1_1, detectedContours_t1_1, 2, 1000, 2000)
cv.imwrite(outputPath+"cropped_t1_1.jpg", cropped_t1_1)
print("\n\nt3_4 test...")
detectedContours_t3_4 = detect_contours(outputPath+"contoursImg_t3_4.jpg",img_t3_4)
cropped_t3_4 = cropContour.crop_contour(img_t3_4, detectedContours_t3_4, 2, 1000, 2000)
cv.imwrite(outputPath+"cropped_t3_4.jpg", cropped_t3_4)
print("\n\nt9_16 test...")
detectedContours_t9_16 = detect_contours(outputPath+"contoursImg_t9_16.jpg",img_t9_16)
cropped_t9_16 = cropContour.crop_contour(img_t9_16, detectedContours_t9_16, 2, 1000, 2000)
cv.imwrite(outputPath+"cropped_t9_16.jpg", cropped_t9_16)
print("\n\nt_full test...")
detectedContours_t_full = detect_contours(outputPath+"contoursImg_t_full.jpg",img_t_full)
cropped_t_full = cropContour.crop_contour(img_t_full, detectedContours_t_full, 2, 1000, 2000)
cv.imwrite(outputPath+"cropped_t_full.jpg", cropped_t_full)

path = "D:/studyProj_android/traceUrine_NewAlgorithm/testingImages/29_5_t/"

img_t1_1 = cv.imread(path+"29_5_t1_1.jpg")
img_t3_4 = cv.imread(path+"29_5_t3_4.jpg")
img_t9_16 = cv.imread(path+"29_5_t9_16.jpg")
img_t_full = cv.imread(path+"29_5_t_full.jpg")

print("t1_1 test...")
detectedContours_29_5_t1_1 = detect_contours(outputPath+"contoursImg_29_5_t1_1.jpg",img_t1_1)
cropped_29_5_t1_1 = cropContour.crop_contour(img_t1_1, detectedContours_29_5_t1_1, 2, 1000, 2000)
cv.imwrite(outputPath+"cropped_29_5_t1_1.jpg", cropped_29_5_t1_1)
print("\n\nt3_4 test...")
detectedContours_29_5_t3_4 = detect_contours(outputPath+"contoursImg_29_5_t3_4.jpg",img_t3_4)
cropped_29_5_t3_4 = cropContour.crop_contour(img_t3_4, detectedContours_29_5_t3_4, 3, 1000, 2000)
cv.imwrite(outputPath+"cropped_29_5_t3_4.jpg", cropped_29_5_t3_4)
print("\n\nt9_16 test...")
detectedContours_29_5_t9_16 = detect_contours(outputPath+"contoursImg_29_5_t9_16.jpg",img_t9_16)
cropped_29_5_t9_16 = cropContour.crop_contour(img_t9_16, detectedContours_29_5_t9_16, 2, 1000, 2000)
cv.imwrite(outputPath+"cropped_29_5_t9_16.jpg", cropped_29_5_t9_16)
print("\n\nt_full test...")
detectedContours_29_5_t_full = detect_contours(outputPath+"contoursImg_29_5_t_full.jpg",img_t_full)
cropped_29_5_t_full = cropContour.crop_contour(img_t_full, detectedContours_29_5_t_full, 2, 1000, 2000)
cv.imwrite(outputPath+"cropped_29_5_t_full.jpg", cropped_29_5_t_full)

croppedImg_t1_1 = cv.imread(outputPath+"cropped_t1_1.jpg")
croppedImg_t3_4 = cv.imread(outputPath+"cropped_t3_4.jpg")
croppedImg_t9_16 = cv.imread(outputPath+"cropped_t9_16.jpg")
croppedImg_t_full = cv.imread(outputPath+"cropped_t_full.jpg")
croppedImg_29_5_t1_1 = cv.imread(outputPath+"cropped_29_5_t1_1.jpg")
croppedImg_29_5_t3_4 = cv.imread(outputPath+"cropped_29_5_t3_4.jpg")
croppedImg_29_5_t9_16 = cv.imread(outputPath+"cropped_29_5_t9_16.jpg")
croppedImg_29_5_t_full = cv.imread(outputPath+"cropped_29_5_t_full.jpg")

detectedContours_cropped_t1_1 = detect_contours(outputPath+"contoursImg_cropped_t1_1.jpg",croppedImg_t1_1)
detectedContours_cropped_t3_4 = detect_contours(outputPath+"contoursImg_cropped_t3_4.jpg",croppedImg_t3_4)
detectedContours_cropped_t9_16 = detect_contours(outputPath+"contoursImg_cropped_t9_16.jpg",croppedImg_t9_16)
detectedContours_cropped_t_full = detect_contours(outputPath+"contoursImg_cropped_t_full.jpg",croppedImg_t_full)
detectedContours_cropped_29_5_t1_1 = detect_contours(outputPath+"contoursImg_cropped_29_5_t1_1.jpg",croppedImg_29_5_t1_1)
detectedContours_cropped_29_5_t3_4 = detect_contours(outputPath+"contoursImg_cropped_29_5_t3_4.jpg",croppedImg_29_5_t3_4)
detectedContours_cropped_29_5_t9_16 = detect_contours(outputPath+"contoursImg_cropped_29_5_t9_16.jpg",croppedImg_29_5_t9_16)
detectedContours_cropped_29_5_t_full = detect_contours(outputPath+"contoursImg_cropped_29_5_t_full.jpg",croppedImg_29_5_t_full)

getContourRect(detectedContours_cropped_t1_1, 1)
getContourRect(detectedContours_cropped_t3_4, 1)
getContourRect(detectedContours_cropped_t9_16, 1)
getContourRect(detectedContours_cropped_t_full, 1)
getContourRect(detectedContours_cropped_29_5_t1_1, 1)
getContourRect(detectedContours_cropped_29_5_t3_4, 1)
getContourRect(detectedContours_cropped_29_5_t9_16, 1)
getContourRect(detectedContours_cropped_29_5_t_full, 1)

# path = "D:/studyProj_android/traceUrine_NewAlgorithm/testingImages/traceUrine_testImg/"

# img_a80_test1 = cv.imread(path+"A80_test1.bmp")
# img_a80_test2 = cv.imread(path+"A80_test2.bmp")
# img_a80_test3 = cv.imread(path+"A80_test3.bmp")
# img_s21u_test1 = cv.imread(path+"S21U_test1.bmp")
# img_s21u_test2 = cv.imread(path+"S21U_test2.bmp")
# img_s21u_test3 = cv.imread(path+"S21U_test3.bmp")
# img_zfold3_1_test1 = cv.imread(path+"ZFOLD3_1_test1.bmp")
# img_zfold3_1_test2 = cv.imread(path+"ZFOLD3_1_test2.bmp")
# img_zfold3_1_test3 = cv.imread(path+"ZFOLD3_1_test3.bmp")
# img_zfold3_2_test1 = cv.imread(path+"ZFOLD3_2_test1.bmp")
# img_zfold3_2_test2 = cv.imread(path+"ZFOLD3_2_test2.bmp")
# img_zfold3_2_test3 = cv.imread(path+"ZFOLD3_2_test3.bmp")
# img_y700_test1 = cv.imread(path+"Y700_test1.bmp")
# img_y700_test2 = cv.imread(path+"Y700_test2.bmp")
# img_y700_test3 = cv.imread(path+"Y700_test3.bmp")

# print("A80_test1 test...")
# detectedContours_a80_test1 = detect_contours(path+"contoursImg_a80_test1.jpg",img_a80_test1)
# cropped_a80_test1 = cropContour.crop_contour(img_a80_test1, detectedContours_a80_test1, 1, 1000, 2000)
# cv.imwrite(path+"cropped_a80_test1.jpg", cropped_a80_test1)
# print("\n\nA80_test2 test...")
# detectedContours_a80_test2= detect_contours(path+"contoursImg_a80_test2.jpg",img_a80_test2)
# cropped_a80_test2 = cropContour.crop_contour(img_a80_test2, detectedContours_a80_test2, 1, 1000, 2000)
# cv.imwrite(path+"cropped_a80_test2.jpg", cropped_a80_test2)
# print("\n\nA80_test3 test...")
# detectedContours_a80_test3 = detect_contours(path+"contoursImg_a80_test3.jpg",img_a80_test3)
# cropped_a80_test3 = cropContour.crop_contour(img_a80_test3, detectedContours_a80_test3, 1, 1000, 2000)
# cv.imwrite(path+"cropped_a80_test3.jpg", cropped_a80_test3)

# print("\n\nS21U_test1 test...")
# detectedContours_s21u_test1 = detect_contours(path+"contoursImg_s21u_test1.jpg",img_s21u_test1)
# cropped_s21u_test1 = cropContour.crop_contour(img_s21u_test1, detectedContours_s21u_test1, 1, 1000, 2000)
# cv.imwrite(path+"cropped_s21u_test1.jpg", cropped_s21u_test1)
# print("\n\nS21U_test2 test...")
# detectedContours_s21u_test2 = detect_contours(path+"contoursImg_s21u_test2.jpg",img_s21u_test2)
# cropped_s21u_test2 = cropContour.crop_contour(img_s21u_test2, detectedContours_s21u_test2, 1, 1000, 2000)
# cv.imwrite(path+"cropped_s21u_test2.jpg", cropped_s21u_test2)
# print("\n\nS21U_test3 test...")
# detectedContours_s21u_test3 = detect_contours(path+"contoursImg_s21u_test3.jpg",img_s21u_test3)
# cropped_s21u_test3 = cropContour.crop_contour(img_s21u_test3, detectedContours_s21u_test3, 1, 1000, 2000)
# cv.imwrite(path+"cropped_s21u_test3.jpg", cropped_s21u_test3)

# print("\n\nZFold3_1_test1 test...")
# detectedContours_zfold3_1_test1 = detect_contours(path+"contoursImg_zfold3_1_test1.jpg",img_zfold3_1_test1)
# cropped_zfold3_1_test1 = cropContour.crop_contour(img_zfold3_1_test1, detectedContours_zfold3_1_test1, 1, 1000, 2000)
# cv.imwrite(path+"cropped_zfold3_1_test1.jpg", cropped_zfold3_1_test1)
# print("\n\nZFold3_1_test2 test...")
# detectedContours_zfold3_1_test2 = detect_contours(path+"contoursImg_zfold3_1_test2.jpg",img_zfold3_1_test2)
# cropped_zfold3_1_test2 = cropContour.crop_contour(img_zfold3_1_test2, detectedContours_zfold3_1_test2, 1, 1000, 2000)
# cv.imwrite(path+"cropped_zfold3_1_test2.jpg", cropped_zfold3_1_test2)
# print("\n\nZFold3_1_test3 test...")
# detectedContours_zfold3_1_test3 = detect_contours(path+"contoursImg_zfold3_1_test3.jpg",img_zfold3_1_test3)
# cropped_zfold3_1_test3 = cropContour.crop_contour(img_zfold3_1_test3, detectedContours_zfold3_1_test3, 1, 1000, 2000)
# cv.imwrite(path+"cropped_zfold3_1_test3.jpg", cropped_zfold3_1_test3)

# print("\n\nZFold3_2_test1 test...")
# detectedContours_zfold3_2_test1 = detect_contours(path+"contoursImg_zfold3_2_test1.jpg",img_zfold3_2_test1)
# cropped_zfold3_2_test1 = cropContour.crop_contour(img_zfold3_2_test1, detectedContours_zfold3_2_test1, 1, 1000, 2000)
# cv.imwrite(path+"cropped_zfold3_2_test1.jpg", cropped_zfold3_2_test1)
# print("\n\nZFold3_2_test2 test...")
# detectedContours_zfold3_2_test2 = detect_contours(path+"contoursImg_zfold3_2_test2.jpg",img_zfold3_2_test2)
# cropped_zfold3_2_test2 = cropContour.crop_contour(img_zfold3_2_test2, detectedContours_zfold3_2_test2, 1, 1000, 2000)
# cv.imwrite(path+"cropped_zfold3_2_test2.jpg", cropped_zfold3_2_test2)
# print("\n\nZFold3_2_test3 test...")
# detectedContours_zfold3_2_test3 = detect_contours(path+"contoursImg_zfold3_2_test3.jpg",img_zfold3_2_test3)
# cropped_zfold3_2_test3 = cropContour.crop_contour(img_zfold3_2_test3, detectedContours_zfold3_2_test3, 1, 1000, 2000)
# cv.imwrite(path+"cropped_zfold3_2_test3.jpg", cropped_zfold3_2_test3)

# print("\n\nY700_test1 test...")
# detectedContours_y700_test1 = detect_contours(path+"contoursImg_y700_test1.jpg",img_y700_test1)
# cropped_y700_test1 = cropContour.crop_contour(img_y700_test1, detectedContours_y700_test1, 1, 1000, 2000)
# cv.imwrite(path+"cropped_y700_test1.jpg", cropped_y700_test1)
# print("\n\nY700_test2 test...")
# detectedContours_y700_test2 = detect_contours(path+"contoursImg_y700_test2.jpg",img_y700_test2)
# cropped_y700_test2 = cropContour.crop_contour(img_y700_test2, detectedContours_y700_test2, 1, 1000, 2000)
# cv.imwrite(path+"cropped_y700_test2.jpg", cropped_y700_test2)
# print("\n\nY700_test3 test...")
# detectedContours_y700_test3 = detect_contours(path+"contoursImg_y700_test3.jpg",img_y700_test3)
# cropped_y700_test3 = cropContour.crop_contour(img_y700_test3, detectedContours_y700_test3, 1, 1000, 2000)
# cv.imwrite(path+"cropped_y700_test3.jpg", cropped_y700_test3)

# cv.waitKey(0)
# cv.destroyAllWindows()
