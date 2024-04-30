import cv2 as cv
import numpy as np
from MovingAverageFilter import MovingAverageFilter

def cutOffAvg(values):
    valuesAvg = sum(values) / len(values)

    for i in range(0, len(values)):
        if values[i] > valuesAvg:
            values[i] = values[i] - valuesAvg
        else:
            values[i] = 0

    return values

def cutOff(values, standardNum):
    for i in range(0, len(values)):
        if values[i] > standardNum:
            values[i] = values[i] - standardNum
        else:
            values[i] = 0
            
    return values

def checkMinAndMaxValue(values):
    minValue = 255
    maxValue = 0
    
    for i in range(0, len(values)):
        if values[i] < minValue:
            minValue = values[i]
            
        if values[i] > maxValue:
            maxValue = values[i]
            
    return minValue, maxValue

def makeValueToStr(value):
    valueStr = []

    for i in range(0, len(value)):
        valueStr.append(str(value[i])+"\n")

    return valueStr

def checkHeight(img, outputPath, file_name):
    img_copy = img.copy()

    gray_file_path = outputPath + file_name + "_gray.txt"
    gray_diff_file_path = outputPath + file_name + "_gray_diff.txt"

    r_file_path = outputPath + file_name + "_r.txt"
    g_file_path = outputPath + file_name + "_g.txt"
    b_file_path = outputPath + file_name + "_b.txt"

    # denoisedImg = cv.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
    denoisedImg = cv.medianBlur(img, 5)

    grayImg = cv.cvtColor(denoisedImg, cv.COLOR_BGR2GRAY)
    rgbImg = cv.cvtColor(denoisedImg, cv.COLOR_BGR2RGB)

    grayValue = []
    minGrayValue = 255
    rgbValue = []

    detectorWidth = 10
    detectorStartPoint = 20

    #get gray values of center line
    for i in range(detectorStartPoint, (img.shape[0] - detectorStartPoint)):
        valueTotalSum = 0

        for j in range(0, 20):
            value = grayImg[i, grayImg.shape[1]//2 + (detectorWidth//2) - j]
            cv.circle(img_copy, (int(img.shape[1]/2 + (detectorWidth//2) - j), i), 1, (0, 0, 255), -1)
            valueTotalSum += value

        valueAvg = valueTotalSum / detectorWidth
        
        if valueAvg < minGrayValue:
            minGrayValue = valueAvg
            
        # value = grayImg[i, grayImg.shape[1]//2]
        grayValue.append(valueAvg)

        # value = rgbImg[i, img.shape[1]//2]
        # rgbValue.append(value)
        
    grayValue = cutOff(grayValue, minGrayValue)
    
    minValue, maxValue = checkMinAndMaxValue(grayValue)
    
    midValue = (minValue + maxValue) / 2
    
    grayValue = cutOff(grayValue, midValue)

    #make data denoise of gray values
    # for i in range(0, len(grayValue)):
    #     if i == 0 or i == len(grayValue) - 1:
    #         grayValue[i] = grayValue[i]

    #     if i == 1 or i == len(grayValue) - 2:

    # grayValue = cutOffAvg(grayValue)

    # maf = MovingAverageFilter(100)

    # for origin in grayValue:
    #     filtered = maf.push(origin)
    #     grayValue[grayValue.index(origin)] = filtered
    
    grayValueDiff = np.diff(grayValue)

    grayValueStr = makeValueToStr(grayValue)
    grayValueDiffStr = makeValueToStr(grayValueDiff)
    # rValueStr = makeValueToStr([value[0] for value in rgbValue])
    # gValueStr = makeValueToStr([value[1] for value in rgbValue])
    # bValueStr = makeValueToStr([value[2] for value in rgbValue])

    with open(gray_file_path, 'w') as f:
        f.writelines(grayValueStr)
    print("Gray value txt file writed...\n\n")

    with open(gray_diff_file_path, 'w') as f:
        f.writelines(grayValueDiffStr)
    print("Gray diff value txt file writed...\n\n")

    # with open(r_file_path, 'w') as f:
    #     f.writelines(rValueStr)
    # print("R value txt file writed...\n\n")

    # with open(g_file_path, 'w') as f:
    #     f.writelines(gValueStr)
    # print("G value txt file writed...\n\n")

    # with open(b_file_path, 'w') as f:
    #     f.writelines(bValueStr)
    # print("B value txt file writed...\n\n")
        
    return img_copy, grayValue