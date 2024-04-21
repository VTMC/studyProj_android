import cv2 as cv
import numpy as np

def resizeImgWithRatio(img, target_height):
    width = img.shape[1]
    height = img.shape[0]
    print("[resizeImgWithRatio] img width : %d, height %d" % (width, height))

    height_ratio = target_height / height
    new_width = width * height_ratio

    resized_img = cv.resize(img, (int(new_width), target_height))

    return resized_img

def resizeToSqrImgWithRatio(img, target_width, target_height):
    resultMat = np.zeros((target_width, target_height,3), np.uint8)
    
    width = img.shape[1]
    height = img.shape[0]
    print("[resizeToSqrImgWithRatio] img width : ", width, " / img height : ", height)
    print("ratio of width, height : %f" % (width/height))

    width_ratio = target_width / width
    height_ratio = target_height / height
    print("[resizeToSqrImgWithRatio] img width ratio : ", width_ratio, " / img height ratio : ", height_ratio)

    if width_ratio < height_ratio:
        ratio_fit_size = (int(width*width_ratio), int(height*width_ratio))  
    else:
        ratio_fit_size = (int(width*height_ratio), int(height*height_ratio))

    ratio_fit_img = cv.resize(img, ratio_fit_size)
    print("resized ratio of width, height : %f" % (ratio_fit_img.shape[1] / ratio_fit_img.shape[0]))

    resultMat[int(target_height/2 - ratio_fit_size[1]/2) : int(target_height/2 + ratio_fit_size[1]/2),
    int(target_width/2 - ratio_fit_size[0]/2) : int(target_width/2 + ratio_fit_size[0]/2), :] = ratio_fit_img

    return resultMat

def cropImg(img, cropWidth, cropHeight):
    img_height, img_width = img.shape[:2]

    start_x = max(img_width // 2 - (cropWidth // 2), 0)
    start_y = max(img_height // 2 - (cropHeight // 2), 0)

    # 크롭할 영역의 끝 지점을 계산합니다.
    end_x = min(start_x + cropWidth, img_width)
    end_y = min(start_y + cropHeight, img_height)

    # 이미지를 크롭합니다.
    cropped_image = img[start_y:end_y, start_x:end_x]

    return cropped_image


path = "K:/python/traceUrine_NewAlgorithm/testingImages/"

img1_1 = cv.imread(path+"29_5_t1_1.jpg")
img3_4 = cv.imread(path+"29_5_t3_4.jpg")
img9_16 = cv.imread(path+"29_5_t9_16.jpg")
img_full = cv.imread(path+"29_5_t_full.jpg")

# img1_1 = cv.imread(path+"t1_1.jpg")
# img3_4 = cv.imread(path+"t3_4.jpg")
# img9_16 = cv.imread(path+"t9_16.jpg")
# img_full = cv.imread(path+"t_full.jpg")

# check width, height
print("target size : 2000*4000") #width < height -> ratio = 0.5
print("img1_1 size : ", img1_1.shape) #0=height, 1=width, 2=channel
print("img3_4 size : ", img3_4.shape)
print("img6_19 size : ", img9_16.shape)
print("img_full size : ", img_full.shape)

# check ratio image (순서 : 좌상, 우상, 우하, 좌하)
# src_img1_1 = np.array([[0, 0], [img1_1.shape[1], 0], [img1_1.shape[1], img1_1.shape[0]], [0, img1_1.shape[0]]], dtype =np.float32)
# src_img3_4 = np.array([[0, 0], [img3_4.shape[1], 0], [img3_4.shape[1], img3_4.shape[0]], [0, img3_4.shape[0]]], dtype =np.float32)
# src_img9_16 = np.array([[0, 0], [img9_16.shape[1], 0], [img9_16.shape[1], img9_16.shape[0]], [0, img9_16.shape[0]]], dtype =np.float32)
# src_img_full = np.array([[0, 0], [img_full.shape[1], 0], [img_full.shape[1], img_full.shape[0]], [0, img_full.shape[0]]], dtype =np.float32)
# dst_img_point = np.array([[0,0], [500, 0], [500, 1000], [0, 1000]], dtype= np.float32)

# resize image by img's height
resized_img1_1 = resizeImgWithRatio(img1_1, 4000)
resized_img3_4 = resizeImgWithRatio(img3_4, 4000)
resized_img9_16 = resizeImgWithRatio(img9_16, 4000)
resized_img_full = resizeImgWithRatio(img_full, 4000)

# fit ratio with resize images
resized_sqr_img1_1 = resizeToSqrImgWithRatio(resized_img1_1, 2000, 2000)
resized_sqr_img3_4 = resizeToSqrImgWithRatio(resized_img3_4, 2000, 2000)
resized_sqr_img9_16 = resizeToSqrImgWithRatio(resized_img9_16, 2000, 2000)
resized_sqr_img_full = resizeToSqrImgWithRatio(resized_img_full, 2000, 2000)

# get matrix to transform
# matrix1_1 = cv.getPerspectiveTransform(src_img1_1, dst_img_point)
# matrix3_4 = cv.getPerspectiveTransform(src_img3_4, dst_img_point)
# matrix9_16 = cv.getPerspectiveTransform(src_img9_16, dst_img_point)
# matrix_full = cv.getPerspectiveTransform(src_img_full, dst_img_point)

# transform images to resize
# resized_img1_1 = cv.warpPerspective(img1_1, matrix1_1, (500, 1000))
# resized_img3_4 = cv.warpPerspective(img3_4, matrix3_4, (500, 1000))
# resized_img9_16 = cv.warpPerspective(img9_16, matrix9_16, (500, 1000))
# resized_img_full = cv.warpPerspective(img_full, matrix_full, (500, 1000))

# crop images
# resized_img1_1 = cropImg(img1_1, 1000, 2000)
# resized_img3_4 = cropImg(img3_4, 1000, 2000)
# resized_img9_16 = cropImg(img9_16, 1000, 2000)
# resized_img_full = cropImg(img_full, 1000, 2000)

cv.imwrite(path+"resized_1_1.jpg", resized_sqr_img1_1)
cv.imwrite(path+"resized_3_4.jpg", resized_sqr_img3_4)
cv.imwrite(path+"resized_9_16.jpg", resized_sqr_img9_16)
cv.imwrite(path+"resized_full.jpg", resized_sqr_img_full)