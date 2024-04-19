import cv2 as cv
import numpy as np

def resizeImgWithRatio(img, target_width, target_height):
    width = img.shape[1]
    height = img.shape[0]
    print("[resizeImgWithRatio] img width : ", width, " / img height : ", height)

    width_ratio = target_width / width
    height_ratio = target_height / height
    print("[resizeImgWithRatio] img width ratio : ", width_ratio, " / img height ratio : ", height_ratio)

    ratio_fit_width = width * width_ratio
    ratio_fit_height = height * height_ratio
    print("[resizeImgWithRatio] img fit ratio width : ", ratio_fit_width, " / img fit ratio height: ", ratio_fit_height)

    if(width_ratio > 1): # img width > target_width
        
    elif(width_ratio < 1):
    else:



    # ratio_fit_resized_img = cv.resize(img, (int(ratio_fit_width), int(ratio_fit_height)), interpolation=cv.INTER_LINEAR)

    ratio_fit_resized_img = cv.resize(img, dsize=(0,0), fx=width_ratio, fy=height_ratio)

    # resized_img = cv.resize(ratio_fit_resized_img, (target_width, target_height))

    return ratio_fit_resized_img

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


img1_1 = cv.imread("C:/Users/owner/Downloads/testingImages/1_1.jpg")
img3_4 = cv.imread("C:/Users/owner/Downloads/testingImages/3_4.jpg")
img9_16 = cv.imread("C:/Users/owner/Downloads/testingImages/9_16.jpg")
img_full = cv.imread("C:/Users/owner/Downloads/testingImages/full.jpg")

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

# fit ratio with resize images
resized_img1_1 = resizeImgWithRatio(img1_1, 1000, 2000)
resized_img3_4 = resizeImgWithRatio(img3_4, 1000, 2000)
resized_img9_16 = resizeImgWithRatio(img9_16, 1000, 2000)
resized_img_full = resizeImgWithRatio(img_full, 1000, 2000)

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

cv.imwrite("C:/Users/owner/Downloads/testingImages/resized_1_1.jpg", resized_img1_1)
cv.imwrite("C:/Users/owner/Downloads/testingImages/resized_3_4.jpg", resized_img3_4)
cv.imwrite("C:/Users/owner/Downloads/testingImages/resized_9_16.jpg", resized_img9_16)
cv.imwrite("C:/Users/owner/Downloads/testingImages/resized_full.jpg", resized_img_full)