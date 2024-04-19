import cv2
import numpy as np

def light_remover(img):
    #RGB to Lab
    lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)

    #Median Filter 적용
    median_filter_size = 101
    lab_img = cv2.medianBlur(lab_img, median_filter_size)
    
    # L(Lightness) 채널 분리
    lab_channels = list(cv2.split(lab_img))
    l_channel = lab_channels[0]
    
    # L 채널 반전
    inverted_l_channel = cv2.subtract(np.ones(l_channel.shape, dtype=np.uint8) * 255, l_channel)
    l_bgr = cv2.cvtColor(inverted_l_channel, cv2.COLOR_GRAY2BGR)
    
    # L 채널들 병합
    # l_result = cv2.add(l_channel, inverted_l_channel)
    # lab_channels[0] = inverted_l_channel
    # lab_channels[0] = l_result
    # lab_img = cv2.merge(lab_channels)
    
    # Lab to RGB - None
    # lab_bgr_img = cv2.cvtColor(lab_img, cv2.COLOR_Lab2BGR)

    #가중 평균
    # alpha = 0.85 # 원본 이미지에 대한 가중치
    alpha = 0.7 # 원본 이미지에 대한 가중치
    beta = 1 - alpha # 수정된 이미지에 대한 가중치

    result_img = cv2.addWeighted(img, alpha, l_bgr, beta, 0)
    
    return result_img



image_path = "C:/Users/VTMC/Downloads/NearByShare/20240312_002349.jpg"

image = cv2.imread(image_path)

result_image = light_remover(image)

result_image_path = "C:/Users/VTMC/Downloads/NearByShare/lightRemoved_processed_image.jpg"
cv2.imwrite(result_image_path, result_image)

print("Light removal process completed., Result image saved at : ",result_image_path)