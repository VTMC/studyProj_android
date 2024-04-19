import sys
import cv2
import rawpy
from PIL import Image
import numpy as np

# Load image

print("setting PATH...")
print("set path in command (YOU MUST INPUT ABSOLUTE PATH)")

try:
    path = input("path : ")
    path_split = path.split(':')
    print(path_split)
    path_split = path_split[1].split('/')
    print(path_split)
    path_split = path_split[len(path_split)-1].split('.')
    print(path_split)
except:
    print("!!!!!WRONG INPUT!!!!!\n")
    print("check path of this image\n")
    sys.exit()

if(len(path_split) > 2):
    print("!!!!!WRONG INPUT!!!!!\n")
    print("Don't use your image's name include '.'\n")
    sys.exit()
else:
    imageType = path_split[1]
    if(imageType == 'dng'): # dng ... only RAW image
        read_img = rawpy.imread(path)
        img = read_img.postprocess()
        # with img as rp:
        #     rgb = rp.postprocess(
        #         # output_bps=16,
        #         # output_color=rawpy.ColorSpace.sRGB,
        #     ) #gamma=(1,1), no_auto_bright=True, output_bps=16
        #     # print(len(rgb))
        #     # print(rgb)

        #     # Extract each R,G,B Channels to save Separately
        #     r = rgb[:,:,0]
        #     g = rgb[:,:,1]
        #     b = rgb[:,:,2]
    else: #tiff, jpeg, png ... not RAW image
        img = Image.open(path)
        # rgb = np.array(img)

        # r = rgb[:,:,0]
        # g = rgb[:,:,1]
        # b = rgb[:,:,2]

# Convert to grayScale
grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Convert Threshold Image
thresholdValue = 130
ret, thresholdImg = cv2.threshold(grayImg, thresholdValue, 255, cv2.THRESH_BINARY)

img_pil = Image.fromarray(thresholdImg)
img_pil.show()