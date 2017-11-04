# coding: utf-8
import cv2

# Load two images
img1 = cv2.imread("/Users/huangjintao/Desktop/face/test/test_1.jpg")
img2 = cv2.imread("/Users/huangjintao/Desktop/face/result/2.png")

# create a ROI
rows, cols, channels = img2.shape
roi = img1[0:rows, 0:cols]

# create a mask of logo and create its inverse mask also
img2gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
mask_inv = cv2.bitwise_not(mask)

# black-out the area of logo in ROI
img1_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)

# Take only region of logo from logo image.
img2_fg = cv2.bitwise_and(img2, img2, mask=mask)

# Put logo in ROI and modify the main image
dst = cv2.add(img1_bg, img2_fg)
img1[0:rows, 0:cols] = dst


cv2.imwrite('/Users/huangjintao/Desktop/face_2/combine_2.png', img1)