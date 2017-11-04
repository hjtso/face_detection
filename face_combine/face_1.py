# coding: utf-8
import cv2

image = cv2.imread("/Users/huangjintao/Desktop/face/test/test_1.jpg")
logo = cv2.imread("/Users/huangjintao/Desktop/face/result/2.png")

# 函数要求两张图必须是同一个size
# 记录了图像行数、列数和通道数的元组
h, w, _ = image.shape
logo2 = cv2.resize(logo, (w,h), interpolation=cv2.INTER_AREA)

#alpha，beta，gamma可调
#其中alpha是第一幅图片中元素的权重，beta是第二个的权重，gamma是加到最后结果上的一个值
alpha = 0.3
beta = 1-alpha
gamma = 0

img_add = cv2.addWeighted(image, alpha, logo2, beta, gamma)
#cv2.namedWindow('addImage')
#cv2.imshow('img_add',img_add)
#cv2.waitKey()
#cv2.destroyAllWindows()

cv2.imwrite('/Users/huangjintao/Desktop/face_2/combine_1.png', img_add)