# coding: utf-8
import cv2

# Load two images
img1 = cv2.imread("/Users/huangjintao/Desktop/face/test/test_1.jpg")
img2 = cv2.imread("/Users/huangjintao/Desktop/face/result/2.png")

face_patterns = cv2.CascadeClassifier('/usr/local/Cellar/opencv/3.3.1/share/OpenCV/haarcascades/haarcascade_frontalface_alt2.xml')
faces = face_patterns.detectMultiScale(img1,scaleFactor=1.1,minNeighbors=5,minSize=(100, 100))

for x, y, w, h in faces:
    cv2.rectangle(img1, (x, y), (x+w, y+h), (0, 255, 0), 2)
    print(x, y, w, h )
    #crop_img = img1[y:y + h, x:x + w]
    #hh, ww, _ = crop_img.shape
    logo2 = cv2.resize(img2, (w, h), interpolation=cv2.INTER_AREA)

    img2gray = cv2.cvtColor(logo2, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)

    roi = img1[y:y+w, x:x+h]

# black-out the area of logo in ROI
    img1_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)

# Take only region of logo from logo image.
    img2_fg = cv2.bitwise_and(logo2, logo2, mask=mask)

# Put logo in ROI and modify the main image
    dst = cv2.add(img1_bg, img2_fg)
    img1[y:y+w, x:x+h] = dst

cv2.imwrite('/Users/huangjintao/Desktop/face_2/combine_4.png', img1)