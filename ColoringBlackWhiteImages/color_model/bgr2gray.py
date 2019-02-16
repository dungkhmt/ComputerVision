import os
import cv2

list_img = os.listdir('./data/BGR/')
count = 1
for image in list_img:
    img = cv2.imread('./data/BGR/' + image)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite('./data/bgr_image/' + str(count) + '.jpg', img)
    cv2.imwrite('./data/gray_image/' + str(count) + '.jpg', gray)
    count += 1
