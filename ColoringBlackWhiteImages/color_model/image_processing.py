from keras.preprocessing.image import load_img
import numpy as np 
from skimage.color import rgb2lab, lab2rgb, rgb2gray, xyz2lab
from skimage.io import imsave
import cv2
import uuid
import os
from preprocessing import themvien
from setting import *

def processing(path):
    start_img = cv2.imread(path)
    start_img = cv2.resize(start_img, (TARGET_SIZE, TARGET_SIZE))
    start_img = cv2.cvtColor(start_img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(start_img)
    return l, a, b

if __name__ == "__main__":
    import os
    src_folder = "/home/hoanlk/Desktop/colordata/images/Train"
    target_folder = "/home/hoanlk/Desktop/RESIZE_SIZExRESIZE_SIZE"
    list_img = os.listdir(src_folder)
    count = 0
    for img in list_img:
        img_ = cv2.imread(os.path.join(src_folder, img))
        img_ = cv2.resize(img_, (RESIZE_SIZE, RESIZE_SIZE))
        cv2.imwrite(os.path.join(target_folder, img), img_)
        count += 1
        if count % 100 == 0:
            print(count, img)


