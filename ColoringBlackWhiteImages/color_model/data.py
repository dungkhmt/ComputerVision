from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from skimage.color import rgb2lab, lab2rgb, rgb2gray, xyz2lab
from skimage.io import imsave
from keras.models import load_model
from keras.callbacks import EarlyStopping
import numpy as np
import os
import random
import tensorflow as tf
from preprocessing import themvien
import cv2

def load_data(DIR):
    list_img = os.listdir(DIR)
    X = []
    Y = []
    for i in list_img:
        try:
            print(DIR + i)
            image = img_to_array(load_img(DIR + i))
            image = np.array(image, dtype = np.float16)
            x = rgb2lab(1.0/255*image)[:,:,0].astype(np.float16)
            y = rgb2lab(1.0/255*image)[:,:,1:].astype(np.float16)
            y /= 128
            x = x.reshape((128, 128, 1))
            y = y.reshape((128, 128, 2))
            X.append(x)
            Y.append(y)
        except:
            pass 
    return np.array(X), np.array(Y)
