import keras
import cv2
from color_model import ColorizationModel, incpetion_predict
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from skimage.color import rgb2lab, lab2rgb, rgb2gray, xyz2lab
from skimage.io import imsave
import numpy as np
import argparse
import uuid
from preprocessing import themvien
from setting import *
from image_processing import processing

class Predict():
    def __init__(self, path):
        model_color = ColorizationModel()
        self.model_= model_color.transfer_model()
        self.model_.load_weights(path)

    def predict(self, path_image):
        img_ = themvien(path_image, (TARGET_SIZE, TARGET_SIZE), 0)
        emb = incpetion_predict(img_)
        lab_img = cv2.cvtColor(img_, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab_img)
        img_resize = cv2.resize(img_, (RESIZE_SIZE, RESIZE_SIZE))
        img_resize = np.array(img_resize, dtype = float)
        x = rgb2lab(1.0/255*img_resize)[:, :, 0]
        x = x.reshape(1, RESIZE_SIZE, RESIZE_SIZE, 1)
        output = self.model_.predict([x, emb])
        output *= 128

        cur = np.zeros((RESIZE_SIZE, RESIZE_SIZE, 3))
        cur[:, :, 0] = x[0][:, :, 0]
        cur[:, :, 1:] = output[0]
        result_name = str(uuid.uuid4()) + '.png'
        imsave('./static/image/' + result_name, lab2rgb(cur))
        l, a, b = processing('./static/image/' + result_name)
        out = np.zeros((TARGET_SIZE, TARGET_SIZE, 3), dtype = np.uint8)
        out[:, :, 0] = l_channel
        out[:, :, 1] = a
        out[:, :, 2] = b
        out = cv2.cvtColor(out, cv2.COLOR_LAB2BGR)
        cv2.imwrite('./static/image/' + result_name, out)
        return result_name

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--image', help = 'link to image')
    parser.add_argument('-m', '--model', help = 'link to model')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    predictor = Predict(args.model)
    predictor.predict(args.image)
    
