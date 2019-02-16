import keras
import cv2
from color_model import ColorizationModel
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from skimage.color import rgb2lab, lab2rgb, rgb2gray, xyz2lab
from skimage.io import imsave
import numpy as np
import argparse

class Predict():
    def __init__(self):
        pass

    def load_trained_model(self,path):
        model_color = ColorizationModel()
        model_= model_color.define_model()
        model_.load_weights(path)
        return model_

    def predict(self, model, path_image):
        img_ = img_to_array(load_img(path_image))
        img_ = np.array(img_, dtype = float)
        x = rgb2lab(1.0/255*img_)[:, :, 0]
        x = x.reshape(1, 128, 128, 1)
        output = model.predict(x)
        output *= 128

        cur = np.zeros((128, 128, 3))
        cur[:, :, 0] = x[0][:, :, 0]
        cur[:, :, 1:] = output[0]
        imsave('result.png', lab2rgb(cur))
        return

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--image', help = 'link to image')
    parser.add_argument('-m', '--model', help = 'link to model')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    predictor = Predict()
    model = predictor.load_trained_model(args.model)
    predictor.predict(model, args.image)
    
