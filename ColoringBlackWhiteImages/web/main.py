from flask import Flask, render_template, request
import tensorflow as tf 
from color_model.predict import Predict
from PIL import Image
import argparse

app = Flask(__name__)

graph = None 
model = None 

#load colorization model
def load_model(model_path):
    global graph
    global model 
    graph = tf.get_default_graph()
    with graph.as_default():
        model = Predict(model_path)

app.route('/', methods = ['GET', 'POST'])
def colorize_image():
    if request.method == 'GET':
        return render_template('demo.html', image = '', result = '')
    else:
        image = Image.open(request.files['file'].stream)
        image.save('../color_model/' + request.files['file'].filename)
        #save image
        with graph.as_default:
            model.predict(path_image)
        result = '../color_model/result.png'
    return render_template('demo.html', image = path_image, result = result)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', help = 'link to model')
    return parser.parse_args()

