from flask import Flask, render_template, request
import tensorflow as tf 
from predict import Predict
from PIL import Image
import argparse
import os
import uuid
import cv2
from preprocessing import themvien

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/image'
graph = None 
model = None 

#load colorization model
def load_model(model_path):
    global graph
    global model 
    graph = tf.get_default_graph()
    with graph.as_default():
        model = Predict(model_path)


@app.route('/')
def index():
    return render_template('demo.html', image = '', result = '')

@app.route('/', methods = ['POST'])
def colorize_image():
    if request.method == 'POST':
        image = Image.open(request.files['file'].stream)
        image_name = str(uuid.uuid4()) + '.jpg'
        image.save(os.path.join(app.config['UPLOAD_FOLDER'], image_name))
        #save image
        del image                                          
        path_image = os.path.join(app.config['UPLOAD_FOLDER'], image_name)
        image = themvien(path_image, (256, 256), 0)
        cv2.imwrite(path_image, image)
        with graph.as_default():
            result_name = model.predict(path_image)
        result = os.path.join(app.config['UPLOAD_FOLDER'],result_name)
        return render_template('demo.html', image = path_image, result = result)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', help = 'link to model')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    load_model(args.model)
    app.debug = True
    app.run(port = 5000)