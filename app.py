from __future__ import division, print_function
# coding=utf-8
import os
from cv2 import data

import tensorflow as tf
from tensorflow.python.ops.gen_array_ops import reverse_sequence

from object_detect import ObjectDetector
from custom_tracker import ObjectTracker 

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__, template_folder='./template', static_folder='static')

upload_folder = 'files'
output_folder = 'static/result/'
flask_output_folder = 'result/'
print('Model loaded. Check http://127.0.0.1:5000/')

weight = {
    'coco': './trained/tf2.2/yolov4-tiny-416',
    'udacity': './trained/yolov4-tiny-vehicles'
}

print(tf.__version__)

img_size = {
    'coco': 418,
    'udacity': 512
}

detector = ObjectDetector(output = output_folder)

tracker = ObjectTracker(output = output_folder)

app = Flask(__name__, template_folder= 'template', static_folder= 'static')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect', methods=['GET','POST'])
def upload_file():
    if(request.method == 'GET'):
        return render_template("detect.html")
        
    form = dict(request.form)
    print(form)
    print(form['dataset'])
    datatype = form['dataset']
    uploaded_file = request.files['file']
    if uploaded_file.filename != '':
        uploaded_file.save(os.path.join(upload_folder, uploaded_file.filename))
        print(weight[datatype])
        print(img_size[datatype])
        detector.load_weight(weight[datatype], datatype == 'coco', img_size[datatype])
        text_result =  detector.detect(image_path = os.path.join(upload_folder, uploaded_file.filename))
        print('detect done!')
        return render_template("detect.html", result_img = os.path.join(output_folder, uploaded_file.filename), text = text_result)

    return render_template("detect.html")  

@app.route('/tracking', methods = ['GET','POST'])
def track():
    if(request.method == 'GET'):
        return render_template("tracking.html")

    form = dict(request.form)
    datatype = form['dataset']
    uploaded_file = request.files['file']
    if uploaded_file.filename != '':
        filename = os.path.join(upload_folder, uploaded_file.filename)
        uploaded_file.save(filename)
        tracker.load_yolo_weight(weight[datatype], datatype == 'coco', img_size[datatype])
        tracker.track(video_path = filename)
        return render_template("tracking.html", result_video = os.path.join(output_folder, uploaded_file.filename), file_video = 'cars.mp4')
        

if __name__ == '__main__':
    app.run(debug = False)