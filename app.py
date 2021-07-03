from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re, glob, os,cv2
import numpy as np
import pandas as pd
from object_detect import ObjectDetector
from shutil import copyfile
import shutil
from distutils.dir_util import copy_tree    

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__, template_folder='./template')

upload_folder = 'files'
output_folder = 'static/result/'
flask_output_folder = 'result/'
print('Model loaded. Check http://127.0.0.1:5000/')

weight = {
    "coco": './trained/yolov4-tiny-416',
    "udacity": './trained/yolov4-tiny-vehicles'
}

img_size = {
    "coco": 418,
    "udacity": 512
}

detector = ObjectDetector(image_size = 416, output = output_folder)

app = Flask(__name__, template_folder= 'template')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def upload_file():
    form = dict(request.form)
    datatype = form['dataset']
    uploaded_file = request.files['file']
    if uploaded_file.filename != '':
        uploaded_file.save(os.path.join(upload_folder, uploaded_file.filename))

        detector.load_weight(weight[datatype], datatype == 'coco', img_size[datatype])
        text_result =  detector.detect(image_path = os.path.join(upload_folder, uploaded_file.filename))

        print('detect done!')
        return render_template("index.html", result_img = os.path.join(output_folder, uploaded_file.filename), text= text_result)
    return render_template("index.html")  


if __name__ == '__main__':
    app.run(debug=False)