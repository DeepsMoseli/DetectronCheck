import numpy as np
import cv2
import pickle
import os
import random
from PIL import Image
import pandas as pd
from matplotlib import pyplot as plt
from flask import Flask
from flask import render_template
from model import trained_model

app = Flask(__name__)


image_path = os.getcwd().replace("scr",'images') + '/'
img = cv2.imread(image_path+'LC 1_4.jpg')

model1 = trained_model()

#to render stuff to index.html
@app.route('/')
@app.route('/index')
def index():
    # assuming you are predicting 2 values from your ml model

    im,outputs = model1.make_prediction(img)
    predicted_image = model1.show_prediction(im,outputs)
    #model.get_json_result(outputs)

    predictions = ['hello',0.67,'hi'] 
    class_name_val = predictions[0]
    probability_val = predictions[1]
    real_val = predictions[2] 

    return render_template('index.html',
                           class_name=class_name_val,
                           probability=probability_val,
                           real = real_val,
                           prediction_image = predicted_image)

app.run(debug=False)