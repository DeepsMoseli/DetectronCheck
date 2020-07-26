import numpy as np
import matplotlib
matplotlib.use('Agg')
import cv2
import pickle
import os
import random
from PIL import Image
import pandas as pd
from flask import Flask
from flask import render_template, request, redirect, url_for
from pdf2image import convert_from_path, convert_from_bytes
from model import trained_model
from pdf2image.exceptions import (
    PDFInfoNotInstalledError,
    PDFPageCountError,
    PDFSyntaxError
)

IMAGE_FOLDER = os.path.join('static','input_folder')
CSV_FOLDER = os.path.join('static','csv_results')


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = IMAGE_FOLDER
app.config['CSV_FOLDER'] = CSV_FOLDER


model1 = trained_model()

#to render stuff to index.html
@app.route('/',methods=['GET','POST'])
def file_upload():
    return render_template('file_upload.html')


@app.route('/index',methods=['GET','POST'])
def index():
    dynamic_filename = lambda x: os.path.join(app.config['UPLOAD_FOLDER'],'%s'%x)
    
    full_filename = dynamic_filename('test1.jpg')
    #take pdf and save it as image
    if request.method =='POST':
        f = request.files['file']
        f.save(f.filename)
        image = convert_from_bytes(open(f.filename, 'rb').read())[0]
        full_filename = dynamic_filename(f.filename.replace('pdf','jpg'))
        image.save(full_filename)
        keep_just_name = f.filename
        del image, f

        img = cv2.imread(full_filename)
        #run prediction
        im,outputs = model1.make_prediction(img)
        predicted_image = model1.show_prediction(im,outputs)
        full_filename = full_filename.replace(".jpg","_pred.jpg")
        predicted_image.save(full_filename)
        
        #save csv (this could be also done from the front end as I have shown a button, but it doesnt work i ddnt connect it)
        json_df = model1.get_json_result(outputs)
        csv_name = os.path.join(app.config['CSV_FOLDER'],'%s'%keep_just_name.replace("jpg","csv").replace("pdf","csv"))
        json_df.to_csv(csv_name)

    return render_template('index.html',
                           tables=[json_df.to_html(classes='data')],
                           titles=json_df.columns.values,
                           prediction_image = full_filename)


if __name__ == '__main__':
    app.run(debug=True)
