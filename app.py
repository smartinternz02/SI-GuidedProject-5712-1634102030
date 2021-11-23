from __future__ import division, print_function
import sys
import os
import glob
from keras.applications.vgg16 import preprocess_input
import numpy as np
from tensorflow.keras.preprocessing import image
#from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model
#from keras import backend
import tensorflow as tf
global graph
#graph=tf.get_default_graph()
from skimage.transform import resize
# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer


# Define a flask app
app = Flask(__name__)
# Load your trained model
# Necessary
# print('Model loaded. Start serving...')
# You can also use pretrained model from Keras
# Check https://keras.io/applications/
#from keras.applications.resnet50 import ResNet50 #model = ResNet50(weights='imagenet') #model.save('')
#print('Model loaded. Check http://127.0.0.1:5000/')

@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('base.html')

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        file = request.files['image']
        filename=file.filename
        # Save the file to ./uploads
        #basepath = os.path.dirname(__file__)
        file_path = os.path.join('uploads', filename)
        file.save(file_path)

        model=load_model("CHEST.h5")
        img = image.load_img(file_path, target_size=(32,32))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        #with graph.as_default():
        img_data=preprocess_input(x)
        classes=model.predict(img_data)
        result=int(classes[0][0])

        if result==0:
            text = "NORMAL"
        else:
            text = "the person is affected by pneumonia"
        text = text
            # ImageNet Decode
        return text

if __name__ == '__main__':
    app.run(debug=True)
