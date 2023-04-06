#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

model_LOR = pickle.load(open('model_LOR.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')




@app.route('/diabetes', methods=['POST', 'GET'])
def rdiabetes():
	return render_template('resultd.html')


@app.route('/resultd.html', methods=['POST', 'GET'])
def diabetes():
     float_features = [float(x) for x in request.form.values()]
     final_features = [np.array(float_features)]
     prediction = model_LOR.predict(final_features)
     if prediction == 0:
          pred ='NON - DIABETIC'
     elif prediction == 1:
          pred = 'DIABETIC'
     output = pred
     return render_template('resultd.html', prediction_text = 'This person is {}'.format(output))



if __name__ == "__main__":
    app.run(debug=True)

