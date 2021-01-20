# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
# ms-python.python added
from __future__ import division, print_function
import os

try:
	os.chdir(os.path.join(os.getcwd(), 'web-app\\mbti_web\prediction\system'))
	print(os.getcwd())
except:
	pass
from keras import backend as K
import pandas as pd
from . import txt_processor
import pickle

#### AUTHOR OF THIS FILE: DUY NGUYEN NGOC


###Make prediction from a certain input
def makePrediction(tokenizer, input, graph0, graph1, graph2, graph3,
             session0, session1, session2, session3, model_load_0, model_load_1, model_load_2, model_load_3):
    print(input)
    data = pd.DataFrame(columns = ['input', 'lemmatized_posts'])
    data['input'] = [input]
    tokens = txt_processor.split_words(data.input)
    lemmatized_input = txt_processor.lemmatize(tokens)
    data['lemmatized_posts'] = lemmatized_input
    print("about to tokenize")
    input_cnn_data = txt_processor.tokenize(data, tokenizer, 1200)
    print(data.head())
    print("about to predict")
    print(input_cnn_data)
    
    with graph0.as_default():
        with session0.as_default():
            model_load_0._make_predict_function()
            prediction_0 = model_load_0.predict(input_cnn_data)
    K.clear_session()

    with graph1.as_default():
        with session1.as_default():
            model_load_1._make_predict_function()
            prediction_1 = model_load_1.predict(input_cnn_data)
    K.clear_session()

    with graph2.as_default():
        with session2.as_default():
            model_load_2._make_predict_function()
            prediction_2 = model_load_2.predict(input_cnn_data)
    K.clear_session()

    with graph3.as_default():
        with session3.as_default():
            model_load_3._make_predict_function()
            prediction_3 = model_load_3.predict(input_cnn_data)
    K.clear_session()

    print(prediction_0 , prediction_1, prediction_2 , prediction_3)

    character_0 = predictCharacter(prediction_0, 0)
    character_1 = predictCharacter(prediction_1, 1)
    character_2 = predictCharacter(prediction_2, 2)
    character_3 = predictCharacter(prediction_3, 3)

    prediction = character_0 + character_1 + character_2 + character_3
    values = [prediction_0[0] , prediction_1[0], prediction_2[0], prediction_3[0]]
    return prediction, values


#This function returns the right character based on the character and the float value from the prediction result.
def predictCharacter(output, character):
    if character == 0:
        if output >= 0.5:
            return 'I'
        else:
            return 'E'
    elif character == 1:
        if output >= 0.5:
            return 'N'
        else:
            return 'S'
    elif character == 2:
        if output >= 0.5:
            return 'F'
        else:
            return 'T'    
    elif character == 3:
        if output >= 0.5:
            return 'P'
        else:
            return 'J'






