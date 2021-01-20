#!/usr/bin/env python
# coding: utf-8

from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Dropout, concatenate, Input, Conv1D, GlobalMaxPooling1D, Embedding, BatchNormalization
from keras.optimizers import SGD
from keras.models import Model


from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


from tqdm import tqdm
import numpy as np
import joblib
from . import txt_processor
import codecs
import matplotlib.pyplot as plt
import calendar
import time
import os

####AUTHOR OF THIS FILE: DUY NGUYEN NGOC


#This function will return lemmatized data using POS-tagging and an Tokenizer object that represents the
#whole text corpus. We will fit this Tokenizer on the training set and the testing set to vectorize the text
#into sequence of integers later on (Read final report for more information on Featurization)
def preprocess_data(data, path, tokenizer = None):
    tokens = txt_processor.split_words(data.posts)
    print("Data are tokens now")
    data['tokens'] = tokens
    sentence_lengths = [len(tokens) for tokens in data["tokens"]]
    print("Max sentence length is %s" % max(sentence_lengths))
    data['sentence_lengths'] = sentence_lengths
    print("about to lemmatize")
    lemmatized_posts = txt_processor.lemmatize(tokens)
    data['lemmatized_posts'] = lemmatized_posts
    print(data.head())
    
    if tokenizer == None:
        tokenizer = txt_processor.create_tokenizer(data, path)
    return data, tokenizer


# In[ ]:

#This function returns stratified split of training data/target and testing data/target.
def split_data(data):
    print("In spilt Data")
    y = data["binarized_target"].values
    print(y)
    try:
        data_train, data_test, y_train, y_test = train_test_split(data, y, test_size=0.10, random_state=500, stratify=y)
    except Exception as e:
        print(e)
    print("about to return")
    return data_train, data_test, y_train, y_test


#Fit Tokenizer on training data and testing data and pad the sequence with MAX_SEQUENCE_LENGTH
#so everyone sequence has the same length
def prepare_cnn_data(data_train, data_test, tokenizer):
    MAX_SEQUENCE_LENGTH = 1200
    train_cnn_data = txt_processor.tokenize(data_train, tokenizer, MAX_SEQUENCE_LENGTH)
    test_cnn_data = txt_processor.tokenize(data_test, tokenizer, MAX_SEQUENCE_LENGTH)
    train_word_index = tokenizer.word_index
    return train_cnn_data, test_cnn_data, train_word_index

#Definition of the Convolutional Neural Network ML model to fit the training data
def CNNModel(embedding_matrix, max_sequence_length, num_words, embedding_dim):
    inputs = Input(shape=(max_sequence_length, ))
    embedding_layer = Embedding(num_words,embedding_dim, weights=[embedding_matrix], input_length=max_sequence_length, trainable=True)(inputs)
    convs = []
    filter_sizes = [1,2,3,5]

    for filter_size in filter_sizes:
        l_conv = Conv1D(filters=256, kernel_size=filter_size, activation='relu')(embedding_layer)
        l_pool = GlobalMaxPooling1D()(l_conv)
        convs.append(l_pool)

    l_merge = concatenate(convs, axis=1)
    y = Dense(128, activation='relu')(l_merge)
    y = Dropout(0.1)(y)  
    preds = Dense(1, activation='sigmoid')(y)
    model = Model(inputs, preds)
    model.compile(loss='binary_crossentropy',
                 optimizer='adam',
                 metrics=['acc'])
    model.summary()
    return model


#This function calls on the preprocessing of the initial data and also prepares the pre-trained embedding matrix
#on the word_index of the Tokenizer object (Read the final report for more information)
def prepareTrain(data, path):
    data, tokenizer = preprocess_data(data, path)
    print("done preprocessing")
    embeddings_index = {}
    try:
        f = codecs.open('./prediction/system/data/wiki-news-300d-1M.vec', encoding='utf-8')
        print("opened the file")
    except Exception as e:
        print(e)
    for line in tqdm(f):
        values = line.rstrip().rsplit(' ')
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    print('found %s word vectors' % len(embeddings_index))
    return data, embeddings_index, tokenizer

#Fit the data into the Machine Learning pipeline (defined in the final report) and train them.
def trainModel(data, tokenizer, embeddings_index, character, path):
    MAX_SEQUENCE_LENGTH = 1200
    EMBEDDING_DIM = 300
    data = txt_processor.split_target_variable(data, character)
    print("split of targets done")
    data = txt_processor.binarize_target_variable(data)
    print("binarizetion of targets ")
    # data['sentence_lengths'].hist(bins = 30)
    undersampled_data = txt_processor.undersample(data)
    print("undersampled data is done")
    print(undersampled_data.head())
    X_train, X_test, y_train, y_test = split_data(undersampled_data)
    print("dataset Splitting is done")
    cnn_train, cnn_test, word_index = prepare_cnn_data(X_train, X_test, tokenizer)
    
    print('preparing embedding matrix...')
    words_not_found = []
    nb_words = min(999999, len(word_index) + 1)
    embedding_matrix = np.zeros((nb_words, EMBEDDING_DIM))
    for word, i in word_index.items():
        if i >= nb_words:
            continue
        embedding_vector = embeddings_index.get(word)
        if (embedding_vector is not None) and len(embedding_vector) > 0:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
        else:
            words_not_found.append(word)
    print('number of null word embeddings: %d' % np.sum(np.sum(embedding_matrix, axis=1) == 0))    
    


    filepath=path + 'weights_' + str(character) + '.best.hdf5'
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]
    
    model = CNNModel(embedding_matrix, MAX_SEQUENCE_LENGTH, len(word_index)+1, EMBEDDING_DIM)
    hist = model.fit(cnn_train, y_train, epochs=4, validation_data=(cnn_test, y_test), shuffle=True, callbacks=callbacks_list)

    

    filename = path + str(character) + '.sav'
    joblib.dump(model, filename)

    y_pred = model.predict(cnn_test)

    print("About to check how well it went")
    y_pred_bool = [1 * (x[0]>=0.5) for x in y_pred]


    return classification_report(y_test, y_pred_bool, output_dict=True)

#Train 4 models for 4 different characters of the personality type
def trainAllModel(data, path):
    prepared_data, embeddings_index, tokenizer = prepareTrain(data, path)
    print("Data has been prepaired")
    eva_val0 = trainModel(prepared_data, tokenizer, embeddings_index, 0, path)
    eva_val1 = trainModel(prepared_data, tokenizer, embeddings_index, 1, path)
    eva_val2 = trainModel(prepared_data, tokenizer, embeddings_index, 2, path)
    eva_val3 = trainModel(prepared_data, tokenizer, embeddings_index, 3, path)
    print("about to return evals")
    return eva_val0, eva_val1, eva_val2, eva_val3

#This function returns the classification report of a model for evaluation purposes.
def evaluateNewModel(data, character, model, tokenizer):
    MAX_SEQUENCE_LENGTH = 1200
    data, tokenizer = preprocess_data(data, tokenizer)
    data = txt_processor.split_target_variable(data, character)
    print("split of targets done")
    data = txt_processor.binarize_target_variable(data)
    print("binarizetion of targets ")
    undersampled_data = txt_processor.undersample(data)
    y = undersampled_data["binarized_target"].values
    X = txt_processor.tokenize(undersampled_data, tokenizer, MAX_SEQUENCE_LENGTH)
    print("About to predict!")
    
    model._make_predict_function()
    y_pred = model.predict(X)

    print("About to check how well it went")
    y_pred_bool = [1 * (x[0]>=0.5) for x in y_pred]


    return classification_report(y, y_pred_bool, output_dict=True)

### ALL FUNCTIONS BELOW CAN BE USED FOR FUTURE USE. THE FUNCTIONS CONTAIN FEATURIZATION OF THE TABULAR DATA (no. of CAPS, TF-IDF, etc.)
### WE CAN FIT 2 TYPES OF INPUT: TABULAR INPUT AND TEXT SEQUENCE FOR BETTER ACCURACY.

def prepare_tabular_data(data, data_train, data_test, y_train):
    tabular_train_data, tabular_test_data = txt_processor.tabular_features(data, data_train, data_test)
    tabular_train_data, tabular_test_data = txt_processor.tabular_scaler(tabular_train_data, tabular_test_data)
    tabular_train_best_data, tabular_test_best_data = txt_processor.chi2_features(tabular_train_data, tabular_test_data, y_train)
    return tabular_train_best_data, tabular_test_best_data



def hybrid_model(embedding_matrix, max_sequence_length, num_words, embedding_dim):
    
    inputA = Input(shape=(100,))
    inputB = Input(shape=(max_sequence_length, ))
    
    # Tabular data branch
    x = Dense(2, activation="relu")(inputA)
    x = Model(inputs=inputA, outputs=x)   
    
    
    # CNN data branch
    embedding_layer = Embedding(num_words,embedding_dim, weights=[embedding_matrix], input_length=max_sequence_length, trainable=True)(inputB)
    
    convs = []
    filter_sizes = [1,2,3,5]

    for filter_size in filter_sizes:
        l_conv = Conv1D(filters=256, kernel_size=filter_size, activation='relu')(embedding_layer)
        l_pool = GlobalMaxPooling1D()(l_conv)
        convs.append(l_pool)

    l_merge = concatenate(convs, axis=1)
    y = Dense(128, activation='relu')(l_merge)
    y = Dropout(0.1)(y)  
    y = Dense(2, activation="relu")(y)
    y = Model(inputs=inputB, outputs=y)
    
    combined = concatenate([x.output, y.output])
    
    z = BatchNormalization()(combined)
    

    preds = Dense(1, activation='sigmoid')(z)

    model = Model(inputs = [x.input, y.input], outputs=preds)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['acc'])
    model.summary()
    return model


def testModel():
    inputA = Input(shape=(100,))    
    preds = Dense(1, activation="sigmoid")(inputA)
    model = Model(inputA, preds)
    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
    model.summary()
    return model

