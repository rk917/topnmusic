# CS 467
# Group Music Genre - Duke Doan, Gerrit Van Ruiswyk, Robert King
# Fit the model, 6/2/21

# This program fits a number of different models with various features
# using the data loaded from the h5 file

# run this code with:
# python3 2.fitModel.py 

import os
import gc
from google.cloud import storage
import h5py
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import backend as k
from tensorflow.keras.callbacks import Callback
import keras
from keras.applications import InceptionV3
from keras.applications import VGG16
from keras.applications import Xception
from keras.applications import ResNet50V2
from keras.models import save_model
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.applications.resnet_v2 import ResNet50V2
from tensorflow.python.keras.layers.noise import GaussianNoise
from tensorflow.python.keras.layers.normalization import BatchNormalization

# *****************************************************
#                GLOBAL VARIABLES
#
# Change these settings to modify:
# -> the type of model to be fit
# -> the hyperparameters of the model
# -> whether to use early stopping or not
# -> the resulting model file name
# *****************************************************
MODELTYPE = "xception"                # options: sequential, inception, vgg16, xception, resnet50
WEIGHTS = 'imagenet'                  # for use in vgg16, inceptionv3, or xception options: None, 'imagenet'
OPTIMIZERTYPE = "adam"                # options: rms, adam
TRAINANDTESTDATA = "train224.h5"      # file name for local copy of training data
MODELFILE = "xception-224-model6"     # name of model to be saved

LEARNINGRATE = 0.00001                # hyperparameters and model features
USEDROPOUT = True
DROPOUTRATE = 0.5
USEBATCHNORMALIZATION = True
USENOISE = True
NOISELEVEL = 0.05

EPOCHS = 300
BATCHSIZE = 16

DOWNLOAD = False                # set to True for h5 download from Cloud, otherwise will use local copy

EARLYSTOPPING = False
STOPPATIENCE = 3

# ---------------------------------------
# Function returns the name of the genre
# ---------------------------------------
def genreLookup(genreNum):
    return {
        0: 'classical', # classical
        1: 'country', # country
        2: 'electronic', # electronic
        3: 'hiphop', # hiphop
        4: 'jazz', # jazz
        5: 'blues', # blues
        6: 'rock', # rock
    }.get(genreNum, "unknown")

# ---------------------------------------
# Function to connect to the google storage bucket
# returns the storage_client and bucket
# ---------------------------------------
def connectToStorage():
    # set up the google storage client and open the storage bucket
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = 'geometric-shore.json'
    storage_client = storage.Client()
    bucket = storage_client.get_bucket('cs467')
    return storage_client, bucket

# ---------------------------------------
# Funtion to open the h5py file
# returns two data sets for x_train and y_train
# ---------------------------------------
def openH5py():
    # open the h5 file in read mode
    h5f = h5py.File(TRAINANDTESTDATA, "r")
    # get the names of the data sets in the file
    dsNames = h5f.keys()
    # check to see if x_train is already in the file.  If so, we'll add to it.
    if "x_train" in dsNames:
        xNumpy = np.array(h5f['x_train'])
        yNumpy = np.array(h5f['y_train'])
    else:
        print("Error! x_train data not found!")
    h5f.close()
    return xNumpy, yNumpy

# ---------------------------------------
# Function to stop memory leak in model.fit()
# Found as a resolution to an issue in the following github discussion:
# https://github.com/tensorflow/tensorflow/issues/31312
# ---------------------------------------
class ClearMemory(Callback):
    def on_epoch_end(self, epoch, logs=None):
        gc.collect()
        k.clear_session()

# ---------------------------------------
#           Main code
# ---------------------------------------
if __name__ == "__main__":
    
    # download the h5 file from google storage
    if DOWNLOAD:
        storage_client, bucket = connectToStorage()
        blob = bucket.blob("week9/train224.h5")
        blob.download_to_filename(TRAINANDTESTDATA)
        print("h5 file downloaded")

    # open the h5 file and return the datasets as numpy arrays 
    xNumpyRGB, yNumpy = openH5py()

    # preprocess the data, which is required for some models
    if MODELTYPE == "inception":
        from keras.applications.inception_v3 import preprocess_input
        xNumpyRGB = preprocess_input(xNumpyRGB)

    if MODELTYPE == "vgg16":
        from keras.applications.vgg16 import preprocess_input
        xNumpyRGB = preprocess_input(xNumpyRGB)

    if MODELTYPE == 'xception':
        from keras.applications.xception import preprocess_input
        xNumpyRGB = preprocess_input(xNumpyRGB)

    if MODELTYPE == 'resnet50':
        from keras.applications.resnet_v2 import preprocess_input
        xNumpyRGB = preprocess_input(xNumpyRGB)


    # split the data into test and training sets
    X_train, X_test, y_train, y_test = train_test_split(xNumpyRGB, yNumpy, random_state=42, test_size=0.20)
    
    # clean up memory
    del xNumpyRGB 
    del yNumpy

    # Builds the model as set by the global variables

    # -------------------
    # INCEPTION MODEL
    # -------------------
    if MODELTYPE == "inception":
        # create a base model with input shape matching our data
        base_model = InceptionV3(input_shape=(224, 224, 3), include_top=False, weights=WEIGHTS)
        # add a flattening layer
        x = layers.Flatten()(base_model.output)
        
        # add layers as set by the global variables above
        if USEDROPOUT:
            x = layers.Dense(60, activation='relu')(x)
            x = layers.Dropout(DROPOUTRATE)(x)
        if USEBATCHNORMALIZATION:
            x = layers.BatchNormalization()(x)
        if USENOISE:
            x = layers.GaussianNoise(NOISELEVEL)(x)
        
        # final dense layer, which matches the number of genres/categories
        x = layers.Dense(7, activation='softmax')(x)
        model = tf.keras.models.Model(base_model.input, x)

    # Remaining model code is similar to the above

    # -------------------
    # XCEPTION MODEL
    # -------------------
    if MODELTYPE == "xception":
        base_model = Xception(input_shape=(224, 224, 3), include_top=False, weights=WEIGHTS)
        x = layers.Flatten()(base_model.output)
        if USEDROPOUT:
            x = layers.Dense(60, activation='relu')(x)
            x = layers.Dropout(DROPOUTRATE)(x)
        if USEBATCHNORMALIZATION:
            x = layers.BatchNormalization()(x)
        if USENOISE:
            x = layers.GaussianNoise(NOISELEVEL)(x)
        x = layers.Dense(7, activation='softmax')(x)
        model = tf.keras.models.Model(base_model.input, x)

    # -------------------
    # ResNet50V2 MODEL
    # -------------------
    if MODELTYPE == "resnet50":
        base_model = ResNet50V2(input_shape=(224, 224, 3), include_top=False, weights=WEIGHTS)
        x = layers.Flatten()(base_model.output)
        if USEDROPOUT:
            x = layers.Dense(60, activation='relu')(x)
            x = layers.Dropout(DROPOUTRATE)(x)
        if USEBATCHNORMALIZATION:
            x = layers.BatchNormalization()(x)
        if USENOISE:
            x = layers.GaussianNoise(NOISELEVEL)(x)
        x = layers.Dense(7, activation='softmax')(x)
        model = tf.keras.models.Model(base_model.input, x)

    # -------------------
    # VGG16 MODEL
    # -------------------
    if MODELTYPE == "vgg16":
        x = layers.Flatten()(base_model.output)
        base_model = VGG16(input_shape=(224, 224, 3),include_top=False, weights=WEIGHTS)
        if USEDROPOUT:
            x = layers.Dense(60, activation='relu')(x)
            x = layers.Dropout(DROPOUTRATE)(x)
        if USEBATCHNORMALIZATION:
            x = layers.BatchNormalization()(x)
        if USENOISE:
            x = layers.GaussianNoise(NOISELEVEL)(x)
        x = layers.Dense(7, activation='softmax')(x)
        model = tf.keras.models.Model(base_model.input, x)

    # -------------------
    # Sequential MODEL
    # this model does not use transfer learning and can be highly customized
    # however, performance was found to be subpar with extensive experimentation
    # -------------------
    if MODELTYPE == "sequential":
        model = keras.Sequential()
        model.add(layers.Flatten())
        model.add(layers.Dense(299, activation='relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.GaussianNoise(NOISELEVEL))
        model.add(layers.Dense(299, activation='relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(rate=DROPOUTRATE))
        model.add(layers.Dense(299, activation='relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.GaussianNoise(NOISELEVEL))
        model.add(layers.Dense(299, activation='relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.Dense(8, activation='softmax'))

    # add either the adam or rms optimizer             
    if OPTIMIZERTYPE == "adam":
        optimiser = keras.optimizers.Adam(learning_rate=LEARNINGRATE)

    if OPTIMIZERTYPE == "rms":
        optimiser = keras.optimizers.RMSprop(learning_rate=LEARNINGRATE)
    
    # compile the model
    model.compile(optimizer=optimiser,
                run_eagerly=True,
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

    # early stopping
    es = EarlyStopping(patience=STOPPATIENCE, monitor='val_loss', mode='min', restore_best_weights=True, verbose=1)

    # train model
    # if early stopping, add that as a call back, otherwise run without es
    if EARLYSTOPPING:
        model.fit(X_train, y_train, shuffle=True, callbacks=[ClearMemory(), es], validation_data=(X_test, y_test), batch_size=BATCHSIZE, epochs=EPOCHS)
    else:
        model.fit(X_train, y_train, shuffle=True, callbacks=[ClearMemory()], validation_data=(X_test, y_test), batch_size=BATCHSIZE, epochs=EPOCHS)

    del X_train
    del X_test
    del y_train
    del y_test

    # remove the download h5 file
    if DOWNLOAD:
        os.remove(TRAINANDTESTDATA)

    # save the model
    save_model(model, MODELFILE) 
