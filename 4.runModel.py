# CS 467
# Group Music Genre - Duke Doan, Gerrit Van Ruiswyk, Robert King
# Test the model, 6/2/21

# This program loads a single mp3 or wav file and runs the saved model on it

# run this code with:
# python3 4.runModel.py <musicFile>

# where <musicFile> is the name of an mp3 or wav file in the local directory (example: rock.wav)

import io
import os
import sys
import numpy as np
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from keras.models import load_model
from keras.preprocessing.image import img_to_array
import pydub

# *****************************************************
#                GLOBAL VARIABLES
# *****************************************************
MODELFILE = "xception-224-model5"  # name of the saved model
MODELTYPE = "xception"             # options: sequential, inception, vgg16, xception. Note: must be set to the same as the original model.


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
# Function to convert MP3 to Wav
# returns bytes object with wav file
# ---------------------------------------
def mp3ToWav(blob):
    downloadedFile = blob.download_to_filename("./temp.mp3")
    mp3 = pydub.AudioSegment.from_mp3("./temp.mp3")
    wav = mp3.export("./temp.wav", format="wav")    
    with open('./temp.wav', 'rb') as f:
        contents = f.read()
    #clean up the temp files
    os.remove("./temp.mp3")
    os.remove("./temp.wav")
    return contents

# ---------------------------------------
# Funtion to scale array between 0 and 1
# This function was sourced from the following discussion on github:
# https://github.com/pytorch/audio/issues/1023
# ---------------------------------------
def scale_minmax(X, min=0.0, max=1.0):
    X_std = (X - X.min()) / (X.max() - X.min())
    X_scaled = X_std * (max - min) + min
    return X_scaled

# ---------------------------------------
# Funtion to convert wav files to numpy arrays
# This function was also inspired by the above github discussion, 
# but was significantly modified
# ---------------------------------------
def convertWav(wavFile):
    # load the wavFile as a BytesIO file
    with io.BytesIO(wavFile) as f:
        # load the file into librosa
        y, sr = librosa.load(f)
        # build a mel spectogram and take the log
        mels = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=224, n_fft=1024, hop_length=2954)
        mels = np.log(mels + 1e-9)
        # scale it from 0 to 1 and convert it to int8 format
        mels2 = scale_minmax(mels, 0, 255).astype(np.uint8)
        # plot the spectogram in a 224x224 figure
        plt.figure(figsize=(2.24,2.24))
        librosa.display.specshow(mels2, sr=sr, x_axis='time', y_axis='mel')
        # create a new BytesIO and save the figure to this buffer
        buffer = io.BytesIO()
        # these are necessary to clip the edges of the mel spectogram and not waste space
        plt.axis('off')  
        plt.tight_layout(pad=0)
        plt.savefig(buffer, format='png', bbox_inches=None, pad_inches=0)
        # These calls clean up memory and are needed to prevent a leak and slowdown
        plt.close()
        plt.cla()
        # open the buffer in PIL and convert to a numpy array
        img = Image.open(buffer)
        img = img_to_array(img)
        # drop the alpha layer
        img = img[:,:,:3]
        return img

# ---------------------------------------
#               Main Code
# ---------------------------------------
if __name__ == "__main__":

    # load the model
    model = load_model(MODELFILE)

    # open up the input file and convert to a numpy array
    fileName = sys.argv[1]

    # create a bytes file and load the wav or mp3 file to it
    contents = b''
    with open(fileName, 'rb') as f:
        contents = f.read()
    f.close()

    # convert a wav file
    if 'wav' in fileName:
        testNumpy = convertWav(contents)

    if 'mp3' in fileName:
        wavFile = mp3ToWav(contents)
        testNumpy = convertWav(wavFile)

    # add a dimension so it matches the model
    testNumpy = np.expand_dims(testNumpy, axis=0)

    # preprocess the data, which is required for some models
    if MODELTYPE == "inception":
        from keras.applications.inception_v3 import preprocess_input
        xNumpyRGB = preprocess_input(testNumpy)

    if MODELTYPE == "vgg16":
        from keras.applications.vgg16 import preprocess_input
        xNumpyRGB = preprocess_input(testNumpy)

    if MODELTYPE == 'xception':
        from keras.applications.xception import preprocess_input
        xNumpyRGB = preprocess_input(testNumpy)
           
    # run model predictions and close file
    yPredict = model.predict(xNumpyRGB, verbose=1)   
    print("prediction complete")
    
    # display the results to the user
    count = 0
    for each in np.nditer(yPredict):
        print(count, genreLookup(count), '=', np.round(each*100, decimals=1), "percent")
        count += 1
    