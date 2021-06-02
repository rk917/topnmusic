# CS 467
# Group Music Genre - Duke Doan, Gerrit Van Ruiswyk, Robert King
# Build H5 file by Genre, 6/2/21

# This program creates an H5 file containing numpy arrays of mel spectograms of wav or mp3 files 
# downloaded from a google cloud storage container

# run this code with:
# python3 1.buildH5-224.py <directoryName> <saveFileName> <filesToGet>
# where:
# <directoryName> is the google cloud directory (example: freeMusicArchive)
# <saveFileName> is the name of the genre to download (example: rock)
# note that the name of the saved file will be the genre name + .h5 (example, rock.h5)
# <filesToGet> is the number of files to grab (example: 100)

from google.cloud import storage
import io
import os
import h5py_cache
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing.image import img_to_array
from PIL import Image
import pydub
import time
import sys


# *****************************************************
#                GLOBAL VARIABLES
# *****************************************************
directoryName = "freeMusicArchive"      # Options include: trainingData, dddSpotMusic, freeMusicArchive, vanruisgMusic, dddM1000
limitFiles = True

# ---------------------------------------
# Function to convert MP3 to Wav
# returns bytes object with wav file
# requires FFMPEG installed on the system
# ---------------------------------------
def mp3ToWav(blob):
    downloadedFile = blob.download_to_filename("./temp.mp3") 
    # load the file into pydub and export as wav
    mp3 = pydub.AudioSegment.from_mp3("./temp.mp3")
    wav = mp3.export("./temp.wav", format="wav")    
    with open('./temp.wav', 'rb') as f:
        contents = f.read()
    #clean up the temp files
    os.remove("./temp.mp3")
    os.remove("./temp.wav")
    return contents

# ---------------------------------------
# Function to encode genre to integer
# returns integer representation of genre
# ---------------------------------------
def encodeGenre(genreStr):
    return {
        'cla': 0, # classical
        'cou': 1, # country
        'ele': 2, # electronic
        'hip': 3, # hiphop
        'jaz': 4, # jazz
        'blu': 5, # blues
        'roc': 6, # rock
    }.get(genreStr, 99) # default to 99 if not found

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
        
        # FOR TESTING
        # Uncomment this code to produce PNG files in directory
        #filename = "temp" + str(time.time()) + ".png"
        #skimage.io.imsave(filename, img)

        return img

# ---------------------------------------
# Funtion to save the h5py file
# ---------------------------------------
def saveH5(xData, yData, saveFileName):
    
    print("received xData with shape=", xData.shape)
    print("received yData with shape=", yData.shape)

    # save a new file in append mode. This uses h5py_cache with a 500mb buffer to speed up the operation
    h5f = h5py_cache.File(saveFileName, "a", 500*1024**2)
    
    # create the datasets in int8 format, using maximum gzip compression
    h5f.create_dataset("x_train", data=xData, dtype=np.uint8, chunks=True, compression='gzip', compression_opts=9)
    h5f.create_dataset("y_train", data=yData, dtype=np.uint8, chunks=True, compression='gzip', compression_opts=9)
    print("Saving successful.")

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
# Function to count files in the google storage bucket by genre
# ---------------------------------------
def countFiles(genre):
    # iterage through each name, checking to see if it contains the genre name
    count = 0
    for each in storage_client.list_blobs('cs467', prefix=directoryName):
        if genre in each.name:
            # check to make sure they are either mp3 or wav files, and not some other file with the name
            if 'mp3' in each.name:
                count += 1
            if 'wav' in each.name:
                count += 1
    return count

# ---------------------------------------
#           Main code
# ---------------------------------------
if __name__ == "__main__":

    # record the starting time
    startLoop = time.time()

    # open the google cloud storage bucket
    storage_client, bucket = connectToStorage()
    
    # Uncomment if getting the file count is helpful
    # fileCount = countFiles(sys.argv[1])
    
    # Grab the arguments from the command line
    directoryName = sys.argv[1]
    saveFileName = sys.argv[2] + ".h5"
    filesToGet = int(sys.argv[3])
    print("saving to ", saveFileName)

    # create two numpy arrays of the right size full of zeros
    xData = np.zeros((filesToGet, 224, 224, 3))
    yData = np.zeros((filesToGet))

    print("x shape =", xData.shape)
    print("y shape =", yData.shape)

    # counters used in the for loop
    counter = 0
    fileCount = 1

    # iterate through each file in the cloud container
    for each in storage_client.list_blobs('cs467', prefix=directoryName):
        
        # find each file with a matching genre name (such as "rock")
        if sys.argv[2] in each.name:

            # record the time
            startConv = time.time()

            # create a bytes type object
            wavFile = b''

            # if it is an mp3...
            if 'mp3' in each.name:
                # convert it using the mp3ToWav function
                tempBlob = bucket.blob(each.name)
                wavFile = mp3ToWav(tempBlob)
                # then turn the wavFile into a numpy array
                xData[counter] = convertWav(wavFile)
                # then save the first three letters of the file name ("rock00012.mp3" becomes "roc") to yData
                tempStr = each.name[len(directoryName)+1: len(each.name)-10]
                tempStr = tempStr[0:3]
                yData[counter] = encodeGenre(tempStr)

            # if it is a wav file, same as above minus the mp3ToWav conversion
            if 'wav' in each.name:
                tempBlob = bucket.blob(each.name)
                wavFile = tempBlob.download_as_bytes()
                xData[counter] = convertWav(wavFile)
                tempStr = each.name[len(directoryName)+1: len(each.name)-10]
                tempStr = tempStr[0:3]
                yData[counter] = encodeGenre(tempStr)

            counter += 1
            endConv = time.time()
            print(counter, each.name, "saved file to numpy array in", round(endConv-startConv, 2), "seconds")
            fileCount += 1

            # End the loop when the number of files is reached.
            # this can also be changed to grab all the files in the directory, but could result in unbalanced genres
            if limitFiles:
                if fileCount > filesToGet:
                    break


    print("Done processing data, saving to h5 file")
    saveH5(xData, yData, saveFileName)
    endLoop = time.time()
    print("End of program. Total runtime =", round(endLoop-startLoop, 2))

