# CS 467
# Group Music Genre - Duke Doan, Gerrit Van Ruiswyk, Robert King
# Test the model, 6/2/21

# This program loads a saved model and a test data set
# and provides detailed feedback on accuracy to the user

# run this code with:
# python3 3.testModel.py 

import h5py_cache
import numpy as np
from keras.models import load_model

# *****************************************************
#                GLOBAL VARIABLES
# *****************************************************
MODELFILE = "xception-224-model5"
MODELTYPE = "xception"             # options: sequential, inception, vgg16, xception. Note: must be set to the same as the original model.
PREDICTDATA = "test224.h5"
TARGET = 6                         # Provides detailed feedback about one genre only so it isn't overwhelming.  #6 = Rock

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
#               Main Code
# ---------------------------------------
if __name__ == "__main__":

    # load the model
    model = load_model(MODELFILE)

    # Open up the predict dataset from an H5py file
    h5f2 = h5py_cache.File(PREDICTDATA, "r", 250*1024**2)
    xNew = np.array(h5f2['x_train'])
    
    # preprocess the data, which is required for some models
    if MODELTYPE == "inception":
        from keras.applications.inception_v3 import preprocess_input
        xNumpyRGB = preprocess_input(xNew)

    if MODELTYPE == "vgg16":
        from keras.applications.vgg16 import preprocess_input
        xNumpyRGB = preprocess_input(xNew)

    if MODELTYPE == 'xception':
        from keras.applications.xception import preprocess_input
        xNumpyRGB = preprocess_input(xNew)

    if MODELTYPE == 'resnet50':
        from keras.applications.resnet_v2 import preprocess_input
        xNumpyRGB = preprocess_input(xNew)

    # load y values
    yNew = np.array(h5f2['y_train'])
    
    # run model predictions and close file
    yPredict = model.predict(xNumpyRGB)
    h5f2.close()    
    
    # collect some information about the number of correct and incorrect
    correct = [0, 0, 0, 0, 0, 0, 0, 0]
    incorrect = [0, 0, 0, 0, 0, 0, 0, 0]

    # gather the number of correct and incorrect guesses (TOP1)
    # also print detailed feedback about the genre specified above using the TARGET global variable
    for i in range(len(yNew)):

        # formating for detailed feedback        
        if yNew[i] == TARGET:
            print("\n---------------------------------------------------------------")

        # record correct and incorrects in TOP1 format
        # print feedback if this is the genre specified
        if yNew[i] == (np.argmax(yPredict[i])):
            correct[yNew[i]] += 1
            if yNew[i] == TARGET:
                print(i, "Correct! ", end='') 
        else:
            incorrect[yNew[i]] += 1
            if yNew[i] == TARGET:
                print(i, "Incorrect. ", end='')

        if yNew[i] == TARGET:
            print("actual:", genreLookup(yNew[i]), "predicted:", genreLookup(np.argmax(yPredict[i])), "\n")                  
            count = 0
            # show the match for each genre for the specified file (example, blues = 50 percent, rock = 10 percent, etc)
            for each in np.nditer(yPredict[i]):
                print(count, genreLookup(count), '=', np.round(each*100, decimals=1), "percent")
                count += 1
               

    # finally, print out the overall accuracy of the corrects/incorrects in Top 1 format
    print("--------------------------------------------")
    print(MODELFILE, MODELTYPE)
    for i in range(len(correct)):
        if (correct[i]+incorrect[i]) > 0:
            print(genreLookup(i), "correct:", int(correct[i] / (correct[i] + incorrect[i]) * 100), "percent using ", int(correct[i]+incorrect[i]), "files.")
    #
    print("overall correct:", int(sum(correct) / ( sum(correct) + sum(incorrect) ) * 100), "percent using", int(sum(correct)+sum(incorrect)), "files.")
  