# CS 467
# Group Music Genre - Duke Doan, Gerrit Van Ruiswyk, Robert King
# Simple utility to describe the contents of an H5 file, 6/2/21

# run this file with:
# python3 UTILdescribe.py <h5fileName>

import h5py
import numpy as np
import sys

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
#           Main code
# ---------------------------------------
if __name__ == "__main__":
    
    # open the h5 file using the command line argument
    fileName = sys.argv[1]
    hf1 = h5py.File(fileName, "a")
    
    # load the y data set
    y1 = hf1['y_train'][:]

    # get the count of unique numbers from the 
    unique, counts = np.unique(y1, return_counts=True)
    # combine it in a dictionary
    yList = dict(zip(unique, counts))

    # print out the counts by genre
    for each in yList:
        print("#", each, genreLookup(each), "has", yList[each].item())

    # and the total number of files
    print("Total =", sum(counts))

    hf1.close()