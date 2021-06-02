# CS 467
# Group Music Genre - Duke Doan, Gerrit Van Ruiswyk, Robert King
# Simple utility to change the genres in an existing h5 file, 6/2/21

# run this file with:
# python3 UTILreclass.py <oldH5fileName> <newH5fileName>

import h5py_cache
import numpy as np
import sys

if __name__ == "__main__":

    # get the arguments from the command line
    file1 = sys.argv[1]
    reclass = sys.argv[2]

    print("reclassifying", file1)

    # opent the old file
    hf1 = h5py_cache.File(file1, "a", 250*1024**2)

    # open the new file
    hfNew = h5py_cache.File(reclass, "a", 250*1024**2)

    # load the data
    x1 = hf1['x_train'][:]
    y1 = hf1['y_train'][:]


    # iterate through the y series, reclassifying numbers as needed
    # could also be used to collapse multiple genres into one
    for each in range(0, len(y1)):
        if y1[each] == 5:  # change all 5s to 4s
            y1[each] = 4
        if y1[each] == 6:  # change all 6s to 5s, etc
            y1[each] = 5
        if y1[each] == 7:
            y1[each] = 6
        if y1[each] == 9:
            y1[each] = 7

    # create the datasets in the new h5 file
    hfNew.create_dataset("x_train", data=x1, dtype=np.uint8, chunks=True, compression='gzip', compression_opts=9)
    hfNew.create_dataset("y_train", data=y1, dtype=np.uint8, chunks=True, compression='gzip', compression_opts=9)

    hf1.close()
    hfNew.close()
    print("file reclassified!")
