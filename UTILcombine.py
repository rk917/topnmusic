# CS 467
# Group Music Genre - Duke Doan, Gerrit Van Ruiswyk, Robert King
# Simple utility to combine two existing h5 files, 6/2/21

# run this file with:
# python3 UTILcombine.py <file1name> <file2name> <combinedFileName>

import h5py_cache
import numpy as np
import sys


if __name__ == "__main__":
    # get the file names from the command line
    file1 = sys.argv[1]
    file2 = sys.argv[2]
    combined = sys.argv[3]

    print("combining", file1, "and", file2, "into", combined)

    # open the files with h5py_cache using a 250mb buffer for speed
    hf1 = h5py_cache.File(file1, "a", 250*1024**2)
    hf2 = h5py_cache.File(file2, "a", 250*1024**2)
    hfNew = h5py_cache.File(combined, "a", 250*1024**2)

    # load the x_train and y_train from the two files
    x1 = hf1['x_train'][:]
    y1 = hf1['y_train'][:]

    x2 = hf2['x_train'][:]
    y2 = hf2['y_train'][:]

    # combine the two x arrays
    xCombined = np.concatenate((x1, x2), axis=0)
    print("Combined X shape = ", xCombined.shape)

    # combine the two y arrays
    yCombined = np.concatenate((y1, y2), axis=0)
    print("Combined Y shape = ", yCombined.shape)

    # save the combined arrays to the new file
    hfNew.create_dataset("x_train", data=xCombined, dtype=np.uint8, chunks=True, compression='gzip', compression_opts=9)
    hfNew.create_dataset("y_train", data=yCombined, dtype=np.uint8, chunks=True, compression='gzip', compression_opts=9)

    hf1.close()
    hf2.close()
    hfNew.close()
    print("files combined!")
