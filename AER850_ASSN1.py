# %% Setup: imports
import numpy as np
import csv
# import matplotlib.pyplot as plt

# %% Setup: function to read the data
def getdata(filename):
    with open(filename, 'r') as csvfile:
        Dataset_1_test = csv.reader(csvfile)
        X = []
        Y = []
        for row in Dataset_1_test:
            X.append(row[0])
            Y.append(row[1])
    X = np.array(X).astype(np.float32)
    Y = np.array(Y).astype(np.float32)
    X = np.reshape(X, (-1, 1))
    Y = np.reshape(Y, (-1, 1))
    return X, Y

# %% Setup: load the data
X_train, Y_train = getdata("Dataset_1_train.csv")
X_valid, Y_valid = getdata("Dataset_1_valid.csv")
X_test,  Y_test  = getdata("Dataset_1_test.csv")
print(X_train.shape, X_valid.shape, X_test.shape)