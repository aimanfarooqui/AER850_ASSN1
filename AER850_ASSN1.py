# %% Setup: imports
import numpy as np
import csv
import matplotlib.pyplot as plt

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

# %% Setup: helper functions
# Builds the polynomial feature matrix: columns are x^0, x^1, ..., x^polynomial 
def getfeaturematrix(X, polynomial):
    Xm = []
    for i in range(0, polynomial + 1):
        a = np.power(X, i)                  # raise every x to the power i 
        Xm.append(a)
    Xm = np.array(Xm)
    Xm = np.squeeze(Xm, axis=(2,)).T        # rearrange into (data points) x (polynomial + 1) 
    return Xm

# Finds the weights W that minimize the squared error between Xm*W and Y 
# lstsq solves this directly and is more stable than inverting Xm^T Xm 
def computeW(Xm_train, Y_train):
    W = np.linalg.lstsq(Xm_train, Y_train, rcond=None)[0]
    return W

# Mean square error: average of the squared differences between prediction and target 
def computeMSE(Hypothesis, Y):
    MSE = np.mean((Hypothesis - Y)**2)
    return MSE

# %% Part (a): Degree 20 polynomial, no regularization
degree = 20

# Build feature matrices for the training and validation data #
Xm_train = getfeaturematrix(X_train, degree)
Xm_valid = getfeaturematrix(X_valid, degree)
print("Feature matrix size:", Xm_train.shape)   # should be (50, 21) #

# Find the weights using the training data only #
W_a = computeW(Xm_train, Y_train)

# Use the weights to predict y for the training and validation data #
Hypothesis_train = np.dot(Xm_train, W_a)
Hypothesis_valid = np.dot(Xm_valid, W_a)

# Compute and print the training and validation MSE #
MSE_train_a = computeMSE(Hypothesis_train, Y_train)
MSE_valid_a = computeMSE(Hypothesis_valid, Y_valid)
print("Part (a) Training MSE:  ", MSE_train_a)
print("Part (a) Validation MSE:", MSE_valid_a)

# Make 300 evenly spaced x values across the training AND validation range, so we can draw a smooth curve #
# (the largest validation x is past the largest training x, and that's where the fit blows up) #
x_both = np.vstack([X_train, X_valid])
x_plot = np.linspace(x_both.min(), x_both.max(), 300).reshape(-1, 1)

# Use the fitted weights to predict y at each of those x values #
y_plot = np.dot(getfeaturematrix(x_plot, degree), W_a)

# Plot the data as dots and the fitted polynomial as a line #
plt.figure()
plt.scatter(X_train, Y_train, label='Training data')
plt.scatter(X_valid, Y_valid, label='Validation data')
plt.plot(x_plot, y_plot, 'r', label='Degree 20 fit')
plt.ylim(Y_train.min() - 10, Y_train.max() + 10)   # keep the plot readable if the curve shoots off #
plt.xlabel('x')
plt.ylabel('y')
plt.title('Part (a): Degree 20 polynomial, no regularization')
plt.legend()
plt.show()