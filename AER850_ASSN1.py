# %% Setup: imports
import numpy as np
import csv
import matplotlib.pyplot as plt

# %% Read the data
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

# %% Load the data
X_train, Y_train = getdata("Dataset_1_train.csv")
X_valid, Y_valid = getdata("Dataset_1_valid.csv")
X_test,  Y_test  = getdata("Dataset_1_test.csv") 

print(X_train.shape, X_valid.shape, X_test.shape) #ensuring data is being read

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

# Build feature matrices for the training and validation data 
Xm_train = getfeaturematrix(X_train, degree)
Xm_valid = getfeaturematrix(X_valid, degree)
print("Feature matrix size:", Xm_train.shape)   # should be (50, 21) 

# Find the weights using the training data only 
W_a = computeW(Xm_train, Y_train)

# Use the weights to predict y for the training and validation data 
Hypothesis_train = np.dot(Xm_train, W_a)
Hypothesis_valid = np.dot(Xm_valid, W_a)

# Compute and print the training and validation MSE 
MSE_train_a = computeMSE(Hypothesis_train, Y_train)
MSE_valid_a = computeMSE(Hypothesis_valid, Y_valid)
print("Part (a) Training MSE:  ", MSE_train_a)
print("Part (a) Validation MSE:", MSE_valid_a)

# Making 300 evenly spaced x values across the training AND validation range, so we can draw a smooth curve 
# (the largest validation x is past the largest training x, and that's where the fit blows up) 
x_both = np.vstack([X_train, X_valid])
x_plot = np.linspace(x_both.min(), x_both.max(), 300).reshape(-1, 1)

# Using the fitted weights to predict y at each of those x values 
y_plot = np.dot(getfeaturematrix(x_plot, degree), W_a)

# Plotting the data as dots and the fitted polynomial as a line 
plt.figure()
plt.scatter(X_train, Y_train, label='Training data')
plt.scatter(X_valid, Y_valid, label='Validation data')
plt.plot(x_plot, y_plot, 'r', label='Degree 20 fit')
plt.ylim(Y_train.min() - 10, Y_train.max() + 10)   # keep the plot readable if the curve shoots off 
plt.xlabel('x')
plt.ylabel('y')
plt.title('Part (a): Degree 20 polynomial, no regularization')
plt.legend()
plt.show()

# %% Part (b): Degree 20 polynomial with L1 regularization (Lasso)
# Lasso minimizes  (1/2n)*||Xm*W - Y||^2 + lambda*(|w1| + |w2| + ... + |w20|) 
# The bias w0 is not penalized: it only shifts the curve up/down, it can't cause overfitting 
# |w| has a sharp corner at 0, so there's no closed-form solution like least squares. 
# Instead we use coordinate descent: update one weight at a time while holding the others 
# fixed, and keep sweeping through all the weights until they stop changing. 
def computeW_lasso(Xm_train, Y_train, lam, max_iter=100000, tol=1e-8):
    n, d = Xm_train.shape                        # n = 50 data points, d = 21 weights 
    W = np.zeros((d, 1))                         # start with all weights at 0 
    z = np.sum(Xm_train**2, axis=0) / n          # (1/n)*||column j||^2 for each column 
    residual = Y_train - np.dot(Xm_train, W)     # what's left to explain: Y minus prediction 
    for it in range(max_iter):
        max_change = 0
        for j in range(d):
            w_old = W[j, 0]
            # rho: how well column j explains the residual (with weight j's own part added back) 
            rho = np.dot(Xm_train[:, j], residual[:, 0]) / n + z[j] * w_old
            if j == 0:
                w_new = rho / z[j]                                    # bias: no penalty 
            else:
                w_new = np.sign(rho) * max(abs(rho) - lam, 0) / z[j]  # soft-thresholding: 
                                                                      # if |rho| < lambda, weight is set to exactly 0 
            if w_new != w_old:                   # skip the work if the weight didn't move (e.g. stuck at 0) 
                residual = residual - Xm_train[:, [j]] * (w_new - w_old)  # update residual for the new weight 
                W[j, 0] = w_new
                max_change = max(max_change, abs(w_new - w_old))
        if max_change < tol:                     # stop once the weights have settled 
            break
    return W

# Try lambda = 0, 0.01, 0.02, ..., 1 (takes ~30 seconds) 
lambdas_b = np.linspace(0, 1, 101)
MSE_train_b = []
MSE_valid_b = []
for lam in lambdas_b:
    if lam == 0:
        W = computeW(Xm_train, Y_train)          # lambda = 0 is just Part (a), no penalty 
    else:
        W = computeW_lasso(Xm_train, Y_train, lam)
    MSE_train_b.append(computeMSE(np.dot(Xm_train, W), Y_train))
    MSE_valid_b.append(computeMSE(np.dot(Xm_valid, W), Y_valid))
MSE_train_b = np.array(MSE_train_b)
MSE_valid_b = np.array(MSE_valid_b)

# The best lambda is the one with the lowest validation MSE 
best_b = np.argmin(MSE_valid_b)
best_lambda_b = lambdas_b[best_b]
print("Part (b) Best lambda:          ", best_lambda_b)
print("Part (b) Training MSE (best):  ", MSE_train_b[best_b])
print("Part (b) Validation MSE (best):", MSE_valid_b[best_b])

plt.figure()
plt.plot(lambdas_b, MSE_train_b, label='Training MSE')
plt.plot(lambdas_b, MSE_valid_b, label='Validation MSE')
plt.axvline(best_lambda_b, color='k', linestyle='--', label='Best lambda = %.2f' % best_lambda_b)
plt.ylim(0, 60)          # validation MSE at lambda = 0 is ~1400, which would squash the rest of the plot 
plt.xlabel('lambda')
plt.ylabel('MSE')
plt.title('Part (b): MSE vs lambda (degree 20, L1 regularization)')
plt.legend()
plt.show()

# %% Part (b): Test performance and fit at the best lambda
W_b = computeW_lasso(Xm_train, Y_train, best_lambda_b)
Xm_test = getfeaturematrix(X_test, degree)
MSE_test_b = computeMSE(np.dot(Xm_test, W_b), Y_test)
print("Part (b) Test MSE (best lambda):", MSE_test_b)

# Lasso sets many weights to exactly 0: show which powers of x survived (useful for part e) 
print("Part (b) Weights (w0 ... w20):", np.round(W_b[:, 0], 2))
print("Part (b) Nonzero weights at powers:", np.nonzero(W_b[:, 0])[0])

# Draw the curve over the range of all three datasets 
x_all = np.vstack([X_train, X_valid, X_test])
x_plot = np.linspace(x_all.min(), x_all.max(), 300).reshape(-1, 1)
y_plot_b = np.dot(getfeaturematrix(x_plot, degree), W_b)

plt.figure()
plt.scatter(X_train, Y_train, label='Training data')
plt.scatter(X_valid, Y_valid, label='Validation data')
plt.scatter(X_test, Y_test, label='Test data')
plt.plot(x_plot, y_plot_b, 'r', label='Degree 20 fit, lambda = %.2f' % best_lambda_b)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Part (b): Degree 20 polynomial with L1 regularization')
plt.legend()
plt.show()

# %% Part (c) 

# Ridge minimizes ||Xm*W - Y||^2 + lambda*||W||^2 
# Same as least squares on a stacked system: [Xm; sqrt(lambda)*I] W = [Y; 0] 
def computeW_ridge(Xm_train, Y_train, lam):                        # function that solves ridge regression
    n_w = Xm_train.shape[1]                                        # number of weights (21) 
    Xm_aug = np.vstack([Xm_train, np.sqrt(lam) * np.eye(n_w)])     # add 21 penalty rows under Xm 
    Y_aug = np.vstack([Y_train, np.zeros((n_w, 1))])               # their targets are 0 
    W = np.linalg.lstsq(Xm_aug, Y_aug, rcond=None)[0]
    return W

# Trying 200 lambda values spread evenly on a log scale from 1e-6 to 10 
lambdas = np.logspace(-6, 1, 200)
MSE_train_b = []
MSE_valid_b = []
for lam in lambdas:
    W = computeW_ridge(Xm_train, Y_train, lam)                     # fit on training data only 
    MSE_train_b.append(computeMSE(np.dot(Xm_train, W), Y_train))
    MSE_valid_b.append(computeMSE(np.dot(Xm_valid, W), Y_valid))
MSE_train_b = np.array(MSE_train_b)
MSE_valid_b = np.array(MSE_valid_b)

# The best lambda is the one with the lowest validation MSE 
best = np.argmin(MSE_valid_b)          # index of the smallest validation MSE 
best_lambda = lambdas[best]
print("Part (c) Best lambda:          ", best_lambda)
print("Part (c) Training MSE (best):  ", MSE_train_b[best])
print("Part (c) Validation MSE (best):", MSE_valid_b[best])

# Plotting MSE againts lambda
plt.figure()
plt.semilogx(lambdas, MSE_train_b, label='Training MSE')       # log scale on the x axis 
plt.semilogx(lambdas, MSE_valid_b, label='Validation MSE')
plt.axvline(best_lambda, color='k', linestyle='--', label='Best lambda = %.3g' % best_lambda)
plt.xlabel('lambda')
plt.ylabel('MSE')
plt.title('Part (c): MSE vs lambda (degree 20, L2 regularization)')
plt.legend()
plt.show()



