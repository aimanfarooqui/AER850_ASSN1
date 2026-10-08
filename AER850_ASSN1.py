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
    X = np.array(X).astype(np.float64)
    Y = np.array(Y).astype(np.float64)
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

# %% Part (c): Degree 20 polynomial with L2 regularization (Ridge)
# Ridge minimizes  (1/2n)*||Xm*W - Y||^2 + (lambda/2)*(w1^2 + w2^2 + ... + w20^2) 
# (same scaling as Part (b) so the lambdas can be compared; bias w0 is again not penalized) 
# Multiplying by 2n, this is least squares on a stacked system: 
#   [ Xm            ]       [ Y ] 
#   [ sqrt(n*lam)*D ] W  =  [ 0 ]     where D is the identity with D[0,0] = 0 (no penalty on w0) 
# so unlike Lasso it can be solved in one step with lstsq 
def computeW_ridge(Xm_train, Y_train, lam):
    n, d = Xm_train.shape                        # n = 50 data points, d = 21 weights 
    D = np.eye(d)
    D[0, 0] = 0                                  # don't penalize the bias 
    Xm_aug = np.vstack([Xm_train, np.sqrt(n * lam) * D])   # add 21 penalty rows under Xm 
    Y_aug = np.vstack([Y_train, np.zeros((d, 1))])          # their targets are 0 
    W = np.linalg.lstsq(Xm_aug, Y_aug, rcond=None)[0]
    return W

# Try lambda = 0, plus 100 values from 1e-6 to 1 spaced on a log scale. 
# Log spacing is needed here: the best ridge lambda is ~0.0004, which a 0.01-step grid would skip over 
lambdas_c = np.concatenate(([0], np.logspace(-6, 0, 100)))
MSE_train_c = []
MSE_valid_c = []
for lam in lambdas_c:
    W = computeW_ridge(Xm_train, Y_train, lam)   # lambda = 0 gives plain least squares (Part a) 
    MSE_train_c.append(computeMSE(np.dot(Xm_train, W), Y_train))
    MSE_valid_c.append(computeMSE(np.dot(Xm_valid, W), Y_valid))
MSE_train_c = np.array(MSE_train_c)
MSE_valid_c = np.array(MSE_valid_c)

# The best lambda is the one with the lowest validation MSE 
best_c = np.argmin(MSE_valid_c)
best_lambda_c = lambdas_c[best_c]
print("Part (c) Best lambda:          ", best_lambda_c)
print("Part (c) Training MSE (best):  ", MSE_train_c[best_c])
print("Part (c) Validation MSE (best):", MSE_valid_c[best_c])

# Plot training and validation MSE against lambda 
plt.figure()
plt.plot(lambdas_c, MSE_train_c, label='Training MSE')
plt.plot(lambdas_c, MSE_valid_c, label='Validation MSE')
plt.axvline(best_lambda_c, color='k', linestyle='--', label='Best lambda = %.2g' % best_lambda_c)
plt.xscale('symlog', linthresh=1e-6)     # log scale that can still show lambda = 0
plt.xlim(0, 1)
plt.ylim(0, 60)                          # validation MSE at lambda = 0 is ~1400 
plt.xlabel('lambda')
plt.ylabel('MSE')
plt.title('Part (c): MSE vs lambda (degree 20, L2 regularization)')
plt.legend()
plt.show()

# Part (c): Test performance and fit at the best lambda
W_c = computeW_ridge(Xm_train, Y_train, best_lambda_c)
MSE_test_c = computeMSE(np.dot(Xm_test, W_c), Y_test)     # Xm_test was built in Part (b) 
print("Part (c) Test MSE (best lambda):", MSE_test_c)

# Ridge shrinks weights but doesn't set them to exactly 0 (compare with Part b) 
print("Part (c) Weights (w0 ... w20):", np.round(W_c[:, 0], 2))

y_plot_c = np.dot(getfeaturematrix(x_plot, degree), W_c)  # x_plot (all three datasets) was built in Part (b) 

plt.figure()
plt.scatter(X_train, Y_train, label='Training data')
plt.scatter(X_valid, Y_valid, label='Validation data')
plt.scatter(X_test, Y_test, label='Test data')
plt.plot(x_plot, y_plot_c, 'r', label='Degree 20 fit, lambda = %.2g' % best_lambda_c)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Part (c): Degree 20 polynomial with L2 regularization')
plt.legend()
plt.show()


# %% Part (d): Degree 20 polynomial with Elastic Net regularization
# Elastic Net mixes the L1 and L2 penalties with a ratio r: 
#   (1/2n)*||Xm*W - Y||^2 + lambda * [ r*(|w1| + ... + |w20|) + ((1-r)/2)*(w1^2 + ... + w20^2) ] 
# r = 0 -> pure Ridge (Part c),  r = 1 -> pure Lasso (Part b) 
# lambda is kept fixed at the best Lasso value from Part (b), and only r is varied. 
# Solved with the same coordinate descent as Part (b); the only change is the update for w_j: 
# soft-threshold by lambda*r (the L1 part), then divide by an extra lambda*(1-r) (the L2 part). 
def computeW_elasticnet(Xm_train, Y_train, lam, r, max_iter=100000, tol=1e-8):
    n, d = Xm_train.shape                        # n = 50 data points, d = 21 weights 
    W = np.zeros((d, 1))                         # start with all weights at 0 
    z = np.sum(Xm_train**2, axis=0) / n          # (1/n)*||column j||^2 for each column 
    residual = Y_train - np.dot(Xm_train, W)     # what's left to explain: Y minus prediction 
    for it in range(max_iter):
        max_change = 0
        for j in range(d):
            w_old = W[j, 0]
            rho = np.dot(Xm_train[:, j], residual[:, 0]) / n + z[j] * w_old
            if j == 0:
                w_new = rho / z[j]                                    # bias: no penalty 
            else:
                w_new = np.sign(rho) * max(abs(rho) - lam * r, 0) / (z[j] + lam * (1 - r))
            if w_new != w_old:                   # skip the work if the weight didn't move 
                residual = residual - Xm_train[:, [j]] * (w_new - w_old)
                W[j, 0] = w_new
                max_change = max(max_change, abs(w_new - w_old))
        if max_change < tol:                     # stop once the weights have settled 
            break
    return W

lambda_d = best_lambda_b                         # fixed lambda (best from Part b) 
r_values = np.linspace(0, 1, 101)                # r = 0, 0.01, ..., 1 
MSE_train_d = []
MSE_valid_d = []
for r in r_values:
    W = computeW_elasticnet(Xm_train, Y_train, lambda_d, r)
    MSE_train_d.append(computeMSE(np.dot(Xm_train, W), Y_train))
    MSE_valid_d.append(computeMSE(np.dot(Xm_valid, W), Y_valid))
MSE_train_d = np.array(MSE_train_d)
MSE_valid_d = np.array(MSE_valid_d)

# The best r is the one with the lowest validation MSE 
best_d = np.argmin(MSE_valid_d)
best_r = r_values[best_d]
print("Part (d) Fixed lambda:         ", lambda_d)
print("Part (d) Best r:               ", best_r)
print("Part (d) Training MSE (best):  ", MSE_train_d[best_d])
print("Part (d) Validation MSE (best):", MSE_valid_d[best_d])

# Plot training and validation MSE against r 
plt.figure()
plt.plot(r_values, MSE_train_d, label='Training MSE')
plt.plot(r_values, MSE_valid_d, label='Validation MSE')
plt.axvline(best_r, color='k', linestyle='--', label='Best r = %.2f' % best_r)
plt.xlabel('r  (0 = Ridge, 1 = Lasso)')
plt.ylabel('MSE')
plt.title('Part (d): MSE vs r (degree 20, Elastic Net, lambda = %.2f)' % lambda_d)
plt.legend()
plt.show()


# Test performance and fit at the best r
W_d = computeW_elasticnet(Xm_train, Y_train, lambda_d, best_r)
MSE_test_d = computeMSE(np.dot(Xm_test, W_d), Y_test)     # Xm_test was built in Part (b) 
print("Part (d) Test MSE (best r):", MSE_test_d)
print("Part (d) Weights (w0 ... w20):", np.round(W_d[:, 0], 2))
print("Part (d) Nonzero weights at powers:", np.nonzero(W_d[:, 0])[0])

y_plot_d = np.dot(getfeaturematrix(x_plot, degree), W_d)  # x_plot was built in Part (b) 

plt.figure()
plt.scatter(X_train, Y_train, label='Training data')
plt.scatter(X_valid, Y_valid, label='Validation data')
plt.scatter(X_test, Y_test, label='Test data')
plt.plot(x_plot, y_plot_d, 'r', label='Degree 20 fit, r = %.2f' % best_r)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Part (d): Degree 20 polynomial with Elastic Net')
plt.legend()
plt.show()
