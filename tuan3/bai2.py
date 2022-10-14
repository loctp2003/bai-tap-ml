import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error
X = np.array([[147, 150, 153, 158, 163, 165, 168, 170, 173, 175, 178, 180, 183]]).T
y = np.array([ 49, 50, 51, 54, 58, 59, 60, 62, 63, 64, 66, 67, 68])

regr = linear_model.LinearRegression()
regr.fit(X, y) # in scikit-learn, each sample is one row
print("scikit-learn’s solution : w_1 = ", regr.coef_[0], "w_0 = ", regr.intercept_)
y_mu = regr.predict(X)
loss = mean_squared_error(y, y_mu)
loss = loss/2
print(loss)
