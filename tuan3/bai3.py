import numpy as np
from sklearn import datasets, linear_model, metrics
from sklearn.metrics import mean_squared_error
X = np.array([[ 0,  1],
       [ 5,  1],
       [15,  2],
       [25,  5],
       [35, 11],
       [45, 15],
       [55, 34],
       [60, 35]])
y = np.array([[4, 5, 20, 14, 32, 22, 38, 43]]).T

regr = linear_model.LinearRegression()
regr.fit(X, y) # in scikit-learn, each sample is one row
print(regr.coef_)
print(regr.intercept_)

y_mu = regr.predict(X)
loss = mean_squared_error(y, y_mu)
loss = loss/2
print(loss)