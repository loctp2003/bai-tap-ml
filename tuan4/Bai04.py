from sklearn import linear_model
import numpy as np
from sklearn.metrics import mean_squared_error

X = np.array([[35.3, 29.7, 30.8, 58.8, 61.4, 71.3, 74.4, 76.7, 70.7, 57.5, 46.4, 28.9, 28.1, 39.1, 46.8, 48.5, 59.3, 70.0, 70.0, 74.5, 72.1, 58.1, 44.6, 33.4, 28.6],
              [20, 20, 23, 20, 21, 22, 11, 23, 21, 20, 20, 21, 21, 19, 23, 20, 22, 22, 11, 23, 20, 21, 20, 20, 22]]).T

y = np.array([10.98, 11.13, 12.51, 8.40, 9.27, 8.73, 6.36, 8.50, 7.82, 9.14, 8.24, 12.19, 11.88, 9.57, 10.94, 9.58, 10.09, 8.11, 6.83, 8.88, 7.68, 8.47, 8.86, 10.36, 11.08])

regr = linear_model.LinearRegression()
regr.fit(X, y) 
w1 = regr.coef_[0]
w2 = regr.coef_[1]
b = regr.intercept_
print('b  = %.4f' % b)
print('w1 = %.4f' % w1)
print('w2 = %.4f' % w2)

y_mu = regr.predict(X)
print(y_mu)

sai_so = mean_squared_error(y,y_mu)
N = X.shape[0]
print(sai_so*N)

