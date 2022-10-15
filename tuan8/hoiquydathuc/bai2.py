import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

N = 30
N_test = 20 
def main():
    #khởi tạo bộ tạo số ngẫu nhiên
    np.random.seed(100)
    
    X_true = np.linspace(0, 5, 51)
    y_true = 3*(X_true -2) * (X_true - 3)*(X_true-4)
    
    X = np.random.rand(N, 1)*5
    y = 3*(X -2) * (X - 3)*(X-4) +  10*np.random.randn(N, 1)

    X_test = (np.random.rand(N_test,1) - 1/8) *10
    y_test = 3*(X_test -2) * (X_test - 3)*(X_test-4) +  10*np.random.randn(N_test, 1)

    plt.plot(X, y, 'ro', markersize = 3)
    plt.plot(X_true, y_true, 'y')
    
    poly_features = PolynomialFeatures(degree=2, include_bias=False)
    X_poly = poly_features.fit_transform(X)
    lin_reg = LinearRegression()
    lin_reg.fit(X_poly, y)
    
    print(lin_reg.intercept_, lin_reg.coef_)
    w0 = lin_reg.intercept_[0]
    w1 = lin_reg.coef_[0,0]
    w2 = lin_reg.coef_[0,1]
    
    y_predict = w0 + X_true*w1 + X_true**2*w2
    plt.plot(X_true, y_predict, 'b')

    #tinh sai so tren tap test
    X_test_poly = poly_features.fit_transform(X_test)
    y_test_predict = lin_reg.predict(X_test_poly)
    mse_test = mean_squared_error(y_test, y_test_predict)
    rmse_test = np.sqrt(mse_test)
    print('Sai so binh phuong trung binh - test: ')
    print('%.4f' % rmse_test)

    plt.show()
    
if __name__ == '__main__':
    main()