import numpy as np
from matplotlib import pyplot as plt

X = np.array([[147, 150, 153, 158, 163, 165, 168, 170, 173, 175, 178, 180, 183]])
mot = np.ones((1, 13), dtype = np.int32)
X_bar = np.vstack((mot, X))
X_bar_T = X_bar.T
A = np.matmul(X_bar, X_bar_T)
y = np.array([[ 49, 50, 51, 54, 58, 59, 60, 62, 63, 64, 66, 67, 68]]).T
b = np.matmul(X_bar, y)
A_inv = np.linalg.pinv(A)
w = np.matmul(A_inv, b)
print(w)

x1 = X[0, 0]
y1 = x1*w[1, 0] + w[0, 0]
x2 = X[0, -1]
y2 = x2*w[1, 0] + w[0, 0]

plt.plot(X, y.T, 'ro')
plt.plot([x1, x2], [y1, y2])
sample = 155
ket_qua = sample*w[1, 0] + w[0, 0]
print('Chieu cao: %d thi can nang la: %d' % (sample, ket_qua))
sample = 160
ket_qua = sample*w[1, 0] + w[0, 0]
print('Chieu cao: %d thi can nang la: %d' % (sample, ket_qua))

#Tinh sai so
loss = 0
sai_so = 0;
for i in range(0, 13):
    y_mu = np.matmul(X_bar[:,i],w)
    sai_so = y[i, 0] - y_mu
    loss = loss + sai_so**2
loss = loss/(2*13)
print(loss)
#Tinh sai so ham norm
loss = np.linalg.norm(y - np.matmul(X_bar_T, w))
loss = loss**2
loss = loss/(2*13)
print(loss)

plt.show()

#loss (hàm tổn thất tính sai số)