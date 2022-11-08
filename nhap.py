from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np

N = 150
centers = [[2, 2], [7, 7]]
n_classes = len(centers)
data, labels = make_blobs(N, 
                          centers=np.array(centers),
                          random_state=1)
res = train_test_split(data, labels, 
                       train_size=0.8,
                       test_size=0.2,
                       random_state=12)
train_data, test_data, train_labels, test_labels = res 

svc = LinearSVC(max_iter = 10000)

svc.fit(train_data, train_labels) 
predicted = svc.predict(test_data)
accuracy = accuracy_score(predicted, test_labels)
print('Do chinh xac: %.0f%%' % (accuracy*100))

w = svc.coef_[0]
intercept = svc.intercept_[0]
a = -w[0] / w[1]

nhom_0 = []
nhom_1 = []
for i in range(150):
    if labels[i] == 0:
        nhom_0.append([data[i,0], data[i,1]])
    elif labels[i] == 1:
        nhom_1.append([data[i,0], data[i,1]])
nhom_0 = np.array(nhom_0)
nhom_1 = np.array(nhom_1)

plt.plot(nhom_0[:,0],nhom_0[:,1],'go', markersize=5)
plt.plot(nhom_1[:,0],nhom_1[:,1],'rs', markersize=5)

xx = np.linspace(2, 7)
yy = a * xx - intercept / w[1]
plt.plot(xx, yy, 'b')

xx = np.linspace(3, 8)                                                  
yy = a * xx - intercept / w[1]  + 0.5 / w[1]
plt.plot(xx, yy, 'b--')

xx = np.linspace(2, 6)                                                  
yy = a * xx - intercept / w[1]  - 0.5 / w[1]
plt.plot(xx, yy, 'b--')

plt.legend([0,1])
plt.show()




