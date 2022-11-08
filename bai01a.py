from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
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

svc = SVC(kernel="linear")

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

margin = 1 / np.sqrt(np.sum(svc.coef_**2))
yy_down = yy - np.sqrt(1 + a**2) * margin
yy_up = yy + np.sqrt(1 + a**2) * margin

plt.plot(xx, yy, 'b')
plt.plot(xx, yy_down, 'b--')
plt.plot(xx, yy_up, 'b--')

plt.plot(svc.support_vectors_[:, 0],
         svc.support_vectors_[:, 1], 'bs  ')

plt.legend([0,1])
plt.show()




