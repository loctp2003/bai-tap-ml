from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np

N = 150
centers = [[2, 3], [5, 5], [1, 8]]
n_classes = len(centers)
data, labels = make_blobs(N, 
                          centers=np.array(centers),
                          random_state=1)

nhom_0 = []
nhom_1 = []
nhom_2 = []
for i in range(150):
    if labels[i] == 0:
        nhom_0.append([data[i,0], data[i,1]])
    elif labels[i] == 1:
        nhom_1.append([data[i,0], data[i,1]])
    else:
        nhom_2.append([data[i,0], data[i,1]])
nhom_0 = np.array(nhom_0)
nhom_1 = np.array(nhom_1)
nhom_2 = np.array(nhom_2)

plt.plot(nhom_0[:,0],nhom_0[:,1],'go', markersize=2)
plt.plot(nhom_1[:,0],nhom_1[:,1],'ro', markersize=2)
plt.plot(nhom_2[:,0],nhom_2[:,1],'bo', markersize=2)
plt.legend([0,1,2])
plt.show()

res = train_test_split(data, labels, 
                       train_size=0.8,
                       test_size=0.2,
                       random_state=12)
train_data, test_data, train_labels, test_labels = res 
knn = KNeighborsClassifier()
knn.fit(train_data, train_labels) 
predicted = knn.predict(test_data)
accuracy = accuracy_score(predicted, test_labels)
print('Do chinh xac: %.0f%%' % (accuracy*100))
