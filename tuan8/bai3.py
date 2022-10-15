
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

centers = [[2, 3], [5, 5], [1, 8]]
n_classes = len(centers)
data, labels = make_blobs(n_samples=150, 
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
    else :
        nhom_2.append([data[i,0], data[i,1]])
nhom_0 = np.array(nhom_0)
nhom_1 = np.array(nhom_1)
nhom_2 = np.array(nhom_2)

plt.plot(nhom_0[:,0], nhom_0[:,1], 'go', markersize = 3)
plt.plot(nhom_1[:,0], nhom_1[:,1], 'ro', markersize = 3)
plt.plot(nhom_2[:,0], nhom_2[:,1], 'bo', markersize = 3)

plt.show()

