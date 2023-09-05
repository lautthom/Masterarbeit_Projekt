import numpy as np
import matplotlib.pyplot as plt

data_train = np.load('/home/lautthom/Desktop/bioviddatasetfiles-master/bioviddatasetfiles-master/PartA/Train/GSR/120717-09_w_23_data.npy')
labels_train = np.load('/home/lautthom/Desktop/bioviddatasetfiles-master/bioviddatasetfiles-master/PartA/Train/GSR/120717-09_w_23_label.npy')
data_test = np.load('/home/lautthom/Desktop/bioviddatasetfiles-master/bioviddatasetfiles-master/PartA/Test/GSR/120717-09_w_23_data.npy')
labels_test = np.load('/home/lautthom/Desktop/bioviddatasetfiles-master/bioviddatasetfiles-master/PartA/Test/GSR/120717-09_w_23_label.npy')
print(data_train.shape)
print(data_test.shape)
print(labels_train.shape)
print(labels_test.shape)
#print(data_test)
#print(labels_test)

datapoint = 150
#print(labels_train)
print(labels_train[datapoint])

fig, ax = plt.subplots()
ax.plot(range(1152), data_train[datapoint])
plt.show()




