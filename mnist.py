import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

#data = pd.read_csv("train.csv")

#print("Data\n",data)

data = pd.read_csv("train.csv").as_matrix()

clf = DecisionTreeClassifier()

x_train = data[0:21000, 1:]
y_train = data[0:21000, 0]

x_test = data[21000:, 1:]
y_test = data[21000:, 0]

#clf.fit(x_train, y_train)

#for ploting the pic
immg=x_train[4:5,:]
disp = immg.reshape([28,28])
plt.imshow(disp,cmap=plt.get_cmap("gray_r"))
plt.show()

p = clf.predict([x_test[8]])
