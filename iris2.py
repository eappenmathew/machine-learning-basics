from sklearn import neighbors, datasets
from sklearn.datasets import load_iris

iris = datasets.load_iris()

features = iris.data
labels = iris.target

from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=.3)
knn = neighbors.KNeighborsClassifier()
k=knn.fit(x_train, y_train)

p = k.predict(x_test)

from sklearn.metrics import accuracy_score
print("Accuracy=",accuracy_score(y_test, p))

