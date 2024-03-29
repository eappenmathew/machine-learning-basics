from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris

iris = load_iris()

features = iris.data
labels = iris.target

from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=.3)
knn = LogisticRegression()
knn.fit(x_train, y_train)

p = knn.predict(x_test)

from sklearn.metrics import accuracy_score
print("Accuracy=",accuracy_score(y_test, p))
