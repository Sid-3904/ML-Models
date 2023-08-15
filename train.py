import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from knn import KNN

iris = datasets.load_iris()
X, y = iris.data, iris.target
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size = 0.2, random_state = 1234)

# LinearRegression Model
model = KNN(k = 5)
model.fit(X_train, Y_train)
predictions = model.predict(X_test)

def accuracy(Y_test, predictions) :
    return 100 * np.sum(predictions == Y_test) / len(Y_test)

acc = accuracy(Y_test, predictions)
print(f'Accuracy: {acc}%')

# prediction_line = l_r.predict(X)
# cmap = plt.get_cmap('viridis')
# fig  = plt.figure(figsize=(8, 6))
# m1 = plt.scatter(X_train, Y_train, color = cmap(0.9), s = 10)
# m2 = plt.scatter(X_test, Y_test, color = cmap(0.5), s = 10)
# plt.plot(X, prediction_line, color = 'black', linewidth = 1, label = 'Linear Regression Model')
# plt.show()