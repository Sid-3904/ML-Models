import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from LinearRegression import LinearRegression

X, Y = datasets.make_regression(n_samples = 100, n_features = 1, noise = 20, random_state = 4)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 1234)
fig = plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], Y, color = 'b', marker = 'o', s = 30)
plt.show()

# LinearRegression Model
l_r = LinearRegression()
l_r.fit(X_train, Y_train)
predictions = l_r.predict(X_test)

def mse(Y_test, predictions) :
    return np.mean((Y_test-predictions)**2)

mse = mse(Y_test, predictions)
print('Accuracy: ', mse)

prediction_line = l_r.predict(X)
cmap = plt.get_cmap('viridis')
fig  = plt.figure(figsize=(8, 6))
m1 = plt.scatter(X_train, Y_train, color = cmap(0.9), s = 10)
m2 = plt.scatter(X_test, Y_test, color = cmap(0.5), s = 10)
plt.plot(X, prediction_line, color = 'black', linewidth = 1, label = 'Linear Regression Model')
plt.show()