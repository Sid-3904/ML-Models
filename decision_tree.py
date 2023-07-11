import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# loading iris dataset
iris = load_iris()

# setting target and features
X = iris.data
y = iris.target
train_X, test_X, train_y, test_y = train_test_split(X, y, random_state = 1)

# setting up model
dt_model = DecisionTreeRegressor(random_state = 1)

# training testing and validating
dt_model.fit(train_X, train_y)
predictions = dt_model.predict(test_X)
accuracy = accuracy_score(y_true = test_y, y_pred = predictions)
print('test accuracy: ', accuracy)