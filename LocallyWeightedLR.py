import numpy as np

t = 1000

def weight(X):
    weight = np.array([float(2.7**(np.dot((X - X[i]).T, X - X[i])/(-2*t**2))) for i in X])
    return weight

class WeightedLinearRegression():
    def __init__(self, lr=0.01, iters=1000):
        self.lr = lr
        self.iters = iters
        self.weights = None
        self.bias = None

    def fit(self, X, Y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        for _ in range(self.iters):
            hypothesis = np.dot(X, self.weights) + self.bias
            self.weights -= (1/n_samples) * self.lr * np.dot(X.T, (hypothesis - Y))
            self.bias -= (1/n_samples) * self.lr * sum(hypothesis - Y)

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias

Model = WeightedLinearRegression()
X_test = np.array([[i] for i in range(0, 100)])
Y_test = np.array([i**2 for i in range(0, 100)])
Model.fit(np.array([[i] for i in range(0, 100)]), np.array([i**2 for i in range(0, 100)]))
print(Model.predict([6]))
