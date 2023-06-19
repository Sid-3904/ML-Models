import numpy as np

class LinearRegression() :
    def __init__(self, lr=0.01, iters=1000) :
        self.lr = lr
        self.iters = iters
        self.weights = None
        self.bias = None

    def fit(self, X, Y) :
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        for _ in range(self.iters) :
            hypothesis = np.dot(X, self.weights) + self.bias
            self.weights -= (1/n_samples) * self.lr * np.dot(X.T, (hypothesis - Y))  
            self.bias -= (1/n_samples) * self.lr * sum(hypothesis - Y)

    def predict(self, X) :
        return np.dot(X, self.weights) + self.bias