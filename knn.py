import numpy as np
from collections import Counter

def euclidean_distance(x1, x2) :
    return np.sqrt(np.sum((x1-x2)**2))

class KNN :
    def __init__(self, k = 3) :
        self.k = k

    def fit(self, train_X, train_y) :
        self.train_X = train_X
        self.train_y = train_y

    def predict(self, test_X) :
        predictions = [self._predict(x) for x in test_X]
        return predictions

    def _predict(self, x) :
        distances = [euclidean_distance(x, train_x) for train_x in self.train_X]
        k_indices = np.argsort(distances)[: self.k]
        k_nearest_labels = [self.train_y[i] for i in k_indices]
        majority = Counter(k_nearest_labels).most_common()
        return majority[0][0]
    


