import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

class SVMClassifier:
    def __init__(self, lr = 1e-3, lambda_param=1e-4, max_iter=10000):
        self.lr = lr
        self.lambda_param = lambda_param
        self.max_iter = max_iter
        self.W = None
        self.b = None

    def loss(self, x, y):
        hinge_loss = np.maximum(0, 1 - y * (np.dot(X, self.W) + self.b))
        l = self.lambda_param * (np.linalg.norm(self.W) ** 2) + np.mean(hinge_loss)
        return l

    def fit(self, X, y):
        self.W = np.ones(X.shape[1])
        self.b = 1

        for _ in range(self.max_iter):
            for i, xi in enumerate(X):
                err = 1 - y[i] *( np.dot(X[i], self.W) + self.b )
                if max(0, err) == 0:
                    self.W -= self.lr * 2 * self.lambda_param * self.W
                else:
                    self.W -= self.lr * (2 * self.lambda_param * self.W - y[i] * X[i] )
                    self.b -= self.lr * (-y[i])

    def predict(self, X):
        return np.sign(np.dot(X, self.W) + self.b)

    def get_accuracy(self, X, y):
        y_preds = self.predict(X)
        correct = y_preds == y
        return (100 * sum(correct) / len(y)).round(2)

if __name__=='__main__':
    diabetes = pd.read_csv('diabetes.csv').values
    x= diabetes[:,:-1]
    y= diabetes[:,-1]
    y= np.where(y==0, -1, 1)

    X_train, X_test, y_train, Y_test= train_test_split(x, y, test_size=0.2, random_state=42)
    model = SVMClassifier()
    model.fit(X_train, y_train)
    print('test result: ', model.get_accuracy(X_test, Y_test))

