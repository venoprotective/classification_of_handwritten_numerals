import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def int_to_onehot(y, num_labels):
    
    ary = np.zeros((y.shape[0], num_labels))
    for i, val in enumerate(y):
        ary[i, val] = 1
    
    return ary

X, y = fetch_openml('mnist_784', version=1, 
                    return_X_y=True)

X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=10_000,
                                                  random_state=123, stratify=y)

X_train, X_valid, y_train, y_valid = train_test_split(X_temp, y_temp, test_size=5000,
                                                      random_state=123, stratify=y_temp)

