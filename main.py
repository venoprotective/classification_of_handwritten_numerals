import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from classMLP import NeuralNetMLP


num_epochs = 50
minibatch_size = 100

X, y = fetch_openml('mnist_784', version=1, 
                    return_X_y=True)

X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=10_000,
                                                  random_state=123, stratify=y)

X_train, X_valid, y_train, y_valid = train_test_split(X_temp, y_temp, test_size=5000,
                                                      random_state=123, stratify=y_temp)

def minibatch_generator(X, y, minibatch_size):
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    for start_idx in range(0, indices.shape[0] - minibatch_size + 1, minibatch_size):
        batch_idx = indices[start_idx:start_idx + minibatch_size]
        
        yield X[batch_idx], y[batch_idx]

def int_to_onehot(y, num_labels):
        ary = np.zeros((y.shape[0], num_labels))
        for i, val in enumerate(y):
            ary[i, val] = 1
        
        return ary
        
def mse_loss(targets, probas, num_labels=10):
    onehot_targets = int_to_onehot(targets, num_labels=num_labels)    
    return np.mean((onehot_targets - probas)**2)

def accuracy(targets, predicted_labels):
    return np.mean(predicted_labels == targets)

def compute_mse_and_acc(nnet : NeuralNetMLP, X, y, num_labels=10, minibatch_size=100):
    
    mse, correct_pred, num_examples = 0.0, 0, 0
    minibatch_gen = minibatch_generator(X, y, minibatch_size)
    
    for i, (features, targets) in enumerate(minibatch_gen):
        _, probas = nnet.forward(features)
        predicted_labels = np.argmax(probas, axis=1)
        onehot_targets = int_to_onehot(targets, num_labels=num_labels)
        loss = np.mean((onehot_targets - probas) ** 2)
        corrected_pred += (predicted_labels == targets).sum()
        num_examples += targets.shape[0]
        mse+=loss
        
    mse /=i
    acc = correct_pred / num_examples
    
    return mse, acc
    
model = NeuralNetMLP(num_features=28*28,  
                     num_hidden=50,
                     num_classes=10)
