import numpy as np

from classMLP import NeuralNetMLP
from func_module import train, compute_mse_and_acc
from data import X_train, y_train, X_valid, y_valid, num_epochs, minibatch_size, X_test, y_test


model = NeuralNetMLP(num_features=28*28,  
                     num_hidden=50,
                     num_classes=10)

np.random.seed(123)
epoch_loss, epoch_train_acc, epoch_valid_acc = train(model, X_train, y_train, X_valid, y_valid, 
                                                     num_epochs, learning_rate=0.1, minibatch_size=minibatch_size)

# промежуточная оценка точности модели в цифрах
test_mse, test_acc = compute_mse_and_acc(model, X_test, y_test)
print(f'Accuracy in testing : {test_acc * 100:.2f}%')


