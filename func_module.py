import numpy as np

def minibatch_generator(X, y, minibatch_size):
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    for start_idx in range(0, indices.shape[0] - minibatch_size + 1, minibatch_size):
        batch_idx = indices[start_idx:start_idx + minibatch_size]
        
        yield X[batch_idx], y[batch_idx]
        
def sigmoid(z):
        return 1.0 / (1.0 + np.exp(-z))

def int_to_onehot(y, num_labels):
        ary = np.zeros((y.shape[0], num_labels))
        for i, val in enumerate(y):
            ary[i, val] = 1
        
        return ary
        
def compute_mse_and_acc(nnet, X, y, num_labels=10, minibatch_size=100):
    
    mse, correct_pred, num_examples = 0.0, 0, 0
    minibatch_gen = minibatch_generator(X, y, minibatch_size)
    
    for i, (features, targets) in enumerate(minibatch_gen):
        _, probas = nnet.forward(features)
        predicted_labels = np.argmax(probas, axis=1)
        onehot_targets = int_to_onehot(targets, num_labels=num_labels)
        loss = np.mean((onehot_targets - probas) ** 2)
        correct_pred += (predicted_labels == targets).sum()
        num_examples += targets.shape[0]
        mse+=loss
        
    mse /=i
    acc = correct_pred / num_examples
    
    return mse, acc

def train(model, X_train, y_train, X_valid, y_valid, num_epochs, learning_rate=0.1, minibatch_size=None):
    epoch_loss = []
    epoch_train_acc = []
    epoch_valid_acc = []
    
    for e in range(num_epochs):
        minibatch_gen = minibatch_generator(X_train, y_train, minibatch_size)
        
        for X_train_mini, y_train_mini in minibatch_gen:
            # вычисление выходов
            a_h, a_out = model.forward(X_train_mini)
            
            # вычисление градиентов
            d_loss__d_w_out, d_loss__d_b_out, d_loss__d_w_h, d_loss__d_b_h = model.backward(X_train_mini, a_h, a_out, y_train_mini)
            
            # обновление весов
            model.weight_h -= learning_rate * d_loss__d_w_h
            model.bias_h -= learning_rate * d_loss__d_b_h
            model.weight_out -= learning_rate * d_loss__d_w_out
            model.bias_out -= learning_rate * d_loss__d_b_out
            
        # журнал эпох
        train_mse, train_acc = compute_mse_and_acc(model, X_train, y_train)
        
        valid_mse, valid_acc = compute_mse_and_acc(model, X_valid, y_valid)
        
        train_acc, valid_acc = train_acc * 100, valid_acc * 100
        epoch_train_acc.append(train_acc)
        epoch_valid_acc.append(valid_acc)
        epoch_loss.append(train_mse)

        # можно сказать, что модель вероятно склонна 
        # к переобучению т.к. valid acc < train acc при росте количества эпох (25+)        
        print(f'Эпоха: {e+1:03d} / {num_epochs:03d}'
              f'Train MSE: {train_mse:.2f} '
              f'Train Acc: {train_acc:.2f} '
              f'Valid Acc: {valid_acc:.2f} ')
        
    
    return epoch_loss, epoch_train_acc, epoch_valid_acc
