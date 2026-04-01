import numpy as np

class NeuralNetMLP:
    
    def __init__(self, num_features, num_hidden, num_classes, random_seed=123):
        
        super().__init__()
        
        self.num_classes = num_classes
        
        # hidden layer
        rng = np.random.RandomState(random_seed)
        
        self.weight_h = rng.normal(loc=0.0, scale=0.1, size=(num_classes, num_hidden))
        self.bias_h = np.zeros(num_hidden)
                    
        # out
        self.weight_out = rng.normal(loc=0.0, scale=0.1, size=(num_classes, num_hidden))
        self.bias_out = np.zeros(num_classes)
        
        
    def sigmoid(z):
        return 1.0 / (1.0 + np.exp(-z))
    
    
    def int_to_onehot(y, num_labels):
        ary = np.zeros((y.shape[0], num_labels))
        for i, val in enumerate(y):
            ary[i, val] = 1
        
        return ary


    def forward(self, x):
        # hidden layer
        
        z_h = np.dot(x, self.weight_h.T) + self.bias_h        
        a_h = self.sigmoid(z_h)
        
        # output layer
        z_out = np.dot(a_h, self.weight_h.T) + self.bias_h
        a_out = self.sigmoid(z_out)
        
        return a_h, a_out
    
    def backward(self, x, a_h, a_out, y):
        
        y_onehot = self.int_to_onehot(y, self.num_classes)
        
        # dOutWeights = dLoss/dOutAct * dOutAct/dOutNet * dOutNet/dOutWeight
        # where deltaOut = dLoss/dOutAct * dOutAct/dOutNet
        
        # размер входа/выхода [n_examples, n_classes]
        d_loss__d_a_out = 2.0 * (a_out - y_onehot) / y.shape[0]
        
        # размер входа/выхода [n_examples, n_classes]
        d_a_out__d_z_out = a_out *  (1.0 - a_out) # сигмоидная производная
        
        # размер выхода: [n_examples, n_classes]
        delta_out = d_loss__d_a_out * d_a_out__d_z_out
        
        # градиент для выходных весов
        
        # [n_examples, n_hidden]
        d_z_out__dw_out = a_h
        
        # размер входа: [n_classes, n_examples] dot [n_examples, n_hidden]
        # размер выхода: [n_classes, n_hidden]
        d_loss_dw_out = np.dot(delta_out.T, d_z_out__dw_out)
        d_loss_db_out = np.sum(delta_out, axis=0)
        
        
        # dHiddenWeights = DeltaOut * dOutNet/dHiddenAct * dHiddenAct/dHiddenNet * dHiddenNet/dWeight
        
        # [n_classes, n_hidden]
        d_z_out__a_h = self.weight_out
        
        # [n_examples, n_hidden]
        d_loss__a_h = np.dot(delta_out, d_z_out__a_h)
        
        # [n_examples, n_hidden]
        d_a_h__d_z_h = a_h * (1.0 - a_h) # производная сигмоидная
        
        # [n_examples, n_features]
        d_z_h__d_w_h = x
        
        # размер выхода [n_hidden, n_features]
        d_loss__d_w_h = np.dot((d_loss__a_h * d_a_h__d_z_h).T, d_z_h__d_w_h)
        d_loss__d_b_h = np.sum((d_loss__a_h * d_a_h__d_z_h), axis=0)
        
        
        return (d_loss_dw_out, d_loss_db_out, d_loss__d_w_h, d_loss__d_b_h)
    