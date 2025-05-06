import os
import sys
import math
import numpy as np
import pandas as pd
import argparse
import logging
from collections import deque

from sklearn.model_selection import train_test_split

from ml_utils import MinMaxScaler, stratified_kfold_indices, calculate_metrics


VERBOSE = False

if os.environ.get('DEBUG_MODE', False):
    VERBOSE = True

def dprint(msg, verbose=VERBOSE):
    '''Simple debug print (logging)'''
    if verbose:
        print(msg)


class NN:
    '''Neural Network'''
    def __init__(self, ninput=8, noutput=1, hidden_neurons=[4, 2], Weight=None, Bias=None, rlambda=0.01, activation='sigmoid', output_activation='sigmoid', loss_func = 'binary_classify', random_state=None):
        self.ninput = ninput
        self.noutput = noutput
        self.hidden_neurons = list(hidden_neurons) if hidden_neurons is not None else None
        self.g = self._relu if activation == 'relu' else self._sigmoid
        self.og = self._sigmoid if output_activation == 'sigmoid' else self._relu
        self.rlambda = rlambda
        self.random_state = random_state
        
        self.Weight = Weight
        self.Bias = Bias
        
        self.loss_func = None
        self.loss_prime = None

        if self.random_state is not None:
            np.random.seed(self.random_state)

        if self.Weight is None or self.Bias is None:
            self._init_weights()
        else:
            self._layers = sorted(self.Weight.keys())

        if loss_func == 'binary_classify':
            self.loss_func = self._loss_bin
            self.loss_prime = self._loss_bin_prime
        elif loss_func == 'mse':   
            self.loss_func = self._loss_mse
            self.loss_prime = self._loss_mse_prime
        else:  # 'softmax'
            self.loss_func = self._softmax

    def _init_weights(self):
        nodes = [x for x in self.hidden_neurons]   # copy
        nodes.insert(0, self.ninput)
        nodes.append(self.noutput)
        
        self.Weight = {l:np.random.randn(nodes[l-1], nodes[l]) for l in range(1, len(nodes))}
        # self.Bias = {l:np.random.randn(1, nodes[l]) for l in range(1, len(nodes))}
        self.Bias = {l:np.random.randn(1, nodes[l]) for l in range(1, len(nodes))}
        dprint(f'Weight: \n{self.Weight}')
        dprint(f'Bias: \n{self.Bias}')
                
        layers = sorted(self.Weight.keys())
        self._layers = layers

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def _sigmoid_prime(self, x):
        s = self._sigmoid(x)
        result = s * (1 - s)
        # result = np.array([x[0] for x in result])
        dprint(f'_sigmoid_prime: \n{result}')
        return result

    def _relu(self, x):
        return np.maximum(0, x)

    def _relu_prime(self, x):
        return (x > 0).astype(float)

    def _loss_bin(self, y_true, y_pred):
        loss_arr = -y_true * np.log(y_pred) - (1-y_true) * np.log(1-y_pred) 
        return  np.mean(np.sum(loss_arr, axis=1))  # sum over classes, then average over samples

    def _loss_bin_prime(self, y_true, y_pred):
        return -(y_true /y_pred - (1-y_true) / (1-y_pred)) 

    def _loss_mse(self, y_true, y_pred):
        result = np.mean((y_pred - y_true) ** 2)
        dprint(f'\n\n***** mse: {result} *****\n\n')
        return result

    def _loss_mse_prime(self, y_true, y_pred):
        # y_pred = np.array([x[0] for x in y_pred])
        return (y_pred - y_true) / y_true.size

    def _softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)

    def _grad(self, ix, func, epsilon = 0.001):
        return (func(x + epsilon) - func(x - epsilon)) / (2 * epsilon)

    def _weight_square_sum(self):
        wsum = 0   # Weight squared sum
        for l in self._layers:
            A = self.Weight[l]
            A_square = A * A    # element-wise square
            wsum += A_square.sum()
        return wsum


    def _forward(self, X):

        a = {0: X}
        z = {}
        
        # Hidden layers
        for l in self._layers[:-1]:  # layers: 1, 2, ..., N
            dprint(f'a[{l-1}].shape: {a[l-1].shape}')
            z[l] = a[l-1] @ self.Weight[l] + self.Bias[l]
            a[l] = self.g(z[l])

        # Output layer
        l = self._layers[-1]
        z[l] = a[l-1] @ self.Weight[l] + self.Bias[l]
        a[l] = self.og(z[l])

        dprint(f'z: \n{z}')
        dprint(f'a: \n{a}')

        # print(f'forward output: \n{a[l]}', end='\r')

        return z, a

    def _backpropagation(self, X, y, z, a, lr=0.01, batch_size=1, rlambda=0.01):
        m = X.shape[0]

        dWeight = {}
        dBias = {}
        delta = {}    # dLoss over z^l

        layers = sorted(self.Weight.keys())
        layers.reverse()

        # Output layer
        l = layers[0]
        delta[l] = self.loss_prime(y, a[l]) * self._sigmoid_prime(z[l])
        
        dprint(f'Output l={l} delta[{l}]: \n{delta[l]}')

        # Hidden layers
        for l in layers[1:]:

            dprint(f'SHAPES: {self._sigmoid_prime(z[l]).shape} {self.Weight[l+1].shape} {delta[l+1].shape}')

            delta[l] = (delta[l+1] @ self.Weight[l+1].T) * self._sigmoid_prime(z[l])


        dprint(f'delta: \n{delta}')

        for l in layers:
            dWeight[l] = a[l-1].T @ delta[l] 
            # dBias[l] = delta[l]
            dBias[l] = np.sum(delta[l], axis=0, keepdims=True)
        
        dprint(f'dWeight: \n{dWeight}')
        dprint(f'dBias: \n{dBias}')
       
        grads = {}
        for l in self._layers:
            grads[l] = dWeight[l] + rlambda  * self.Weight[l]

        for l in self._layers:
            # With regularization
            # self.Weight[l] -= lr * dWeight[l] + rlambda / batch_size * self.Weight[l]
            self.Weight[l] -= lr * grads[l]
            self.Bias[l] -= lr * dBias[l]

        return delta, dWeight, dBias, grads
    
    def fit(self, X_train, y_train, lr=0.01, rlambda=0.01, batch_size=16, epochs=1000, loss_delta_threshold=0.001):
        if not rlambda:
            rlambda = self.rlambda

        m = math.ceil(len(X_train) / batch_size) 
        m_total = len(X_train)

        # Monitor the last 20 losses for possible early stop
        q = deque(maxlen=20)
        q.append(np.inf)

        epoch_losses = {}

        for epoch in range(epochs):

            losses = []
            for i in range(m):
                X_batch = X_train[i*batch_size: (i+1)*batch_size]
                y_batch = y_train[i*batch_size: (i+1)*batch_size]

                z, a = self._forward(X_batch)
                self._backpropagation(X_batch, y_batch, z, a, lr=lr, batch_size=batch_size, rlambda=rlambda)

                # Regularization
                wsum = self._weight_square_sum()

                loss = self.loss_func(y_batch, a[self._layers[-1]]) + rlambda / (2*batch_size) * wsum
                
                losses.append(loss)

                if epoch % 50 == 0:
                    epoch_losses[f'Epoch {epoch}'] = losses

            mean_loss = np.mean(losses)
            end = '\r' if epoch % 100 != 0 else '\n'
            print(f'Epoch {epoch}/{100*(int(epoch/100)+1)}: loss = {mean_loss}', end=end)
            sys.stdout.flush() 
            
            q.append(mean_loss)
            if max(q) - min(q) < loss_delta_threshold:
                break
            
        # last one
        epoch_losses[f'Epoch {epoch}'] = losses
        
        index = [i * batch_size for i in range(m)]
        df_losses = pd.DataFrame(epoch_losses, index=index)
        
        print(f'\n\nepoch_losses: \n{df_losses}')

        return df_losses

    def predict(self, X):
        z, a = self._forward(X)
        output = a[self._layers[-1]]     
        output_norm =  (output > 0.5).astype(int) 
        
        dprint(f'output: \n{output}')
        dprint(f'output_norm: \n{output_norm}')

        return output_norm, output

    def evaluate(self, y_pred, y_test):
        return calculate_metrics(y_pred, y_test)

    def __repr__(self):
        return f'<NN: input {self.ninput}, hidden layers {self.hidden_neurons}, output {self.noutput}, random_state {self.random_state}>'


def cross_validation(args):
    '''Stratified K-fold validation of NN'''

    # args = get_args()
    
    if args.random_state is not None:
        np.random.seed(args.random_state)

    df = pd.read_csv(args.path)
    print(f'\ndf: \n{df}')
    X, y = df.iloc[:, :-1].values, df.iloc[:, -1].values
    
    # IMPORTANT: [1,0,1]  to [[1], [0], [1], ...]
    y = np.array(y).reshape(-1, 1)    # auto rows, exactly one column

    df_train_losses = {}
    ensemble = []
    performances = []
    losses = []

    nfold = 5
    for fold_i, (train_idx, test_idx) in enumerate(stratified_kfold_indices(y, nfold)):

        print(f'\n----- Stratified K-Fold {fold_i + 1} -----')

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        dprint(f'\nX_train norm: \n{X_train}')
        dprint(f'X_test norm: \n{X_test}')


        nn = NN(ninput=len(df.columns)-1, noutput=args.num_output, hidden_neurons=args.hidden_neurons, activation=args.activation, rlambda=args.rlambda, random_state=args.random_state)
        print(f'\n\n{nn}\n\n')

        df_train_loss = nn.fit(X_train, y_train, lr=args.learning_rate, rlambda=args.rlambda, batch_size=args.batch_size, loss_delta_threshold=args.loss_delta_threshold, epochs=args.epochs)
        y_pred, y_pred_raw = nn.predict(X_test)
    
        loss = nn.loss_func(y_test, y_pred_raw)
        losses.append(loss)

        dprint(f'y_pred: \n{y_pred}')
        dprint(f'y_test: \n{y_test}')
        
        print(f'\n\nTest fold Mean Loss: {loss}')

        # performance = calculate_metrics(y_pred, y_test)
        performance = nn.evaluate(y_pred, y_test)
        print(f'\nTest Fold Perfornance (Accuracy Precision Recall F1): \n{performance}')
        
        performances.append(performance)

        ensemble.append(nn)

        df_train_losses[fold_i + 1] = df_train_loss

    print(f'\nStratified K-Fold losses: \n{losses} \nmean loss: {np.mean(losses)}')
    
    performances = np.array(performances)
    df = pd.DataFrame(performances, columns=['Accuracy', 'Precision', 'Recall', 'F1'])
    print(f'\nStratified K-Fold Performances: \n{df}')
    print(f'\nStratified K-Fold Mean Performances: \n{df.mean()}')

    report = f"| {nn.ninput}, {nn.hidden_neurons}, {nn.noutput} | {args.batch_size} | {args.learning_rate}| {args.rlambda} | {np.mean(losses):.5f} | {df['Accuracy'].mean():.5f} | {df['Precision'].mean():.5f} | {df['Recall'].mean():.5f} | {df['F1'].mean():.5f} |"

    print(f'Report table line: \n{report}')

    plot_losses(df_train_losses, args)


def plot_losses(df_losses, args):
    for fold_i, df_loss in df_losses.items():
        title = args.path.replace('.csv', '')
        fig = df_loss.plot(xlabel='samples', ylabel='losses', grid=True, alpha=0.7, title=title).get_figure()
        fig.savefig(f'img/nn_{title}_{fold_i}.png')


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', type=str,
                        default='wdbc.csv', help='csv file path, like wdbc.csv')
    parser.add_argument('-input', '--num_input', type=int,
                        default=5, help='number of input, default 5')
    parser.add_argument('-output', '--num_output', type=int,
                        default=1, help='number of output, default 1')
    parser.add_argument('-neuron', '--hidden_neurons', type=lambda s: [int(x) for x in s.split(' ')], default=[4], help='Hidden layer neurons, default [10]')
    parser.add_argument('-activation', '--activation', type=str, 
                        default='sigmoid', help='Non-output activation function, default sigmoid')
    parser.add_argument('-batch', '--batch_size', type=int,
                        default=16, help='batch size, default 16')
    parser.add_argument('-lr', '--learning_rate', type=float,
                        default=0.01, help='Learning rate, default 0.01')
    parser.add_argument('-rlambda', '--rlambda', type=float,
                        default=0.01, help='Regularization lambda, default 0.01')
    parser.add_argument('-loss_delta', '--loss_delta_threshold', type=float,
                        default=0.001, help='Early stop loss limit, default 0.001')
    parser.add_argument('-epoch', '--epochs', type=int,
                        default=1000, help='number of epochs, default 1000')
    parser.add_argument('-random_state', '--random_state', type=lambda s: int(s) if s.lstrip('-').strip().isdigit() else None, default=None, help='random seed like 42, -5, default None')
    
    args, _ = parser.parse_known_args()

    print(f'args: {args}')

    return args


if __name__ == "__main__":
    args = get_args()
    cross_validation(args)
    
