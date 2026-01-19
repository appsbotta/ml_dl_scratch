import argparse
import numpy as np
import pandas as pd
import os
from activations import get_activation_function
from loss_functions import get_loss_function
from optimizers import get_optimizer

def parse_arguments():
    parser = argparse.ArgumentParser(description="Backpropagation Module")
    parser.add_argument('--lr', help='Learning rate', type=float, default=0.001)
    parser.add_argument('--momentum', help='Momentum factor', type=float, default=0.9)
    parser.add_argument('--num_hidden', help="Number of Hidden layers", type=int, default=2)
    parser.add_argument('--sizes',help='size of each hidden layer',type=str)
    parser.add_argument('--activation', help='Activation function', type=str, default='sigmoid')
    parser.add_argument('--loss', help='Loss function', type=str, default='ce')
    parser.add_argument("--opt", help="Optimizer to use", type=str, default="gd")
    parser.add_argument('--batch_size', help='Batch size for training', type=int, default=10)
    parser.add_argument('--epochs', help='Number of training epochs', type=int, default=500)
    parser.add_argument('--anneal', help='Learning rate annealing factor', type=str, default='False')
    parser.add_argument('--save_dir', help='Directory to save the trained model', type=str, default='save_dir/')
    parser.add_argument('--expt_dir', help='logs directory', type=str, default='logs/')
    parser.add_argument('--train', help='Path to training data', type=str)
    parser.add_argument('--val', help='Path to validation data', type=str)
    parser.add_argument('--test', help='Path to test data', type=str)
    parser.add_argument('--testing', help='Flag for testing mode', type=str, default='False')
    return parser.parse_args()

def load_data(file_path):
    data = pd.read_csv(file_path)
    data = data.values
    X = data[:, 1:-1]/255.0
    y = data[:, -1].astype(int)
    return X, y

def one_hot_encode(y, num_classes=10):
    one_hot = np.zeros((len(y), num_classes))
    one_hot[np.arange(len(y)), y] = 1
    return one_hot

class NeuralNetwork:
    def __init__(self, input_size, num_hidden_layers, hidden_layer_sizes, output_size, activation_name, loss_name, optimizer_name, batch_size):
        self.num_hidden_layers = num_hidden_layers
        self.activation_name = activation_name
        self.loss_function_name = loss_name
        self.optimizer_name = optimizer_name
        self.batch_size = batch_size
        self.layers = []

        self.layers = [input_size] + hidden_layer_sizes + [output_size]
        self.params = {}
        for i in range(1, len(self.layers)):
            self.params['W' + str(i)] = np.random.randn(self.layers[i], self.layers[i-1]) * np.sqrt(1. / self.layers[i-1])
            self.params['b' + str(i)] = np.zeros((self.layers[i],1))
            self.params['vW' + str(i)] = np.zeros_like(self.params['W' + str(i)])
            self.params['vb' + str(i)] = np.zeros_like(self.params['b' + str(i)])

        self.m_adam = {}     # For Adam
        self.v_adam = {}     # For Adam
        self.t_adam = 0      # Adam time step
        for key in self.params:
            self.m_adam[key] = np.zeros_like(self.params[key])
            self.v_adam[key] = np.zeros_like(self.params[key])
    
    def grad(self,x):
        if self.activation_name == 'sigmoid':
            activation = get_activation_function(self.activation_name)
            s = activation(x)
            return s * (1 - s)
        elif self.activation_name == 'tanh':
            activation = get_activation_function(self.activation_name)
            t = activation(x)
            return 1 - t ** 2
        elif self.activation_name == 'relu':
            return np.where(x > 0, 1, 0)
        else:
            raise ValueError(f"Gradient for activation function '{self.activation_name}' is not supported.")
        
    def forward_propagation(self, input_data):
        self.cache = {'h0': input_data}
        activation = get_activation_function(self.activation_name)
        for i in range(1, self.num_hidden_layers+1):
            W = self.params['W' + str(i)]
            b = self.params['b' + str(i)]
            self.cache[f'a{i}'] = np.dot(W, self.cache[f'h{i-1}']) + b
            self.cache[f'h{i}'] = activation(self.cache[f'a{i}'])
        
        self.cache['a_out'] = np.dot(self.params['W' + str(self.num_hidden_layers+1)], self.cache[f'h{self.num_hidden_layers}']) + self.params['b' + str(self.num_hidden_layers+1)]
        self.cache['h_out'] = get_activation_function('softmax')(self.cache['a_out'])
        return self.cache['h_out']
    
    def calculate_loss(self, y_true, y_pred):
        loss_function = get_loss_function(self.loss_function_name)
        return loss_function(y_true, y_pred) 
    
    def backward_propagation(self, y_pred, y_true):
        grads = {}
        m = y_true.shape[1]
        if self.loss_function_name == 'ce':
            delta = y_pred - y_true
        elif self.loss_function_name == 'sq':
            delta = (y_pred - y_true) * y_pred * (1 - y_pred)
        else:
            raise ValueError(f"Backward propagation for loss function '{self.loss_function_name}' is not supported.")
        grads[f'delta'+ str(self.num_hidden_layers+1)] = delta
        grads['dW' + str(self.num_hidden_layers+1)] = (1/m) * np.dot(delta, self.cache[f'h{self.num_hidden_layers}'].T)
        grads['db' + str(self.num_hidden_layers+1)] = (1/m)* np.sum(delta, axis=1, keepdims=True)

        for i in range(self.num_hidden_layers, 0, -1):
            delta = np.dot(self.params['W' + str(i+1)].T, delta) * self.grad(self.cache[f'a{i}'])
            grads['delta' + str(i)] = delta
            grads['dW' + str(i)] = (1/m) * np.dot(delta, self.cache[f'h{i-1}'].T)
            grads['db' + str(i)] = (1/m) * np.sum(delta, axis=1, keepdims=True)

        return grads
    
    def update_parameters(self, grads,learning_rate,momentum):
        beta1, beta2, epsilon = 0.9, 0.999, 1e-8
        
        for i in range(1, self.num_hidden_layers + 2):
            if self.optimizer_name == 'gd':
                self.params['W' + str(i)] -= learning_rate * grads['dW' + str(i)]
                self.params['b' + str(i)] -= learning_rate * grads['db' + str(i)]
            elif self.optimizer_name == 'momentum':
                self.params['vW' + str(i)] = momentum * self.params['vW' + str(i)] + learning_rate * grads['dW' + str(i)]
                self.params['vb' + str(i)] = momentum * self.params['vb' + str(i)] + learning_rate * grads['db' + str(i)]
                self.params['W' + str(i)] -= self.params['vW' + str(i)]
                self.params['b' + str(i)] -= self.params['vb' + str(i)]
            elif self.optimizer_name == 'nag':
                vW_prev = self.params['vW' + str(i)].copy()
                vb_prev = self.params['vb' + str(i)].copy()
                self.params['vW' + str(i)] = momentum * self.params['vW' + str(i)] + learning_rate * grads['dW' + str(i)]
                self.params['vb' + str(i)] = momentum * self.params['vb' + str(i)] + learning_rate * grads['db' + str(i)]
                self.params['W' + str(i)] -= -momentum * vW_prev + (1 + momentum) * self.params['vW' + str(i)]
                self.params['b' + str(i)] -= -momentum * vb_prev + (1 + momentum) * self.params['vb' + str(i)]
            elif self.optimizer_name == 'adam':
                self.t_adam += 1

                self.m_adam['W' + str(i)] = beta1 * self.m_adam['W' + str(i)] + (1 - beta1) * grads['dW' + str(i)]
                self.m_adam['b' + str(i)] = beta1 * self.m_adam['b' + str(i)] + (1 - beta1) * grads['db' + str(i)]
                self.v_adam['W' + str(i)] = beta2 * self.v_adam['W' + str(i)] + (1 - beta2) * (grads['dW' + str(i)] ** 2)
                self.v_adam['b' + str(i)] = beta2 * self.v_adam['b' + str(i)] + (1 - beta2) * (grads['db' + str(i)] ** 2)

                m_hat_W = self.m_adam['W' + str(i)] / (1 - beta1 ** self.t_adam)
                m_hat_b = self.m_adam['b' + str(i)] / (1 - beta1 ** self.t_adam)
                v_hat_W = self.v_adam['W' + str(i)] / (1 - beta2 ** self.t_adam)
                v_hat_b = self.v_adam['b' + str(i)] / (1 - beta2 ** self.t_adam)

                self.params['W' + str(i)] -= learning_rate * m_hat_W / (np.sqrt(v_hat_W) + epsilon)
                self.params['b' + str(i)] -= learning_rate * m_hat_b / (np.sqrt(v_hat_b) + epsilon)
            else:
                raise ValueError(f"Optimizer '{self.optimizer_name}' is not supported.")


            

def compute_accuracy(model, X, y_true):
    y_prob = model.forward_propagation(X.T)
    y_pred = np.argmax(y_prob, axis=0)
    accuracy = np.mean(y_pred == y_true) * 100
    return accuracy


def main():
    args = parse_arguments()
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.expt_dir, exist_ok=True)

    X_train, y_train = load_data(args.train)
    X_val, y_val = load_data(args.val)

    y_train_one_hot = one_hot_encode(y_train, 10)
    y_val_one_hot = one_hot_encode(y_val, 10)

    hidden_sizes = [int(size) for size in args.sizes.split(',')]

    model = NeuralNetwork(input_size=784,
                          num_hidden_layers=args.num_hidden,
                          hidden_layer_sizes=hidden_sizes,
                          output_size=10,
                          activation_name=args.activation,
                          loss_name=args.loss,
                          optimizer_name=args.opt,
                          batch_size=args.batch_size)

    global_step = 0
    learning_rate = args.lr
    best_val_loss = float('inf')
    anneal = (args.anneal.lower() == 'true')
    if args.testing.lower() == 'false':
        print("Starting Training...\n------------------------------------")

        for epoch in range(args.epochs):
            permutation = np.random.permutation(X_train.shape[0])
            X_train_shuffled = X_train[permutation]
            y_train_one_hot_shuffled = y_train_one_hot[permutation]

            for i in range(0, X_train.shape[0], args.batch_size):
                X_batch = X_train_shuffled[i:i + args.batch_size].T
                # print(X_batch.shape)
                y_batch = y_train_one_hot_shuffled[i:i + args.batch_size].T
                # print(y_batch.shape)
                y_pred = model.forward_propagation(X_batch)
                loss = model.calculate_loss(y_batch, y_pred)
                grads = model.backward_propagation(y_pred, y_batch)
                model.update_parameters(grads, learning_rate, args.momentum)

                global_step += 1

            val_pred = model.forward_propagation(X_val.T)
            val_loss = model.calculate_loss(y_val_one_hot.T, val_pred)

            # Compute accuracies
            train_acc = compute_accuracy(model, X_train, y_train)
            val_acc = compute_accuracy(model, X_val, y_val)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                # Save the model parameters
                np.savez(os.path.join(args.save_dir, 'best_model.npz'), **model.params)

            if anneal and epoch > 0 and epoch % 10 == 0:
                learning_rate *= 0.5

            print(f"Epoch {epoch+1}/{args.epochs}, Training Loss: {loss:.4f}, Validation Loss: {val_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%")
        
        print("\n------------------------------------")
        print("Training Finished. Evaluating on Test Set...")

    best_model_path = os.path.join(args.save_dir, 'best_model.npz')
    if os.path.exists(best_model_path):
        print("Loading best model weights...")
        best_params = np.load(best_model_path)
        for key in best_params.files:
            model.params[key] = best_params[key]

    test_data = pd.read_csv(args.test).values
    X_test = test_data[:, 1:]/255.0
    y_prob = model.forward_propagation(X_test.T)
    y_pred = np.argmax(y_prob, axis=0)
    print("Test Predictions:")
    # Save predictions to CSV with id and label columns
    predictions_df = pd.DataFrame({'id': test_data[:, 0].astype(int), 'label': y_pred})
    predictions_df.to_csv(os.path.join(args.expt_dir, 'predictions.csv'), index=False)
    print(f"Predictions saved to {os.path.join(args.expt_dir, 'predictions.csv')}")
    print("------------------------------------")

if __name__ == "__main__":
    main()

# python train.py --lr 0.01 --momentum 0.5 --num_hidden 3 --sizes 100,100,100 --activation sigmoid --loss sq --opt gd --batch_size 20 --anneal true --train train.csv --val val.csv --test test.csv --testing True