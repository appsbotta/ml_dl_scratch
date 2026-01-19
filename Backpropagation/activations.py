import numpy as np


def sigmoid(x):
    return 1 /(1 + np.exp(-x))

def tannh(x):
    return np.tanh(x)

def relu(x):
    return np.maximum(0, x)

def softmax(x):
    x = np.array(x)
    # Subtract max for numerical stability (prevents overflow)
    x_shifted = x - np.max(x, axis=0, keepdims=True)
    exp_x = np.exp(x_shifted)
    return exp_x / np.sum(exp_x, axis=0, keepdims=True)

def linear(x):
    return x

def get_activation_function(name: str):
    if name == 'sigmoid':
        return sigmoid
    elif name == 'tanh':
        return tannh
    elif name == 'relu':
        return relu
    elif name == 'softmax':
        return softmax
    elif name == 'linear':
        return linear
    else:
        raise ValueError(f"Activation function '{name}' is not supported.")