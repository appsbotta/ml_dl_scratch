import numpy as np
from activations import softmax

def get_optimizer(name: str):
    pass


def grad_sigmoid(x):
    s = 1 /(1 + np.exp(-x))
    return s * (1 - s)

def grad_tannh(x):
    t = np.tanh(x)
    return 1 - t ** 2

def grad_relu(x):
    return np.where(x > 0, 1, 0)

def grad_softmax(x):
    s = softmax(x)
    return s * (1 - s)

def gradient_descent(params, learning_rate, loss_function):
    epochs = 1000
