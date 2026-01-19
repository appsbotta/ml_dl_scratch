import numpy as np

def mse(y_true, y_pred) -> float:
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.mean((y_true - y_pred) ** 2)

def absolute_error(y_true, y_pred) -> float:
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.mean(np.abs(y_true - y_pred))

def cross_entropy(y_true, y_pred) -> float:
    y_true = np.array(y_true, dtype=np.float32)
    y_pred = np.array(y_pred, dtype=np.float32)
    epsilon = 1e-12
    y_pred = np.clip(y_pred, epsilon, 1. - epsilon)
    return -np.mean(y_true * np.log(y_pred))

def get_loss_function(name: str):
    if name == 'sq':
        return mse
    elif name == 'ae':
        return absolute_error
    elif name == 'ce':
        return cross_entropy
    else:
        raise ValueError(f"Loss function '{name}' is not supported.")
