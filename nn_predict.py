import numpy as np
import json

# === Activation functions ===
def relu(x):
    return np.maximum(0, x)

def softmax(x):
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return np.array([], dtype=np.float64)
    input_ndim = x.ndim
    if input_ndim == 1:
        x = x.reshape(1, -1)
    elif input_ndim == 0:
        return np.array([1.0], dtype=np.float64)
    x = np.clip(x, -1e308, 1e308)
    x_max = np.max(x, axis=-1, keepdims=True)
    e_x = np.exp(x - x_max)
    sum_e_x = np.sum(e_x, axis=-1, keepdims=True)
    sum_e_x = np.where(sum_e_x == 0, 1.0, sum_e_x)
    result = e_x / sum_e_x
    result = result / (np.sum(result, axis=-1, keepdims=True) + 1e-10)
    if input_ndim == 1:
        result = result.flatten()
    return result

# === Flatten ===
def flatten(x):
    return x.reshape(x.shape[0], -1)

# === Dense layer ===
def dense(x, W, b):
    return x @ W + b

# Infer TensorFlow h5 model using numpy
def nn_forward_h5(model_arch, weights, data):
    x = data
    for layer in model_arch:
        lname = layer['name']
        ltype = layer['type']
        cfg = layer['config']
        wnames = layer['weights']
        if ltype == "Flatten":
            x = flatten(x)
        elif ltype == "Dense":
            W = weights[wnames[0]]
            b = weights[wnames[1]]
            x = dense(x, W, b)
            if cfg.get("activation") == "relu":
                x = relu(x)
            elif cfg.get("activation") == "softmax":
                x = softmax(x)
    return x

def nn_inference(model_arch, weights, data):
    return nn_forward_h5(model_arch, weights, data)