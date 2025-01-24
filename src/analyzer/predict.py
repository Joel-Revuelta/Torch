import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def predict(inputs, weights, biases):
    layer_input = np.array(inputs)
    for w, b in zip(weights, biases):
        layer_input = sigmoid(np.dot(layer_input, np.array(w)) + np.array(b))
    return layer_input
