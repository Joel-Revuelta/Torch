import numpy as np

# Sigmoid and ReLU (Rectified Linear Unit) activation functions

class NeuralNetwork:
    def __init__(self, input_size, hidden_layers, output_size, activation="sigmoid"):
        self.activation_func = self._get_activation_func(activation)
        self.activation_derivative = self._get_activation_derivative(activation)

        self.layers = [input_size] + hidden_layers + [output_size]
        self.weights = [np.random.randn(self.layers[i], self.layers[i + 1]) for i in range(len(self.layers) - 1)]
        self.biases = [np.zeros((1, size)) for size in self.layers[1:]]

    def _get_activation_func(self, name):
        if name == "sigmoid":
            return lambda x: 1 / (1 + np.exp(-x))
        elif name == "relu":
            return lambda x: np.maximum(0, x)
        else:
            raise ValueError(f"Unsupported activation function: {name}")

    def _get_activation_derivative(self, name):
        if name == "sigmoid":
            return lambda x: x * (1 - x)
        elif name == "relu":
            return lambda x: (x > 0).astype(float)
        else:
            raise ValueError(f"Unsupported activation function: {name}")

    def forward(self, x):
        self.layer_outputs = []
        for w, b in zip(self.weights, self.biases):
            x = self.activation_func(np.dot(x, w) + b)
            self.layer_outputs.append(x)
        return x
