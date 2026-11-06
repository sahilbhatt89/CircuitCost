"""NumPy MLP with 32- and 16-neuron ReLU hidden layers and a linear output."""

import numpy as np


class DenseLayer:
    def __init__(self, inputs, neurons):
        self.weights = np.random.randn(inputs, neurons)
        self.biases = np.random.randn(1, neurons)

    def forward(self, inputs):
        self.inputs = inputs
        self.outputs = inputs @ self.weights + self.biases
        return self.outputs

    def backward(self, doutput):
        self.dweights = self.inputs.T @ doutput
        self.dbiases = np.sum(doutput, axis=0, keepdims=True)
        self.dinputs = doutput @ self.weights.T
        return self.dinputs

    def step(self, learning_rate):
        self.weights -= learning_rate * self.dweights
        self.biases -= learning_rate * self.dbiases


class ReLU:
    def forward(self, inputs):
        self.inputs = inputs
        return np.maximum(0, inputs)

    def backward(self, doutput):
        return doutput * (self.inputs > 0)


class NeuralNetworks:
    """Preserve the initialization and architecture of the recorded experiment."""

    def __init__(self, n_inputs):
        self.layers = [
            DenseLayer(n_inputs, 32),
            ReLU(),
            DenseLayer(32, 16),
            ReLU(),
            DenseLayer(16, 1),
        ]

    def forward(self, inputs):
        output = inputs

        for layer in self.layers:
            output = layer.forward(output)

        return output

    def backward(self, gradient):
        for layer in reversed(self.layers):
            gradient = layer.backward(gradient)

        return gradient

    def step(self, learning_rate):
        for layer in self.layers:
            if isinstance(layer, DenseLayer):
                layer.step(learning_rate)
