"""Check shapes and derivatives against independent numerical differences."""

import numpy as np
import pytest

from circuitcost.models import DenseLayer, NeuralNetworks, ReLU
from circuitcost.train import mse_gradient, mse_loss


def finite_difference(parameter, objective, epsilon=1e-6):
    gradient = np.zeros_like(parameter)
    for index in np.ndindex(parameter.shape):
        original = parameter[index]
        try:
            parameter[index] = original + epsilon
            upper = objective()
            parameter[index] = original - epsilon
            lower = objective()
        finally:
            parameter[index] = original
        gradient[index] = (upper - lower) / (2 * epsilon)
    return gradient


@pytest.mark.parametrize("batch_size,n_inputs", [(1, 11), (7, 11), (4, 3)])
def test_forward_has_one_prediction_per_design(batch_size, n_inputs):
    np.random.seed(42)
    model = NeuralNetworks(n_inputs)
    predictions = model.forward(np.ones((batch_size, n_inputs)))
    assert predictions.shape == (batch_size, 1)
    assert np.isfinite(predictions).all()


def test_dense_gradients_match_finite_differences():
    np.random.seed(42)
    rng = np.random.default_rng(7)
    layer = DenseLayer(3, 2)
    inputs = rng.normal(size=(4, 3))
    targets = rng.normal(size=(4, 2))
    predictions = layer.forward(inputs)
    layer.backward(mse_gradient(targets, predictions))
    gradients = [layer.dweights.copy(), layer.dbiases.copy(), layer.dinputs.copy()]

    def objective():
        return mse_loss(targets, layer.forward(inputs))

    for parameter, analytical in zip(
        [layer.weights, layer.biases, inputs], gradients, strict=True
    ):
        numerical = finite_difference(parameter, objective)
        np.testing.assert_allclose(analytical, numerical, rtol=1e-5, atol=1e-6)


def test_relu_gradient_passes_only_positive_inputs():
    activation = ReLU()
    inputs = np.array([[-2.0, -0.5, 0.25, 3.0]])
    activation.forward(inputs)
    upstream = np.array([[2.0, 3.0, 4.0, 5.0]])
    np.testing.assert_array_equal(activation.backward(upstream), [[0.0, 0.0, 4.0, 5.0]])


def test_entire_network_backpropagation_matches_finite_differences():
    np.random.seed(3)
    rng = np.random.default_rng(31)
    model = NeuralNetworks(3)
    inputs = rng.normal(size=(4, 3))
    targets = rng.normal(size=(4, 1))
    predictions = model.forward(inputs)
    # Centered differences must not cross ReLU's nondifferentiable zero.
    for layer in model.layers:
        if isinstance(layer, ReLU):
            assert np.min(np.abs(layer.inputs)) > 1e-4
    model.backward(mse_gradient(targets, predictions))
    parameters = []
    for layer in model.layers:
        if isinstance(layer, DenseLayer):
            parameters.extend(
                [
                    (layer.weights, layer.dweights.copy()),
                    (layer.biases, layer.dbiases.copy()),
                ]
            )

    def objective():
        return mse_loss(targets, model.forward(inputs))

    for parameter, analytical in parameters:
        numerical = finite_difference(parameter, objective)
        np.testing.assert_allclose(analytical, numerical, rtol=1e-4, atol=1e-6)
