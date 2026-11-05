"""Verify that training restores an earlier independent parameter snapshot."""

import numpy as np
import pytest

from circuitcost.models import DenseLayer, NeuralNetworks
from circuitcost.train import mse_loss, train_model


class ObservedNetwork(NeuralNetworks):
    def __init__(self):
        super().__init__(2)
        self.snapshots = []

    def step(self, learning_rate):
        super().step(learning_rate)
        self.snapshots.append(
            [
                (layer.weights.copy(), layer.biases.copy())
                for layer in self.layers
                if isinstance(layer, DenseLayer)
            ]
        )


def test_restores_best_epoch_weights_and_biases_instead_of_last_epoch():
    np.random.seed(2)
    model = ObservedNetwork()
    for layer in model.layers:
        if isinstance(layer, DenseLayer):
            layer.weights[:] = 0.05
            layer.biases[:] = 0.01

    # Deliberately different objectives make validation worsen after an early
    # epoch, even as training loss continues to decrease.
    X_train, X_val = np.ones((4, 2)), np.ones((2, 2))
    y_train, y_val = np.full(4, 2.0), np.full(2, 0.2)
    history = train_model(
        model, X_train, y_train, X_val, y_val, epochs=12, learning_rate=0.01
    )

    best_index = int(np.argmin(history["val_loss"]))
    assert best_index < 11
    assert len(model.snapshots) == 12  # One update per epoch; no early stopping.
    assert history["best_epoch"] == best_index + 1
    assert history["best_val_loss"] == min(history["val_loss"])
    assert history["val_loss"][-1] > history["best_val_loss"]

    dense_layers = [layer for layer in model.layers if isinstance(layer, DenseLayer)]
    for layer, (weights, biases) in zip(
        dense_layers, model.snapshots[best_index], strict=True
    ):
        np.testing.assert_array_equal(layer.weights, weights)
        np.testing.assert_array_equal(layer.biases, biases)
    assert not np.allclose(dense_layers[0].weights, model.snapshots[-1][0][0])
    restored_loss = mse_loss(y_val.reshape(-1, 1), model.forward(X_val))
    assert restored_loss == pytest.approx(history["best_val_loss"])


@pytest.mark.parametrize("epochs", [0, -1])
def test_rejects_empty_training_run(epochs):
    with pytest.raises(ValueError, match="epochs must be at least 1"):
        train_model(None, [[1]], [1], [[1]], [1], epochs, 0.01)
