"""Full-batch gradient descent with best-validation checkpoint restoration."""

import numpy as np


def mse_loss(y_true, y_pred):
    """Mean squared error; targets and predictions must have matching shapes."""
    error = y_pred - y_true
    return np.mean(error**2)


def mse_gradient(y_true, y_pred):
    """Gradient of the mean loss; dense layers must not average it again."""
    error = y_pred - y_true
    return 2 * error / error.size


def train_model(
    model,
    X_train,
    y_train,
    X_val,
    y_val,
    epochs,
    learning_rate,
):
    """Train for the epoch budget and restore copies of the best parameters.

    Both recorded losses use the post-update parameters. ``best_epoch`` is
    one-based; the final history entries can differ from the restored model.
    """
    if epochs < 1:
        raise ValueError("epochs must be at least 1")

    X_train = np.asarray(X_train, dtype=float)
    X_val = np.asarray(X_val, dtype=float)

    y_train = np.asarray(y_train).reshape(-1, 1)
    y_val = np.asarray(y_val).reshape(-1, 1)

    history = {
        "train_loss": [],
        "val_loss": [],
    }

    # Find layers that contain trainable parameters
    dense_layers = [
        layer
        for layer in model.layers
        if hasattr(layer, "weights") and hasattr(layer, "biases")
    ]

    best_val_loss = np.inf
    best_epoch = None
    best_parameters = None

    for epoch in range(epochs):
        # Forward pass
        predictions = model.forward(X_train)

        # Backward pass
        gradient = mse_gradient(y_train, predictions)
        model.backward(gradient)

        # Update parameters
        model.step(learning_rate)

        # Measure both losses using the updated parameters
        train_predictions = model.forward(X_train)
        train_loss = mse_loss(y_train, train_predictions)

        val_predictions = model.forward(X_val)
        val_loss = mse_loss(y_val, val_predictions)

        if not np.isfinite(train_loss) or not np.isfinite(val_loss):
            raise ValueError(
                "Loss became nonfinite. Reduce the learning rate or scale the targets."
            )

        history["train_loss"].append(float(train_loss))
        history["val_loss"].append(float(val_loss))

        # Save a checkpoint whenever validation loss improves
        if val_loss < best_val_loss:
            best_val_loss = float(val_loss)
            best_epoch = epoch + 1

            best_parameters = [
                (layer.weights.copy(), layer.biases.copy()) for layer in dense_layers
            ]

    # Restore the best checkpoint
    for layer, (weights, biases) in zip(dense_layers, best_parameters):
        layer.weights = weights.copy()
        layer.biases = biases.copy()

    history["best_epoch"] = best_epoch
    history["best_val_loss"] = best_val_loss

    return history
