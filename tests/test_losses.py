import numpy as np
import pytest

from nnlab.losses import (
    BinaryCrossEntropy,
    CategoricalCrossEntropy,
    HuberLoss,
    MeanAbsoluteError,
    MeanSquaredError,
    available_losses,
    get_loss,
)


def numerical_gradient(loss, y_pred, y_true, eps=1e-6):
    estimate = np.zeros_like(y_pred)
    for index in np.ndindex(y_pred.shape):
        up = y_pred.copy()
        down = y_pred.copy()
        up[index] += eps
        down[index] -= eps
        estimate[index] = (loss.forward(up, y_true) - loss.forward(down, y_true)) / (2 * eps)
    return estimate


def test_mse_value_and_gradient(rng):
    loss = MeanSquaredError()
    y_pred = rng.normal(size=(6, 2))
    y_true = rng.normal(size=(6, 2))
    assert loss.forward(y_pred, y_true) == pytest.approx(np.mean((y_pred - y_true) ** 2))
    np.testing.assert_allclose(loss.backward(y_pred, y_true), numerical_gradient(loss, y_pred, y_true), atol=1e-7)


def test_mae_value_and_gradient(rng):
    loss = MeanAbsoluteError()
    y_pred = rng.normal(size=(5, 3))
    y_true = rng.normal(size=(5, 3))
    assert loss.forward(y_pred, y_true) == pytest.approx(np.mean(np.abs(y_pred - y_true)))
    np.testing.assert_allclose(loss.backward(y_pred, y_true), numerical_gradient(loss, y_pred, y_true), atol=1e-5)


def test_huber_matches_quadratic_and_linear_regions():
    loss = HuberLoss(delta=1.0)
    small = loss.forward(np.array([[0.5]]), np.array([[0.0]]))
    large = loss.forward(np.array([[3.0]]), np.array([[0.0]]))
    assert small == pytest.approx(0.125)
    assert large == pytest.approx(2.5)


def test_huber_gradient_matches_numerical(rng):
    loss = HuberLoss(delta=0.7)
    y_pred = rng.normal(size=(8, 1)) * 2.0
    y_true = rng.normal(size=(8, 1))
    np.testing.assert_allclose(loss.backward(y_pred, y_true), numerical_gradient(loss, y_pred, y_true), atol=1e-5)


def test_binary_crossentropy_gradient_matches_numerical(rng):
    loss = BinaryCrossEntropy()
    y_pred = rng.uniform(0.05, 0.95, size=(10, 1))
    y_true = rng.integers(0, 2, size=(10, 1)).astype(float)
    assert loss.forward(y_pred, y_true) > 0.0
    np.testing.assert_allclose(loss.backward(y_pred, y_true), numerical_gradient(loss, y_pred, y_true), atol=1e-5)


def test_categorical_crossentropy_value_and_gradient(rng):
    loss = CategoricalCrossEntropy()
    y_pred = rng.uniform(0.05, 1.0, size=(4, 3))
    y_pred = y_pred / y_pred.sum(axis=1, keepdims=True)
    labels = rng.integers(0, 3, size=4)
    y_true = np.eye(3)[labels]
    expected = -np.mean(np.log(y_pred[np.arange(4), labels]))
    assert loss.forward(y_pred, y_true) == pytest.approx(expected)
    np.testing.assert_allclose(loss.backward(y_pred, y_true), numerical_gradient(loss, y_pred, y_true), atol=1e-5)


def test_perfect_predictions_have_near_zero_crossentropy():
    y_true = np.eye(3)
    assert CategoricalCrossEntropy().forward(y_true, y_true) < 1e-9


def test_registry_and_aliases():
    names = available_losses()
    assert "mse" in names and "bce" in names and "cce" in names
    assert isinstance(get_loss("cce"), CategoricalCrossEntropy)
    huber = get_loss({"name": "huber", "delta": 2.0})
    assert huber.delta == pytest.approx(2.0)
    with pytest.raises(ValueError):
        get_loss("hinge")
