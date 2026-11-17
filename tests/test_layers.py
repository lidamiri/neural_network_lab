import numpy as np
import pytest

from nnlab.layers import BatchNorm, Dense, Dropout


def build(layer, input_dim, rng):
    layer.build(input_dim, rng)
    return layer


def test_dense_shapes_and_parameter_count(rng):
    layer = build(Dense(5, activation="relu"), 3, rng)
    out = layer.forward(rng.normal(size=(7, 3)), training=True)
    assert out.shape == (7, 5)
    assert layer.params["W"].shape == (3, 5)
    assert layer.params["b"].shape == (5,)
    assert layer.num_parameters() == 20


def test_dense_backward_gradients_match_numerical(rng):
    layer = build(Dense(4, activation="tanh", initializer="xavier_uniform"), 3, rng)
    x = rng.normal(size=(6, 3))
    upstream = rng.normal(size=(6, 4))
    layer.forward(x, training=True)
    grad_input = layer.backward(upstream)

    def scalar(values_x, values_w, values_b):
        saved_w, saved_b = layer.params["W"].copy(), layer.params["b"].copy()
        layer.params["W"], layer.params["b"] = values_w, values_b
        out = layer.forward(values_x, training=True)
        layer.params["W"], layer.params["b"] = saved_w, saved_b
        return float(np.sum(out * upstream))

    eps = 1e-6
    numeric_w = np.zeros_like(layer.params["W"])
    for index in np.ndindex(numeric_w.shape):
        up, down = layer.params["W"].copy(), layer.params["W"].copy()
        up[index] += eps
        down[index] -= eps
        numeric_w[index] = (scalar(x, up, layer.params["b"]) - scalar(x, down, layer.params["b"])) / (2 * eps)
    np.testing.assert_allclose(layer.grads["W"], numeric_w, atol=1e-6)

    numeric_x = np.zeros_like(x)
    for index in np.ndindex(x.shape):
        up, down = x.copy(), x.copy()
        up[index] += eps
        down[index] -= eps
        numeric_x[index] = (scalar(up, layer.params["W"], layer.params["b"]) - scalar(down, layer.params["W"], layer.params["b"])) / (2 * eps)
    np.testing.assert_allclose(grad_input, numeric_x, atol=1e-6)


def test_dense_rejects_non_positive_units():
    with pytest.raises(ValueError):
        Dense(0)


def test_dropout_is_identity_at_inference(rng):
    layer = build(Dropout(0.5), 10, rng)
    x = rng.normal(size=(4, 10))
    np.testing.assert_array_equal(layer.forward(x, training=False), x)
    np.testing.assert_array_equal(layer.backward(np.ones_like(x)), np.ones_like(x))


def test_dropout_preserves_expected_activation_scale(rng):
    layer = build(Dropout(0.3), 2000, rng)
    x = np.ones((50, 2000))
    out = layer.forward(x, training=True)
    assert out.mean() == pytest.approx(1.0, abs=0.02)
    zeros_fraction = np.mean(out == 0.0)
    assert zeros_fraction == pytest.approx(0.3, abs=0.02)


def test_dropout_backward_uses_the_same_mask(rng):
    layer = build(Dropout(0.5), 6, rng)
    x = rng.normal(size=(3, 6))
    out = layer.forward(x, training=True)
    grad = layer.backward(np.ones_like(x))
    np.testing.assert_array_equal(grad == 0.0, out == 0.0)


def test_dropout_rejects_invalid_rate():
    with pytest.raises(ValueError):
        Dropout(1.0)
    with pytest.raises(ValueError):
        Dropout(-0.1)


def test_batchnorm_normalizes_training_batches(rng):
    layer = build(BatchNorm(), 4, rng)
    x = rng.normal(5.0, 3.0, size=(256, 4))
    out = layer.forward(x, training=True)
    np.testing.assert_allclose(out.mean(axis=0), np.zeros(4), atol=1e-7)
    np.testing.assert_allclose(out.std(axis=0), np.ones(4), atol=1e-3)


def test_batchnorm_uses_running_statistics_at_inference(rng):
    layer = build(BatchNorm(momentum=0.0), 3, rng)
    x = rng.normal(2.0, 1.0, size=(128, 3))
    layer.forward(x, training=True)
    out = layer.forward(x, training=False)
    expected = (x - x.mean(axis=0)) / np.sqrt(x.var(axis=0) + layer.epsilon)
    np.testing.assert_allclose(out, expected, atol=1e-6)


def test_batchnorm_backward_matches_numerical(rng):
    layer = build(BatchNorm(), 3, rng)
    x = rng.normal(size=(8, 3))
    upstream = rng.normal(size=(8, 3))
    layer.forward(x, training=True)
    analytic = layer.backward(upstream)
    eps = 1e-6
    numeric = np.zeros_like(x)
    for index in np.ndindex(x.shape):
        up, down = x.copy(), x.copy()
        up[index] += eps
        down[index] -= eps
        plus = np.sum(build_bn_output(layer, up) * upstream)
        minus = np.sum(build_bn_output(layer, down) * upstream)
        numeric[index] = (plus - minus) / (2 * eps)
    np.testing.assert_allclose(analytic, numeric, atol=1e-5)


def build_bn_output(layer, x):
    return layer.forward(x, training=True)


def test_state_arrays_round_trip(rng):
    layer = build(Dense(3), 2, rng)
    arrays = layer.state_arrays()
    replacement = {name: np.zeros_like(value) for name, value in arrays.items()}
    layer.load_arrays(replacement)
    for name in replacement:
        assert np.all(layer.state_arrays()[name] == 0.0)


def test_load_arrays_rejects_wrong_shapes(rng):
    layer = build(Dense(3), 2, rng)
    with pytest.raises(ValueError):
        layer.load_arrays({"W": np.zeros((5, 3)), "b": np.zeros(3)})
