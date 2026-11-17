import numpy as np
import pytest

from nnlab.activations import (
    ELU,
    LeakyReLU,
    Linear,
    ReLU,
    Sigmoid,
    Softmax,
    Tanh,
    available_activations,
    get_activation,
)


def numerical_jacobian_vector(activation, z, grad_output, eps=1e-6):
    estimate = np.zeros_like(z)
    for index in np.ndindex(z.shape):
        shifted_up = z.copy()
        shifted_down = z.copy()
        shifted_up[index] += eps
        shifted_down[index] -= eps
        up = activation.forward(shifted_up)
        down = activation.forward(shifted_down)
        estimate[index] = np.sum(grad_output * (up - down)) / (2 * eps)
    return estimate


@pytest.mark.parametrize("activation", [Linear(), Sigmoid(), Tanh(), ELU(), LeakyReLU(0.1)])
def test_elementwise_backward_matches_numerical(activation, rng):
    z = rng.normal(0.0, 1.0, size=(4, 3))
    z[np.abs(z) < 0.05] = 0.5
    grad_output = rng.normal(size=z.shape)
    a = activation.forward(z)
    analytic = activation.backward(z, a, grad_output)
    numeric = numerical_jacobian_vector(activation, z, grad_output)
    np.testing.assert_allclose(analytic, numeric, atol=1e-6)


def test_relu_zeroes_negative_inputs():
    z = np.array([[-2.0, 0.0, 3.0]])
    np.testing.assert_allclose(ReLU().forward(z), [[0.0, 0.0, 3.0]])
    np.testing.assert_allclose(ReLU().backward(z, None, np.ones_like(z)), [[0.0, 0.0, 1.0]])


def test_sigmoid_is_stable_for_large_magnitudes():
    z = np.array([[-1000.0, 0.0, 1000.0]])
    out = Sigmoid().forward(z)
    assert np.all(np.isfinite(out))
    np.testing.assert_allclose(out, [[0.0, 0.5, 1.0]], atol=1e-12)


def test_softmax_rows_sum_to_one(rng):
    z = rng.normal(size=(5, 7)) * 50.0
    out = Softmax().forward(z)
    np.testing.assert_allclose(out.sum(axis=1), np.ones(5))
    assert np.all(out >= 0.0)


def test_softmax_backward_matches_numerical(rng):
    softmax = Softmax()
    z = rng.normal(size=(3, 4))
    grad_output = rng.normal(size=z.shape)
    a = softmax.forward(z)
    analytic = softmax.backward(z, a, grad_output)
    numeric = numerical_jacobian_vector(softmax, z, grad_output)
    np.testing.assert_allclose(analytic, numeric, atol=1e-6)


def test_registry_lists_all_activations():
    names = available_activations()
    for expected in ["linear", "relu", "leaky_relu", "elu", "sigmoid", "tanh", "softmax"]:
        assert expected in names


def test_get_activation_accepts_string_dict_and_instance():
    assert isinstance(get_activation("tanh"), Tanh)
    leaky = get_activation({"name": "leaky_relu", "alpha": 0.2})
    assert isinstance(leaky, LeakyReLU)
    assert leaky.alpha == pytest.approx(0.2)
    instance = Sigmoid()
    assert get_activation(instance) is instance
    assert isinstance(get_activation(None), Linear)


def test_get_activation_rejects_unknown_names():
    with pytest.raises(ValueError):
        get_activation("swish-plus")
    with pytest.raises(TypeError):
        get_activation(3.5)
