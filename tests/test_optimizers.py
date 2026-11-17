import numpy as np
import pytest

from nnlab.optimizers import AdamW, Adam, RMSProp, SGD, available_optimizers, get_optimizer


def minimize(optimizer, steps=400):
    target = np.array([3.0, -2.0])
    param = np.zeros(2)
    for _ in range(steps):
        grad = 2.0 * (param - target)
        optimizer.step([("0.W", param, grad)])
    return param, target


@pytest.mark.parametrize(
    "optimizer",
    [
        SGD(learning_rate=0.05),
        SGD(learning_rate=0.02, momentum=0.9),
        SGD(learning_rate=0.02, momentum=0.9, nesterov=True),
        RMSProp(learning_rate=0.05),
        Adam(learning_rate=0.1),
        AdamW(learning_rate=0.1, weight_decay=0.0),
    ],
)
def test_optimizers_converge_on_a_quadratic(optimizer):
    param, target = minimize(optimizer)
    np.testing.assert_allclose(param, target, atol=0.05)


def test_adam_first_step_has_bias_corrected_magnitude():
    optimizer = Adam(learning_rate=0.01)
    param = np.array([0.0])
    optimizer.step([("0.b", param, np.array([5.0]))])
    assert param[0] == pytest.approx(-0.01, rel=1e-3)


def test_weight_decay_only_applies_to_weights():
    optimizer = SGD(learning_rate=0.1, weight_decay=1.0)
    weight = np.array([[2.0]])
    bias = np.array([2.0])
    optimizer.step([("0.W", weight, np.zeros((1, 1))), ("0.b", bias, np.zeros(1))])
    assert weight[0, 0] == pytest.approx(1.8)
    assert bias[0] == pytest.approx(2.0)


def test_adamw_decays_weights_independently_of_gradient():
    optimizer = AdamW(learning_rate=0.1, weight_decay=0.5)
    weight = np.array([[4.0]])
    optimizer.step([("0.W", weight, np.zeros((1, 1)))])
    assert weight[0, 0] == pytest.approx(4.0 - 0.1 * 0.5 * 4.0)


def test_gradient_clipping_limits_global_norm():
    optimizer = SGD(learning_rate=1.0, clip_norm=1.0)
    first = np.zeros(2)
    second = np.zeros(2)
    optimizer.step([("0.W", first, np.array([3.0, 0.0])), ("1.W", second, np.array([0.0, 4.0]))])
    update = np.concatenate([first, second])
    assert np.linalg.norm(update) == pytest.approx(1.0, rel=1e-6)


def test_registry_and_validation():
    assert set(available_optimizers()) == {"sgd", "rmsprop", "adam", "adamw"}
    assert isinstance(get_optimizer({"name": "adam", "learning_rate": 0.01}), Adam)
    with pytest.raises(ValueError):
        get_optimizer("lbfgs")
    with pytest.raises(ValueError):
        SGD(learning_rate=0.0)
    with pytest.raises(ValueError):
        SGD(momentum=1.0)
