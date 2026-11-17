import numpy as np
import pytest

from nnlab.builder import build_model, layer_from_spec
from nnlab.layers import Dense
from nnlab.model import Sequential
from nnlab.optimizers import SGD


ARCHITECTURE = [
    {"type": "dense", "units": 6, "activation": "relu"},
    {"type": "batchnorm"},
    {"type": "dropout", "rate": 0.3},
    {"type": "dense", "units": 3, "activation": "softmax"},
]


def test_build_sets_dimensions_through_the_stack():
    model = build_model(ARCHITECTURE, 4, loss="cce", seed=1)
    assert model.input_dim == 4
    assert model.output_dim == 3
    assert [layer.output_dim for layer in model.layers] == [6, 6, 6, 3]


def test_forward_checks_input_dimension():
    model = build_model(ARCHITECTURE, 4, loss="cce", seed=1)
    with pytest.raises(ValueError):
        model.forward(np.zeros((2, 5)))


def test_unbuilt_model_refuses_to_run():
    model = Sequential([Dense(2)], loss="mse")
    with pytest.raises(RuntimeError):
        model.forward(np.zeros((1, 2)))


def test_layers_cannot_be_added_after_build():
    model = build_model([{"type": "dense", "units": 2}], 3, seed=0)
    with pytest.raises(RuntimeError):
        model.add(Dense(2))


def test_empty_model_cannot_be_built():
    with pytest.raises(ValueError):
        Sequential([], loss="mse").build(3)


def test_predict_in_batches_matches_full_forward(rng):
    model = build_model([{"type": "dense", "units": 5, "activation": "tanh"}, {"type": "dense", "units": 2}], 3, seed=2)
    x = rng.normal(size=(23, 3))
    np.testing.assert_allclose(model.predict(x, batch_size=4), model.predict(x))


def test_summary_lists_every_layer():
    model = build_model(ARCHITECTURE, 4, loss="cce", seed=1)
    text = model.summary()
    assert "dense(6, relu)" in text
    assert "batchnorm(6)" in text
    assert "dropout(0.3)" in text
    assert str(model.num_parameters()) in text


def test_train_on_batch_reduces_loss(rng):
    model = build_model([{"type": "dense", "units": 8, "activation": "tanh"}, {"type": "dense", "units": 1}], 2, loss="mse", seed=3)
    x = rng.normal(size=(64, 2))
    y = (x[:, :1] - 2.0 * x[:, 1:]) * 0.5
    optimizer = SGD(learning_rate=0.05)
    first = model.train_on_batch(x, y, optimizer)
    for _ in range(200):
        last = model.train_on_batch(x, y, optimizer)
    assert last < first * 0.2


def test_parameter_groups_expose_names_and_gradients(rng):
    model = build_model(ARCHITECTURE, 4, loss="cce", seed=1)
    x = rng.normal(size=(5, 4))
    y = np.eye(3)[[0, 1, 2, 0, 1]]
    model.train_on_batch(x, y, SGD(learning_rate=1e-9))
    names = [name for name, _, _ in model.parameter_groups()]
    assert "0.W" in names and "0.b" in names and "1.gamma" in names and "3.W" in names


def test_l2_penalty_is_added_to_loss(rng):
    plain = build_model([{"type": "dense", "units": 2}], 3, loss="mse", l2=0.0, seed=4)
    decayed = build_model([{"type": "dense", "units": 2}], 3, loss="mse", l2=0.5, seed=4)
    x = rng.normal(size=(6, 3))
    y = rng.normal(size=(6, 2))
    difference = decayed.total_loss(x, y) - plain.total_loss(x, y)
    expected = 0.5 * 0.5 * float(np.sum(decayed.layers[0].params["W"] ** 2))
    assert difference == pytest.approx(expected)


def test_state_dict_round_trip_restores_predictions(rng):
    source = build_model(ARCHITECTURE, 4, loss="cce", seed=5)
    target = build_model(ARCHITECTURE, 4, loss="cce", seed=99)
    x = rng.normal(size=(7, 4))
    target.load_state_dict(source.state_dict())
    np.testing.assert_allclose(source.predict(x), target.predict(x))


def test_load_state_dict_detects_missing_keys():
    model = build_model([{"type": "dense", "units": 2}], 3, seed=0)
    with pytest.raises(KeyError):
        model.load_state_dict({})


def test_layer_from_spec_rejects_unknown_types():
    with pytest.raises(ValueError):
        layer_from_spec({"type": "conv2d", "units": 3})
