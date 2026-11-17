import numpy as np

from .builder import build_model


def _objective(model, x, y):
    outputs = model.forward(x, training=True)
    return model.loss.forward(outputs, y) + model.regularizer.penalty(model.layers)


def _analytic_gradients(model, x, y):
    outputs = model.forward(x, training=True)
    model.backward(model.loss.backward(outputs, y))
    model.regularizer.add_gradients(model.layers)
    gradients = {}
    for index, layer in enumerate(model.layers):
        for name, value, grad in layer.parameter_items():
            if grad is not None:
                gradients[f"{index}.{name}"] = (value, np.array(grad, copy=True))
    return gradients


def _numerical_error(model, x, y, value, analytic, eps, samples, rng):
    flat_count = value.size
    picks = rng.choice(flat_count, size=min(samples, flat_count), replace=False)
    worst = 0.0
    for flat_index in picks:
        index = np.unravel_index(flat_index, value.shape)
        original = value[index]
        value[index] = original + eps
        plus = _objective(model, x, y)
        value[index] = original - eps
        minus = _objective(model, x, y)
        value[index] = original
        numerical = (plus - minus) / (2.0 * eps)
        exact = analytic[index]
        relative = abs(exact - numerical) / max(1e-8, abs(exact) + abs(numerical))
        worst = max(worst, relative)
    return worst


CASES = [
    {
        "name": "dense-tanh-softmax-cce",
        "architecture": [
            {"type": "dense", "units": 6, "activation": "tanh"},
            {"type": "dense", "units": 3, "activation": "softmax"},
        ],
        "loss": "categorical_crossentropy",
        "l2": 0.01,
        "input_dim": 4,
        "targets": "one_hot",
        "outputs": 3,
        "rows": 8,
    },
    {
        "name": "elu-batchnorm-mse",
        "architecture": [
            {"type": "dense", "units": 5, "activation": "elu"},
            {"type": "batchnorm"},
            {"type": "dense", "units": 1, "activation": "linear"},
        ],
        "loss": "mse",
        "l2": 0.0,
        "input_dim": 3,
        "targets": "regression",
        "outputs": 1,
        "rows": 10,
    },
    {
        "name": "sigmoid-bce",
        "architecture": [
            {"type": "dense", "units": 4, "activation": "sigmoid"},
            {"type": "dense", "units": 1, "activation": "sigmoid"},
        ],
        "loss": "binary_crossentropy",
        "l2": 0.001,
        "input_dim": 2,
        "targets": "binary",
        "outputs": 1,
        "rows": 12,
    },
    {
        "name": "huber-tanh",
        "architecture": [
            {"type": "dense", "units": 4, "activation": "tanh"},
            {"type": "batchnorm"},
            {"type": "dense", "units": 1, "activation": "linear"},
        ],
        "loss": {"name": "huber", "delta": 0.5},
        "l2": 0.0,
        "input_dim": 3,
        "targets": "regression",
        "outputs": 1,
        "rows": 9,
    },
]


def _make_targets(rng, case):
    rows = case["rows"]
    if case["targets"] == "one_hot":
        labels = rng.integers(0, case["outputs"], size=rows)
        return np.eye(case["outputs"])[labels]
    if case["targets"] == "binary":
        return rng.integers(0, 2, size=(rows, 1)).astype(float)
    return rng.normal(0.0, 1.0, size=(rows, case["outputs"]))


def check_case(case, seed=0, eps=1e-4, samples=25, tolerance=1e-5):
    rng = np.random.default_rng(seed)
    model = build_model(case["architecture"], case["input_dim"], loss=case["loss"], l2=case["l2"], seed=seed)
    x = rng.normal(0.0, 1.0, size=(case["rows"], case["input_dim"]))
    y = _make_targets(rng, case)
    gradients = _analytic_gradients(model, x, y)
    worst = 0.0
    details = {}
    for key, (value, analytic) in gradients.items():
        error = _numerical_error(model, x, y, value, analytic, eps, samples, rng)
        details[key] = error
        worst = max(worst, error)
    return {"name": case["name"], "max_relative_error": worst, "passed": worst < tolerance, "parameters": details}


def run_gradcheck(seed=0, tolerance=1e-5, cases=None):
    results = [check_case(case, seed=seed, tolerance=tolerance) for case in (cases or CASES)]
    return results
