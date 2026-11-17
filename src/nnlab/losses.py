import numpy as np

EPSILON = 1e-12


def _clip_probabilities(p):
    return np.clip(p, EPSILON, 1.0 - EPSILON)


class Loss:
    name = "loss"

    def forward(self, y_pred, y_true):
        raise NotImplementedError

    def backward(self, y_pred, y_true):
        raise NotImplementedError

    def get_config(self):
        return {"name": self.name}


class MeanSquaredError(Loss):
    name = "mse"

    def forward(self, y_pred, y_true):
        return float(np.mean((y_pred - y_true) ** 2))

    def backward(self, y_pred, y_true):
        return 2.0 * (y_pred - y_true) / y_pred.size


class MeanAbsoluteError(Loss):
    name = "mae"

    def forward(self, y_pred, y_true):
        return float(np.mean(np.abs(y_pred - y_true)))

    def backward(self, y_pred, y_true):
        return np.sign(y_pred - y_true) / y_pred.size


class HuberLoss(Loss):
    name = "huber"

    def __init__(self, delta=1.0):
        self.delta = float(delta)

    def forward(self, y_pred, y_true):
        error = y_pred - y_true
        small = np.abs(error) <= self.delta
        values = np.where(small, 0.5 * error ** 2, self.delta * (np.abs(error) - 0.5 * self.delta))
        return float(np.mean(values))

    def backward(self, y_pred, y_true):
        error = y_pred - y_true
        grad = np.clip(error, -self.delta, self.delta)
        return grad / y_pred.size

    def get_config(self):
        return {"name": self.name, "delta": self.delta}


class BinaryCrossEntropy(Loss):
    name = "binary_crossentropy"

    def forward(self, y_pred, y_true):
        p = _clip_probabilities(y_pred)
        losses = y_true * np.log(p) + (1.0 - y_true) * np.log(1.0 - p)
        return float(-np.mean(losses))

    def backward(self, y_pred, y_true):
        p = _clip_probabilities(y_pred)
        return (p - y_true) / (p * (1.0 - p)) / y_pred.size


class CategoricalCrossEntropy(Loss):
    name = "categorical_crossentropy"

    def forward(self, y_pred, y_true):
        p = np.clip(y_pred, EPSILON, 1.0)
        return float(-np.sum(y_true * np.log(p)) / y_pred.shape[0])

    def backward(self, y_pred, y_true):
        p = np.clip(y_pred, EPSILON, 1.0)
        return -y_true / p / y_pred.shape[0]


_REGISTRY = {
    "mse": MeanSquaredError,
    "mean_squared_error": MeanSquaredError,
    "mae": MeanAbsoluteError,
    "mean_absolute_error": MeanAbsoluteError,
    "huber": HuberLoss,
    "binary_crossentropy": BinaryCrossEntropy,
    "bce": BinaryCrossEntropy,
    "categorical_crossentropy": CategoricalCrossEntropy,
    "cce": CategoricalCrossEntropy,
}


def available_losses():
    return sorted(_REGISTRY)


def get_loss(spec):
    if isinstance(spec, Loss):
        return spec
    if isinstance(spec, str):
        name, params = spec, {}
    elif isinstance(spec, dict):
        params = dict(spec)
        name = params.pop("name")
    else:
        raise TypeError(f"unsupported loss specification: {spec!r}")
    key = name.lower()
    if key not in _REGISTRY:
        raise ValueError(f"unknown loss '{name}', expected one of {available_losses()}")
    return _REGISTRY[key](**params)
