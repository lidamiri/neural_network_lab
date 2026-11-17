import numpy as np


class Activation:
    name = "activation"

    def forward(self, z):
        raise NotImplementedError

    def backward(self, z, a, grad_output):
        raise NotImplementedError

    def get_config(self):
        return {"name": self.name}

    def __repr__(self):
        return f"{self.__class__.__name__}()"


class Linear(Activation):
    name = "linear"

    def forward(self, z):
        return z

    def backward(self, z, a, grad_output):
        return grad_output


class ReLU(Activation):
    name = "relu"

    def forward(self, z):
        return np.maximum(z, 0.0)

    def backward(self, z, a, grad_output):
        return grad_output * (z > 0.0)


class LeakyReLU(Activation):
    name = "leaky_relu"

    def __init__(self, alpha=0.01):
        self.alpha = float(alpha)

    def forward(self, z):
        return np.where(z > 0.0, z, self.alpha * z)

    def backward(self, z, a, grad_output):
        return grad_output * np.where(z > 0.0, 1.0, self.alpha)

    def get_config(self):
        return {"name": self.name, "alpha": self.alpha}


class ELU(Activation):
    name = "elu"

    def __init__(self, alpha=1.0):
        self.alpha = float(alpha)

    def forward(self, z):
        return np.where(z > 0.0, z, self.alpha * (np.exp(np.minimum(z, 0.0)) - 1.0))

    def backward(self, z, a, grad_output):
        local = np.where(z > 0.0, 1.0, a + self.alpha)
        return grad_output * local

    def get_config(self):
        return {"name": self.name, "alpha": self.alpha}


class Sigmoid(Activation):
    name = "sigmoid"

    def forward(self, z):
        out = np.empty_like(z, dtype=float)
        positive = z >= 0.0
        out[positive] = 1.0 / (1.0 + np.exp(-z[positive]))
        exp_z = np.exp(z[~positive])
        out[~positive] = exp_z / (1.0 + exp_z)
        return out

    def backward(self, z, a, grad_output):
        return grad_output * a * (1.0 - a)


class Tanh(Activation):
    name = "tanh"

    def forward(self, z):
        return np.tanh(z)

    def backward(self, z, a, grad_output):
        return grad_output * (1.0 - a * a)


class Softmax(Activation):
    name = "softmax"

    def forward(self, z):
        shifted = z - np.max(z, axis=-1, keepdims=True)
        exp = np.exp(shifted)
        return exp / np.sum(exp, axis=-1, keepdims=True)

    def backward(self, z, a, grad_output):
        dot = np.sum(grad_output * a, axis=-1, keepdims=True)
        return a * (grad_output - dot)


_REGISTRY = {cls.name: cls for cls in (Linear, ReLU, LeakyReLU, ELU, Sigmoid, Tanh, Softmax)}


def available_activations():
    return sorted(_REGISTRY)


def get_activation(spec):
    if isinstance(spec, Activation):
        return spec
    if spec is None:
        return Linear()
    if isinstance(spec, str):
        name, params = spec, {}
    elif isinstance(spec, dict):
        params = dict(spec)
        name = params.pop("name")
    else:
        raise TypeError(f"unsupported activation specification: {spec!r}")
    key = name.lower()
    if key not in _REGISTRY:
        raise ValueError(f"unknown activation '{name}', expected one of {available_activations()}")
    return _REGISTRY[key](**params)
