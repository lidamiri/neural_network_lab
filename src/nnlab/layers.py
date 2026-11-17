import numpy as np

from .activations import get_activation
from .initializers import get_initializer


class Layer:
    kind = "layer"

    def __init__(self):
        self.params = {}
        self.grads = {}
        self.buffers = {}
        self.input_dim = None
        self.output_dim = None
        self.built = False

    def build(self, input_dim, rng):
        self.input_dim = int(input_dim)
        self.output_dim = int(input_dim)
        self.built = True
        return self.output_dim

    def forward(self, x, training=False):
        raise NotImplementedError

    def backward(self, grad_output):
        raise NotImplementedError

    def parameter_items(self):
        return [(name, value, self.grads.get(name)) for name, value in self.params.items()]

    def num_parameters(self):
        return int(sum(value.size for value in self.params.values()))

    def get_config(self):
        return {"type": self.kind}

    def describe(self):
        return self.kind

    def state_arrays(self):
        arrays = {}
        for name, value in self.params.items():
            arrays[name] = value
        for name, value in self.buffers.items():
            arrays[name] = value
        return arrays

    def load_arrays(self, arrays):
        for name in list(self.params) + list(self.buffers):
            target = self.params if name in self.params else self.buffers
            incoming = np.asarray(arrays[name], dtype=float)
            if incoming.shape != target[name].shape:
                raise ValueError(
                    f"shape mismatch for {self.kind}.{name}: expected {target[name].shape}, got {incoming.shape}"
                )
            target[name] = incoming.copy()


class Dense(Layer):
    kind = "dense"

    def __init__(self, units, activation="linear", initializer="he_normal"):
        super().__init__()
        if int(units) < 1:
            raise ValueError("units must be a positive integer")
        self.units = int(units)
        self.activation = get_activation(activation)
        self.initializer_name = initializer if isinstance(initializer, str) else "custom"
        self.initializer = get_initializer(initializer)
        self.cache = None

    def build(self, input_dim, rng):
        self.input_dim = int(input_dim)
        self.output_dim = self.units
        self.params["W"] = self.initializer(rng, (self.input_dim, self.units))
        self.params["b"] = np.zeros(self.units)
        self.built = True
        return self.output_dim

    def forward(self, x, training=False):
        z = x @ self.params["W"] + self.params["b"]
        a = self.activation.forward(z)
        self.cache = (x, z, a)
        return a

    def backward(self, grad_output):
        x, z, a = self.cache
        dz = self.activation.backward(z, a, grad_output)
        self.grads["W"] = x.T @ dz
        self.grads["b"] = dz.sum(axis=0)
        return dz @ self.params["W"].T

    def get_config(self):
        return {
            "type": self.kind,
            "units": self.units,
            "activation": self.activation.get_config(),
            "initializer": self.initializer_name,
        }

    def describe(self):
        return f"dense({self.units}, {self.activation.name})"


class Dropout(Layer):
    kind = "dropout"

    def __init__(self, rate=0.5):
        super().__init__()
        if not 0.0 <= float(rate) < 1.0:
            raise ValueError("dropout rate must be in [0, 1)")
        self.rate = float(rate)
        self.rng = None
        self.mask = None

    def build(self, input_dim, rng):
        self.input_dim = int(input_dim)
        self.output_dim = int(input_dim)
        self.rng = rng
        self.built = True
        return self.output_dim

    def forward(self, x, training=False):
        if not training or self.rate == 0.0:
            self.mask = None
            return x
        keep = 1.0 - self.rate
        self.mask = (self.rng.random(x.shape) < keep) / keep
        return x * self.mask

    def backward(self, grad_output):
        if self.mask is None:
            return grad_output
        return grad_output * self.mask

    def get_config(self):
        return {"type": self.kind, "rate": self.rate}

    def describe(self):
        return f"dropout({self.rate})"


class BatchNorm(Layer):
    kind = "batchnorm"

    def __init__(self, momentum=0.9, epsilon=1e-5):
        super().__init__()
        self.momentum = float(momentum)
        self.epsilon = float(epsilon)
        self.cache = None

    def build(self, input_dim, rng):
        width = int(input_dim)
        self.input_dim = width
        self.output_dim = width
        self.params = {"gamma": np.ones(width), "beta": np.zeros(width)}
        self.buffers = {"running_mean": np.zeros(width), "running_var": np.ones(width)}
        self.built = True
        return width

    def forward(self, x, training=False):
        if training:
            mean = x.mean(axis=0)
            var = x.var(axis=0)
            self.buffers["running_mean"] = self.momentum * self.buffers["running_mean"] + (1.0 - self.momentum) * mean
            self.buffers["running_var"] = self.momentum * self.buffers["running_var"] + (1.0 - self.momentum) * var
        else:
            mean = self.buffers["running_mean"]
            var = self.buffers["running_var"]
        inv_std = 1.0 / np.sqrt(var + self.epsilon)
        x_hat = (x - mean) * inv_std
        self.cache = (x_hat, inv_std) if training else None
        return self.params["gamma"] * x_hat + self.params["beta"]

    def backward(self, grad_output):
        x_hat, inv_std = self.cache
        n = grad_output.shape[0]
        self.grads["gamma"] = np.sum(grad_output * x_hat, axis=0)
        self.grads["beta"] = np.sum(grad_output, axis=0)
        dx_hat = grad_output * self.params["gamma"]
        return (inv_std / n) * (
            n * dx_hat - np.sum(dx_hat, axis=0) - x_hat * np.sum(dx_hat * x_hat, axis=0)
        )

    def get_config(self):
        return {"type": self.kind, "momentum": self.momentum, "epsilon": self.epsilon}

    def describe(self):
        return f"batchnorm({self.input_dim})"
