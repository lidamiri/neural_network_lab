import numpy as np

from .losses import get_loss
from .regularization import L2Regularizer
from .utils import make_rng


class Sequential:
    def __init__(self, layers=None, loss="mse", seed=None, l2=0.0):
        self.layers = []
        self.loss = get_loss(loss)
        self.rng = make_rng(seed)
        self.regularizer = L2Regularizer(l2)
        self.input_dim = None
        self.output_dim = None
        for layer in layers or []:
            self.add(layer)

    def add(self, layer):
        if self.input_dim is not None:
            raise RuntimeError("cannot add layers after the model has been built")
        self.layers.append(layer)
        return self

    def build(self, input_dim):
        if not self.layers:
            raise ValueError("the model has no layers")
        dimension = int(input_dim)
        self.input_dim = dimension
        for layer in self.layers:
            dimension = layer.build(dimension, self.rng)
        self.output_dim = dimension
        return self

    def _require_built(self):
        if self.input_dim is None:
            raise RuntimeError("the model must be built before it is used")

    def _check_inputs(self, x):
        array = np.asarray(x, dtype=float)
        if array.ndim != 2 or array.shape[1] != self.input_dim:
            raise ValueError(f"expected inputs of shape (n, {self.input_dim}), got {array.shape}")
        return array

    def forward(self, x, training=False):
        self._require_built()
        out = self._check_inputs(x)
        for layer in self.layers:
            out = layer.forward(out, training=training)
        return out

    def predict(self, x, batch_size=None):
        self._require_built()
        array = self._check_inputs(x)
        if batch_size is None or len(array) <= batch_size:
            return self.forward(array, training=False)
        pieces = [
            self.forward(array[start:start + batch_size], training=False)
            for start in range(0, len(array), batch_size)
        ]
        return np.concatenate(pieces, axis=0)

    def backward(self, grad_output):
        grad = grad_output
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

    def train_on_batch(self, x, y, optimizer):
        y_pred = self.forward(x, training=True)
        data_loss = self.loss.forward(y_pred, y)
        penalty = self.regularizer.penalty(self.layers)
        self.backward(self.loss.backward(y_pred, y))
        self.regularizer.add_gradients(self.layers)
        optimizer.step(self.parameter_groups())
        return data_loss + penalty

    def evaluate_loss(self, x, y, batch_size=512):
        predictions = self.predict(x, batch_size=batch_size)
        return self.loss.forward(predictions, y) + self.regularizer.penalty(self.layers)

    def total_loss(self, x, y):
        predictions = self.forward(x, training=False)
        return self.loss.forward(predictions, y) + self.regularizer.penalty(self.layers)

    def parameter_groups(self):
        groups = []
        for index, layer in enumerate(self.layers):
            for name, value, grad in layer.parameter_items():
                if grad is not None:
                    groups.append((f"{index}.{name}", value, grad))
        return groups

    def num_parameters(self):
        return sum(layer.num_parameters() for layer in self.layers)

    def summary(self):
        self._require_built()
        header = f"{'#':>3}  {'layer':<28}{'output':>8}{'params':>10}"
        rule = "-" * len(header)
        lines = [header, rule]
        for index, layer in enumerate(self.layers):
            lines.append(f"{index:>3}  {layer.describe():<28}{layer.output_dim:>8}{layer.num_parameters():>10}")
        lines.append(rule)
        lines.append(f"total trainable and state parameters: {self.num_parameters()}")
        return "\n".join(lines)

    def get_config(self):
        return {
            "input_dim": self.input_dim,
            "loss": self.loss.get_config(),
            "l2": self.regularizer.coefficient,
            "layers": [layer.get_config() for layer in self.layers],
        }

    def state_dict(self):
        state = {}
        for index, layer in enumerate(self.layers):
            for name, value in layer.state_arrays().items():
                state[f"{index}.{name}"] = np.array(value, dtype=float)
        return state

    def load_state_dict(self, state):
        self._require_built()
        for index, layer in enumerate(self.layers):
            arrays = {}
            for name in layer.state_arrays():
                key = f"{index}.{name}"
                if key not in state:
                    raise KeyError(f"missing parameter '{key}' in state")
                arrays[name] = state[key]
            layer.load_arrays(arrays)
        return self
