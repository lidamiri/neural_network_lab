import numpy as np


class L2Regularizer:
    def __init__(self, coefficient=0.0):
        if float(coefficient) < 0.0:
            raise ValueError("l2 coefficient must be non-negative")
        self.coefficient = float(coefficient)

    def _weighted_layers(self, layers):
        return [layer for layer in layers if "W" in layer.params]

    def penalty(self, layers):
        if self.coefficient == 0.0:
            return 0.0
        total = sum(float(np.sum(layer.params["W"] ** 2)) for layer in self._weighted_layers(layers))
        return 0.5 * self.coefficient * total

    def add_gradients(self, layers):
        if self.coefficient == 0.0:
            return
        for layer in self._weighted_layers(layers):
            if layer.grads.get("W") is not None:
                layer.grads["W"] = layer.grads["W"] + self.coefficient * layer.params["W"]
