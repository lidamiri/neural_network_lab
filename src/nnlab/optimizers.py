import numpy as np


class Optimizer:
    name = "optimizer"
    decoupled_decay = False

    def __init__(self, learning_rate=0.01, weight_decay=0.0, clip_norm=None):
        if float(learning_rate) <= 0.0:
            raise ValueError("learning_rate must be positive")
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.clip_norm = None if clip_norm is None else float(clip_norm)
        self.iterations = 0
        self.slots = {}

    def _slot(self, key, slot_name, reference):
        full_key = (key, slot_name)
        if full_key not in self.slots:
            self.slots[full_key] = np.zeros_like(reference)
        return self.slots[full_key]

    def _is_weight(self, key):
        return key.endswith(".W")

    def _clip(self, groups):
        if self.clip_norm is None:
            return groups
        total = np.sqrt(sum(float(np.sum(grad ** 2)) for _, _, grad in groups))
        if total <= self.clip_norm or total == 0.0:
            return groups
        scale = self.clip_norm / (total + 1e-12)
        return [(key, param, grad * scale) for key, param, grad in groups]

    def step(self, groups):
        self.iterations += 1
        for key, param, grad in self._clip(groups):
            self.apply(key, param, grad)

    def apply(self, key, param, grad):
        raise NotImplementedError

    def get_config(self):
        return {
            "name": self.name,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "clip_norm": self.clip_norm,
        }


class SGD(Optimizer):
    name = "sgd"

    def __init__(self, learning_rate=0.01, momentum=0.0, nesterov=False, weight_decay=0.0, clip_norm=None):
        super().__init__(learning_rate=learning_rate, weight_decay=weight_decay, clip_norm=clip_norm)
        if not 0.0 <= float(momentum) < 1.0:
            raise ValueError("momentum must be in [0, 1)")
        self.momentum = float(momentum)
        self.nesterov = bool(nesterov)

    def apply(self, key, param, grad):
        if self.weight_decay and self._is_weight(key):
            grad = grad + self.weight_decay * param
        lr = self.learning_rate
        if self.momentum == 0.0:
            param -= lr * grad
            return
        velocity = self._slot(key, "velocity", param)
        velocity *= self.momentum
        velocity -= lr * grad
        if self.nesterov:
            param += self.momentum * velocity - lr * grad
        else:
            param += velocity

    def get_config(self):
        config = super().get_config()
        config.update({"momentum": self.momentum, "nesterov": self.nesterov})
        return config


class RMSProp(Optimizer):
    name = "rmsprop"

    def __init__(self, learning_rate=0.001, rho=0.9, epsilon=1e-8, weight_decay=0.0, clip_norm=None):
        super().__init__(learning_rate=learning_rate, weight_decay=weight_decay, clip_norm=clip_norm)
        self.rho = float(rho)
        self.epsilon = float(epsilon)

    def apply(self, key, param, grad):
        if self.weight_decay and self._is_weight(key):
            grad = grad + self.weight_decay * param
        cache = self._slot(key, "cache", param)
        cache *= self.rho
        cache += (1.0 - self.rho) * grad * grad
        param -= self.learning_rate * grad / (np.sqrt(cache) + self.epsilon)

    def get_config(self):
        config = super().get_config()
        config.update({"rho": self.rho, "epsilon": self.epsilon})
        return config


class Adam(Optimizer):
    name = "adam"

    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, weight_decay=0.0, clip_norm=None):
        super().__init__(learning_rate=learning_rate, weight_decay=weight_decay, clip_norm=clip_norm)
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        self.epsilon = float(epsilon)

    def apply(self, key, param, grad):
        decay = self.weight_decay if self._is_weight(key) else 0.0
        if decay and not self.decoupled_decay:
            grad = grad + decay * param
        first = self._slot(key, "m", param)
        second = self._slot(key, "v", param)
        first *= self.beta1
        first += (1.0 - self.beta1) * grad
        second *= self.beta2
        second += (1.0 - self.beta2) * grad * grad
        step = max(self.iterations, 1)
        m_hat = first / (1.0 - self.beta1 ** step)
        v_hat = second / (1.0 - self.beta2 ** step)
        if decay and self.decoupled_decay:
            param -= self.learning_rate * decay * param
        param -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)

    def get_config(self):
        config = super().get_config()
        config.update({"beta1": self.beta1, "beta2": self.beta2, "epsilon": self.epsilon})
        return config


class AdamW(Adam):
    name = "adamw"
    decoupled_decay = True

    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, weight_decay=0.01, clip_norm=None):
        super().__init__(
            learning_rate=learning_rate,
            beta1=beta1,
            beta2=beta2,
            epsilon=epsilon,
            weight_decay=weight_decay,
            clip_norm=clip_norm,
        )


_REGISTRY = {cls.name: cls for cls in (SGD, RMSProp, Adam, AdamW)}


def available_optimizers():
    return sorted(_REGISTRY)


def get_optimizer(spec):
    if isinstance(spec, Optimizer):
        return spec
    if isinstance(spec, str):
        name, params = spec, {}
    elif isinstance(spec, dict):
        params = dict(spec)
        name = params.pop("name")
    else:
        raise TypeError(f"unsupported optimizer specification: {spec!r}")
    key = name.lower()
    if key not in _REGISTRY:
        raise ValueError(f"unknown optimizer '{name}', expected one of {available_optimizers()}")
    return _REGISTRY[key](**params)
