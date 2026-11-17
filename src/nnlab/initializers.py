import numpy as np


def _fans(shape):
    if len(shape) < 2:
        raise ValueError("initializers require a shape with at least two dimensions")
    return int(shape[0]), int(shape[1])


def zeros(rng, shape):
    return np.zeros(shape, dtype=float)


def ones(rng, shape):
    return np.ones(shape, dtype=float)


def xavier_uniform(rng, shape):
    fan_in, fan_out = _fans(shape)
    limit = np.sqrt(6.0 / (fan_in + fan_out))
    return rng.uniform(-limit, limit, size=shape)


def xavier_normal(rng, shape):
    fan_in, fan_out = _fans(shape)
    std = np.sqrt(2.0 / (fan_in + fan_out))
    return rng.normal(0.0, std, size=shape)


def he_uniform(rng, shape):
    fan_in, _ = _fans(shape)
    limit = np.sqrt(6.0 / fan_in)
    return rng.uniform(-limit, limit, size=shape)


def he_normal(rng, shape):
    fan_in, _ = _fans(shape)
    std = np.sqrt(2.0 / fan_in)
    return rng.normal(0.0, std, size=shape)


def lecun_normal(rng, shape):
    fan_in, _ = _fans(shape)
    std = np.sqrt(1.0 / fan_in)
    return rng.normal(0.0, std, size=shape)


_REGISTRY = {
    "zeros": zeros,
    "ones": ones,
    "xavier_uniform": xavier_uniform,
    "xavier_normal": xavier_normal,
    "he_uniform": he_uniform,
    "he_normal": he_normal,
    "lecun_normal": lecun_normal,
}


def available_initializers():
    return sorted(_REGISTRY)


def get_initializer(name):
    if callable(name):
        return name
    key = str(name).lower()
    if key not in _REGISTRY:
        raise ValueError(f"unknown initializer '{name}', expected one of {available_initializers()}")
    return _REGISTRY[key]
