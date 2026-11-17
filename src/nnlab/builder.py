from .layers import BatchNorm, Dense, Dropout
from .model import Sequential

LAYER_TYPES = {
    "dense": Dense,
    "dropout": Dropout,
    "batchnorm": BatchNorm,
}


def layer_from_spec(spec):
    params = dict(spec)
    kind = params.pop("type", None)
    if kind not in LAYER_TYPES:
        raise ValueError(f"unknown layer type '{kind}', expected one of {sorted(LAYER_TYPES)}")
    return LAYER_TYPES[kind](**params)


def build_model(architecture, input_dim, loss="mse", l2=0.0, seed=None):
    layers = [layer_from_spec(spec) for spec in architecture]
    model = Sequential(layers, loss=loss, seed=seed, l2=l2)
    model.build(input_dim)
    return model
