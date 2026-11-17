import numpy as np

from .utils import make_rng


class Standardizer:
    def __init__(self):
        self.mean = None
        self.scale = None

    def fit(self, x):
        array = np.asarray(x, dtype=float)
        self.mean = array.mean(axis=0)
        scale = array.std(axis=0)
        self.scale = np.where(scale < 1e-12, 1.0, scale)
        return self

    def transform(self, x):
        self._require_fitted()
        return (np.asarray(x, dtype=float) - self.mean) / self.scale

    def fit_transform(self, x):
        return self.fit(x).transform(x)

    def inverse_transform(self, x):
        self._require_fitted()
        return np.asarray(x, dtype=float) * self.scale + self.mean

    def _require_fitted(self):
        if self.mean is None:
            raise RuntimeError("the scaler has not been fitted")

    def to_dict(self):
        self._require_fitted()
        return {"type": "standardize", "mean": self.mean.tolist(), "scale": self.scale.tolist()}


class MinMaxScaler:
    def __init__(self, feature_range=(0.0, 1.0)):
        self.low, self.high = (float(feature_range[0]), float(feature_range[1]))
        self.minimum = None
        self.span = None

    def fit(self, x):
        array = np.asarray(x, dtype=float)
        self.minimum = array.min(axis=0)
        span = array.max(axis=0) - self.minimum
        self.span = np.where(span < 1e-12, 1.0, span)
        return self

    def transform(self, x):
        if self.minimum is None:
            raise RuntimeError("the scaler has not been fitted")
        unit = (np.asarray(x, dtype=float) - self.minimum) / self.span
        return self.low + unit * (self.high - self.low)

    def fit_transform(self, x):
        return self.fit(x).transform(x)


def one_hot(labels, num_classes):
    labels = np.asarray(labels, dtype=int).reshape(-1)
    if labels.size and (labels.min() < 0 or labels.max() >= num_classes):
        raise ValueError("labels are outside the range of num_classes")
    encoded = np.zeros((labels.shape[0], num_classes), dtype=float)
    encoded[np.arange(labels.shape[0]), labels] = 1.0
    return encoded


def prepare_targets(labels, mode, num_classes):
    if mode == "one_hot":
        return one_hot(labels, num_classes)
    if mode == "binary":
        return np.asarray(labels, dtype=float).reshape(-1, 1)
    if mode == "regression":
        return np.asarray(labels, dtype=float).reshape(len(labels), -1)
    raise ValueError(f"unknown target mode '{mode}'")


def train_val_test_split(x, y, val_fraction=0.0, test_fraction=0.0, rng=None, stratify=True):
    if not 0.0 <= val_fraction < 1.0 or not 0.0 <= test_fraction < 1.0:
        raise ValueError("split fractions must be in [0, 1)")
    if val_fraction + test_fraction >= 1.0:
        raise ValueError("validation and test fractions must leave data for training")
    rng = rng if rng is not None else make_rng(0)
    x = np.asarray(x)
    y = np.asarray(y)
    if len(x) != len(y):
        raise ValueError("x and y must have the same number of samples")
    if stratify and y.ndim == 1:
        groups = [np.where(y == value)[0] for value in np.unique(y)]
    else:
        groups = [np.arange(len(x))]
    train_idx, val_idx, test_idx = [], [], []
    for group in groups:
        shuffled = rng.permutation(group)
        n_test = int(round(len(shuffled) * test_fraction))
        n_val = int(round(len(shuffled) * val_fraction))
        test_idx.append(shuffled[:n_test])
        val_idx.append(shuffled[n_test:n_test + n_val])
        train_idx.append(shuffled[n_test + n_val:])
    train = rng.permutation(np.concatenate(train_idx).astype(int))
    validation = rng.permutation(np.concatenate(val_idx).astype(int))
    test = rng.permutation(np.concatenate(test_idx).astype(int))
    return (x[train], y[train]), (x[validation], y[validation]), (x[test], y[test])


def iterate_minibatches(x, y, batch_size, rng=None, shuffle=True, drop_last=False):
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    count = len(x)
    order = rng.permutation(count) if shuffle and rng is not None else np.arange(count)
    for start in range(0, count, batch_size):
        batch = order[start:start + batch_size]
        if drop_last and len(batch) < batch_size:
            break
        yield x[batch], y[batch]
