from dataclasses import dataclass

import numpy as np

from .digits_font import DIGIT_BITMAPS
from .mnist import load_mnist_arrays
from .preprocessing import Standardizer, prepare_targets, train_val_test_split
from .utils import make_rng


@dataclass
class Dataset:
    name: str
    task: str
    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    num_classes: int
    labels_train: np.ndarray
    labels_val: np.ndarray
    labels_test: np.ndarray

    @property
    def input_dim(self):
        return int(self.x_train.shape[1])

    @property
    def output_dim(self):
        return int(self.y_train.shape[1])

    def describe(self):
        return {
            "name": self.name,
            "task": self.task,
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "num_classes": self.num_classes,
            "train": int(len(self.x_train)),
            "validation": int(len(self.x_val)),
            "test": int(len(self.x_test)),
        }


def make_xor(rng, samples=400, noise=0.05):
    x = rng.uniform(-1.0, 1.0, size=(samples, 2))
    labels = ((x[:, 0] > 0.0) ^ (x[:, 1] > 0.0)).astype(int)
    x = x + rng.normal(0.0, noise, size=x.shape)
    return x, labels, 2


def make_spirals(rng, samples_per_class=200, classes=3, noise=0.15):
    xs, labels = [], []
    for label in range(classes):
        radius = np.linspace(0.05, 1.0, samples_per_class)
        angle = np.linspace(label * 4.0, (label + 1) * 4.0, samples_per_class)
        angle = angle + rng.normal(0.0, noise, size=samples_per_class)
        xs.append(np.stack([radius * np.sin(angle), radius * np.cos(angle)], axis=1))
        labels.append(np.full(samples_per_class, label))
    return np.concatenate(xs), np.concatenate(labels), classes


def make_moons(rng, samples=400, noise=0.1):
    half = samples // 2
    theta_outer = np.linspace(0.0, np.pi, half)
    theta_inner = np.linspace(0.0, np.pi, samples - half)
    outer = np.stack([np.cos(theta_outer), np.sin(theta_outer)], axis=1)
    inner = np.stack([1.0 - np.cos(theta_inner), 1.0 - np.sin(theta_inner) - 0.5], axis=1)
    x = np.concatenate([outer, inner])
    labels = np.concatenate([np.zeros(half, dtype=int), np.ones(samples - half, dtype=int)])
    x = x + rng.normal(0.0, noise, size=x.shape)
    return x, labels, 2


def _render_digit(rng, label, size, noise):
    glyph = np.array([[int(c) for c in row] for row in DIGIT_BITMAPS[label]], dtype=float)
    glyph = np.kron(glyph, np.ones((2, 2)))
    canvas = np.zeros((size, size))
    rows, cols = glyph.shape
    top = rng.integers(0, size - rows + 1)
    left = rng.integers(0, size - cols + 1)
    intensity = rng.uniform(0.7, 1.0)
    canvas[top:top + rows, left:left + cols] = glyph * intensity
    canvas = canvas + rng.normal(0.0, noise, size=canvas.shape)
    return np.clip(canvas, 0.0, 1.0).reshape(-1)


def make_digits(rng, samples_per_class=150, size=16, noise=0.08):
    features, labels = [], []
    for label in range(10):
        for _ in range(samples_per_class):
            features.append(_render_digit(rng, label, size, noise))
            labels.append(label)
    return np.stack(features), np.array(labels, dtype=int), 10


def make_regression(rng, samples=600, noise=0.08):
    x = rng.uniform(-1.0, 1.0, size=(samples, 1))
    y = np.sin(2.0 * np.pi * x) * x + rng.normal(0.0, noise, size=x.shape)
    return x, y, None


def make_sine_surface(rng, samples=800, noise=0.05):
    x = rng.uniform(-2.0, 2.0, size=(samples, 2))
    y = np.sin(x[:, :1]) * np.cos(x[:, 1:]) + 0.5 * x[:, :1] * x[:, 1:] / 4.0
    y = y + rng.normal(0.0, noise, size=y.shape)
    return x, y, None


GENERATORS = {
    "xor": make_xor,
    "spirals": make_spirals,
    "moons": make_moons,
    "digits": make_digits,
    "regression": make_regression,
    "sine_surface": make_sine_surface,
}


def available_datasets():
    return sorted(list(GENERATORS) + ["mnist"])


def _build_split_dataset(name, x, labels, num_classes, target, options, rng):
    val_fraction = float(options.get("val_fraction", 0.15))
    test_fraction = float(options.get("test_fraction", 0.15))
    stratify = target != "regression"
    splits = train_val_test_split(
        x,
        labels if labels is not None else x,
        val_fraction=val_fraction,
        test_fraction=test_fraction,
        rng=rng,
        stratify=stratify,
    )
    (x_train, y_raw_train), (x_val, y_raw_val), (x_test, y_raw_test) = splits
    if options.get("standardize", target == "regression"):
        scaler = Standardizer().fit(x_train)
        x_train, x_val, x_test = scaler.transform(x_train), scaler.transform(x_val), scaler.transform(x_test)
    if target == "regression":
        y_train, y_val, y_test = y_raw_train, y_raw_val, y_raw_test
        if options.get("standardize_targets", True):
            target_scaler = Standardizer().fit(y_train)
            y_train = target_scaler.transform(y_train)
            y_val = target_scaler.transform(y_val)
            y_test = target_scaler.transform(y_test)
        labels_train = labels_val = labels_test = None
        num_classes = 0
        task = "regression"
    else:
        y_train = prepare_targets(y_raw_train, target, num_classes)
        y_val = prepare_targets(y_raw_val, target, num_classes)
        y_test = prepare_targets(y_raw_test, target, num_classes)
        labels_train, labels_val, labels_test = y_raw_train, y_raw_val, y_raw_test
        task = "classification"
    return Dataset(
        name=name,
        task=task,
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        x_test=x_test,
        y_test=y_test,
        num_classes=num_classes,
        labels_train=labels_train,
        labels_val=labels_val,
        labels_test=labels_test,
    )


def load_dataset(name, options=None, seed=0):
    options = dict(options or {})
    rng = make_rng(seed)
    key = name.lower()
    if key == "mnist":
        return _load_mnist(options, rng)
    if key not in GENERATORS:
        raise ValueError(f"unknown dataset '{name}', expected one of {available_datasets()}")
    generator = GENERATORS[key]
    kwargs = {k: v for k, v in options.items() if k in _GENERATOR_OPTIONS.get(key, ())}
    x, labels, num_classes = generator(rng, **kwargs)
    if labels is None:
        target = "regression"
        labels_for_split = None
    else:
        target = options.get("target", "one_hot")
        labels_for_split = labels
    if key == "regression" or key == "sine_surface":
        target = "regression"
    return _build_split_dataset(key, x, labels_for_split, num_classes, target, options, rng)


def _load_mnist(options, rng):
    directory = options.get("directory", "data/mnist")
    arrays = load_mnist_arrays(directory)
    limit = options.get("limit")
    x_all, y_all = arrays["x_train"], arrays["y_train"]
    if limit:
        x_all, y_all = x_all[:int(limit)], y_all[:int(limit)]
    val_fraction = float(options.get("val_fraction", 0.1))
    (x_train, y_train), (x_val, y_val), _ = train_val_test_split(
        x_all, y_all, val_fraction=val_fraction, test_fraction=0.0, rng=rng, stratify=True
    )
    x_test, y_test = arrays["x_test"], arrays["y_test"]
    target = options.get("target", "one_hot")
    return Dataset(
        name="mnist",
        task="classification",
        x_train=x_train,
        y_train=prepare_targets(y_train, target, 10),
        x_val=x_val,
        y_val=prepare_targets(y_val, target, 10),
        x_test=x_test,
        y_test=prepare_targets(y_test, target, 10),
        num_classes=10,
        labels_train=y_train,
        labels_val=y_val,
        labels_test=y_test,
    )


_GENERATOR_OPTIONS = {
    "xor": ("samples", "noise"),
    "spirals": ("samples_per_class", "classes", "noise"),
    "moons": ("samples", "noise"),
    "digits": ("samples_per_class", "size", "noise"),
    "regression": ("samples", "noise"),
    "sine_surface": ("samples", "noise"),
}
