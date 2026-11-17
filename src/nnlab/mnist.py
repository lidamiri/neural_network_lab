import gzip
import os
import pathlib
import struct

import numpy as np

FILES = {
    "x_train": "train-images-idx3-ubyte",
    "y_train": "train-labels-idx1-ubyte",
    "x_test": "t10k-images-idx3-ubyte",
    "y_test": "t10k-labels-idx1-ubyte",
}


def _open_maybe_gzip(path):
    if str(path).endswith(".gz"):
        return gzip.open(path, "rb")
    return open(path, "rb")


def _resolve(directory, stem):
    base = pathlib.Path(directory)
    for candidate in (base / stem, base / f"{stem}.gz"):
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"missing MNIST file '{stem}' (optionally gzipped) in {base}; "
        "place the four IDX files there or use the synthetic digits dataset"
    )


def read_idx_images(path):
    with _open_maybe_gzip(path) as handle:
        magic, count, rows, cols = struct.unpack(">IIII", handle.read(16))
        if magic != 2051:
            raise ValueError(f"unexpected image magic number {magic} in {path}")
        data = np.frombuffer(handle.read(), dtype=np.uint8)
    return data.reshape(count, rows * cols).astype(np.float64) / 255.0


def read_idx_labels(path):
    with _open_maybe_gzip(path) as handle:
        magic, count = struct.unpack(">II", handle.read(8))
        if magic != 2049:
            raise ValueError(f"unexpected label magic number {magic} in {path}")
        data = np.frombuffer(handle.read(), dtype=np.uint8)
    if data.shape[0] != count:
        raise ValueError(f"label count mismatch in {path}")
    return data.astype(int)


def load_mnist_arrays(directory):
    directory = os.fspath(directory)
    arrays = {
        "x_train": read_idx_images(_resolve(directory, FILES["x_train"])),
        "y_train": read_idx_labels(_resolve(directory, FILES["y_train"])),
        "x_test": read_idx_images(_resolve(directory, FILES["x_test"])),
        "y_test": read_idx_labels(_resolve(directory, FILES["y_test"])),
    }
    if arrays["x_train"].shape[0] != arrays["y_train"].shape[0]:
        raise ValueError("training images and labels have different lengths")
    if arrays["x_test"].shape[0] != arrays["y_test"].shape[0]:
        raise ValueError("test images and labels have different lengths")
    return arrays
