import gzip
import struct

import numpy as np
import pytest

from nnlab.datasets import available_datasets, load_dataset
from nnlab.mnist import load_mnist_arrays, read_idx_images, read_idx_labels
from nnlab.preprocessing import (
    MinMaxScaler,
    Standardizer,
    iterate_minibatches,
    one_hot,
    prepare_targets,
    train_val_test_split,
)


def write_idx_images(path, images, compress=False):
    count, rows, cols = images.shape
    payload = struct.pack(">IIII", 2051, count, rows, cols) + images.astype(np.uint8).tobytes()
    opener = gzip.open if compress else open
    with opener(path, "wb") as handle:
        handle.write(payload)


def write_idx_labels(path, labels):
    payload = struct.pack(">II", 2049, len(labels)) + np.asarray(labels, dtype=np.uint8).tobytes()
    with open(path, "wb") as handle:
        handle.write(payload)


def test_standardizer_zero_mean_unit_variance(rng):
    x = rng.normal(10.0, 4.0, size=(500, 3))
    scaled = Standardizer().fit_transform(x)
    np.testing.assert_allclose(scaled.mean(axis=0), np.zeros(3), atol=1e-10)
    np.testing.assert_allclose(scaled.std(axis=0), np.ones(3), atol=1e-10)


def test_standardizer_handles_constant_columns():
    x = np.column_stack([np.ones(10), np.arange(10.0)])
    scaled = Standardizer().fit_transform(x)
    assert np.all(np.isfinite(scaled))


def test_standardizer_inverse_round_trip(rng):
    x = rng.normal(size=(20, 2))
    scaler = Standardizer().fit(x)
    np.testing.assert_allclose(scaler.inverse_transform(scaler.transform(x)), x)


def test_standardizer_requires_fit():
    with pytest.raises(RuntimeError):
        Standardizer().transform(np.zeros((2, 2)))


def test_minmax_scaler_maps_to_range():
    x = np.array([[0.0], [5.0], [10.0]])
    scaled = MinMaxScaler(feature_range=(-1.0, 1.0)).fit_transform(x)
    np.testing.assert_allclose(scaled.ravel(), [-1.0, 0.0, 1.0])


def test_one_hot_encoding():
    encoded = one_hot([0, 2, 1], 3)
    np.testing.assert_array_equal(encoded, np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]], dtype=float))
    with pytest.raises(ValueError):
        one_hot([3], 3)


def test_prepare_targets_modes():
    assert prepare_targets([1, 0], "binary", 2).shape == (2, 1)
    assert prepare_targets([1, 0], "one_hot", 2).shape == (2, 2)
    assert prepare_targets([0.5, 1.5], "regression", 0).shape == (2, 1)
    with pytest.raises(ValueError):
        prepare_targets([1], "ordinal", 2)


def test_stratified_split_preserves_class_proportions(rng):
    labels = np.repeat(np.arange(4), 100)
    x = rng.normal(size=(400, 2))
    (train_x, train_y), (val_x, val_y), (test_x, test_y) = train_val_test_split(
        x, labels, val_fraction=0.2, test_fraction=0.1, rng=rng, stratify=True
    )
    assert len(train_x) + len(val_x) + len(test_x) == 400
    assert len(test_x) == 40
    assert len(val_x) == 80
    for value in range(4):
        assert np.sum(val_y == value) == 20
        assert np.sum(test_y == value) == 10


def test_split_rejects_invalid_fractions(rng):
    with pytest.raises(ValueError):
        train_val_test_split(np.zeros((4, 1)), np.zeros(4), val_fraction=0.6, test_fraction=0.5, rng=rng)


def test_minibatches_cover_every_sample_once(rng):
    x = np.arange(23).reshape(-1, 1)
    y = np.arange(23)
    seen = []
    for batch_x, _ in iterate_minibatches(x, y, 5, rng=rng, shuffle=True):
        seen.extend(batch_x.ravel().tolist())
    assert sorted(seen) == list(range(23))


def test_minibatches_drop_last_partial_batch():
    x = np.zeros((10, 1))
    sizes = [len(batch) for batch, _ in iterate_minibatches(x, x, 4, shuffle=False, drop_last=True)]
    assert sizes == [4, 4]


@pytest.mark.parametrize("name", ["xor", "spirals", "moons", "digits", "regression", "sine_surface"])
def test_builtin_generators_produce_valid_splits(name):
    dataset = load_dataset(name, {}, seed=5)
    assert dataset.name == name
    assert dataset.input_dim >= 1
    assert len(dataset.x_train) > len(dataset.x_val)
    assert len(dataset.x_train) == len(dataset.y_train)
    assert np.all(np.isfinite(dataset.x_train))
    assert np.all(np.isfinite(dataset.y_train))


def test_generators_are_deterministic_for_a_seed():
    first = load_dataset("moons", {"samples": 100}, seed=9)
    second = load_dataset("moons", {"samples": 100}, seed=9)
    np.testing.assert_array_equal(first.x_train, second.x_train)
    third = load_dataset("moons", {"samples": 100}, seed=10)
    assert not np.array_equal(first.x_train, third.x_train)


def test_digits_dataset_has_ten_balanced_classes():
    dataset = load_dataset("digits", {"samples_per_class": 20}, seed=1)
    labels = np.concatenate([dataset.labels_train, dataset.labels_val, dataset.labels_test])
    counts = np.bincount(labels, minlength=10)
    assert counts.sum() == 200
    assert np.all(counts == 20)
    assert dataset.input_dim == 256


def test_binary_target_mode_produces_single_column():
    dataset = load_dataset("xor", {"target": "binary"}, seed=2)
    assert dataset.output_dim == 1
    assert dataset.num_classes == 2


def test_regression_dataset_keeps_scaler_for_inverse_transform():
    dataset = load_dataset("regression", {"samples": 200}, seed=2)
    assert dataset.task == "regression"
    assert dataset.target_scaler is not None
    assert dataset.num_classes == 0


def test_unknown_dataset_is_rejected():
    with pytest.raises(ValueError):
        load_dataset("imagenet", {}, seed=0)
    assert "mnist" in available_datasets()


def test_idx_reader_reads_plain_and_gzipped_files(tmp_path):
    images = np.arange(2 * 3 * 3, dtype=np.uint8).reshape(2, 3, 3)
    write_idx_images(tmp_path / "plain.idx", images)
    write_idx_images(tmp_path / "zipped.idx.gz", images, compress=True)
    plain = read_idx_images(tmp_path / "plain.idx")
    zipped = read_idx_images(tmp_path / "zipped.idx.gz")
    np.testing.assert_allclose(plain, zipped)
    assert plain.shape == (2, 9)
    assert plain.max() <= 1.0


def test_idx_label_reader_validates_magic(tmp_path):
    write_idx_labels(tmp_path / "labels", [3, 1, 4])
    np.testing.assert_array_equal(read_idx_labels(tmp_path / "labels"), [3, 1, 4])
    with open(tmp_path / "bad", "wb") as handle:
        handle.write(struct.pack(">II", 1234, 1) + b"\x00")
    with pytest.raises(ValueError):
        read_idx_labels(tmp_path / "bad")


def test_mnist_loader_reads_a_complete_directory(tmp_path):
    images = np.zeros((6, 2, 2), dtype=np.uint8)
    labels = [0, 1, 2, 3, 4, 5]
    write_idx_images(tmp_path / "train-images-idx3-ubyte", images)
    write_idx_labels(tmp_path / "train-labels-idx1-ubyte", labels)
    write_idx_images(tmp_path / "t10k-images-idx3-ubyte", images[:2])
    write_idx_labels(tmp_path / "t10k-labels-idx1-ubyte", labels[:2])
    arrays = load_mnist_arrays(tmp_path)
    assert arrays["x_train"].shape == (6, 4)
    assert arrays["y_test"].tolist() == [0, 1]


def test_mnist_loader_reports_missing_files(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_mnist_arrays(tmp_path)


def test_mnist_dataset_uses_directory_option(tmp_path):
    images = np.zeros((20, 2, 2), dtype=np.uint8)
    labels = np.arange(20) % 10
    write_idx_images(tmp_path / "train-images-idx3-ubyte", images)
    write_idx_labels(tmp_path / "train-labels-idx1-ubyte", labels)
    write_idx_images(tmp_path / "t10k-images-idx3-ubyte", images[:10])
    write_idx_labels(tmp_path / "t10k-labels-idx1-ubyte", labels[:10])
    dataset = load_dataset("mnist", {"directory": str(tmp_path), "val_fraction": 0.5}, seed=0)
    assert dataset.num_classes == 10
    assert dataset.input_dim == 4
    assert len(dataset.x_val) == 10
