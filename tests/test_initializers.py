import numpy as np
import pytest

from nnlab.initializers import (
    available_initializers,
    get_initializer,
    he_normal,
    he_uniform,
    lecun_normal,
    xavier_normal,
    xavier_uniform,
    zeros,
)


def test_zeros_and_ones_are_constant(rng):
    assert np.all(zeros(rng, (3, 4)) == 0.0)
    assert get_initializer("ones")(rng, (2, 2)).sum() == pytest.approx(4.0)


def test_xavier_uniform_respects_bounds(rng):
    weights = xavier_uniform(rng, (200, 300))
    limit = np.sqrt(6.0 / (200 + 300))
    assert weights.shape == (200, 300)
    assert np.all(np.abs(weights) <= limit + 1e-12)


def test_he_normal_has_expected_variance(rng):
    weights = he_normal(rng, (500, 400))
    assert weights.std() == pytest.approx(np.sqrt(2.0 / 500), rel=0.05)


def test_lecun_normal_has_expected_variance(rng):
    weights = lecun_normal(rng, (500, 400))
    assert weights.std() == pytest.approx(np.sqrt(1.0 / 500), rel=0.05)


def test_xavier_normal_has_expected_variance(rng):
    weights = xavier_normal(rng, (400, 400))
    assert weights.std() == pytest.approx(np.sqrt(2.0 / 800), rel=0.05)


def test_he_uniform_respects_bounds(rng):
    weights = he_uniform(rng, (128, 64))
    assert np.all(np.abs(weights) <= np.sqrt(6.0 / 128) + 1e-12)


def test_one_dimensional_shapes_are_rejected(rng):
    with pytest.raises(ValueError):
        he_normal(rng, (10,))


def test_registry_lookup_and_callables():
    assert "he_normal" in available_initializers()
    assert get_initializer("xavier_uniform") is xavier_uniform
    assert get_initializer(he_normal) is he_normal
    with pytest.raises(ValueError):
        get_initializer("orthogonal")
