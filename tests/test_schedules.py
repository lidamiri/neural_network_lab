import math

import pytest

from nnlab.schedules import (
    Constant,
    CosineAnnealing,
    ExponentialDecay,
    StepDecay,
    available_schedules,
    get_schedule,
)


def test_constant_schedule_never_changes():
    schedule = Constant(0.1)
    assert [schedule.value(epoch, 10) for epoch in range(10)] == [0.1] * 10


def test_step_decay_drops_every_n_epochs():
    schedule = StepDecay(1.0, drop=0.5, every=3)
    assert schedule.value(0, 10) == pytest.approx(1.0)
    assert schedule.value(2, 10) == pytest.approx(1.0)
    assert schedule.value(3, 10) == pytest.approx(0.5)
    assert schedule.value(7, 10) == pytest.approx(0.25)


def test_exponential_decay_is_geometric():
    schedule = ExponentialDecay(0.2, decay_rate=0.9)
    assert schedule.value(0, 5) == pytest.approx(0.2)
    assert schedule.value(2, 5) == pytest.approx(0.2 * 0.81)


def test_cosine_annealing_hits_both_ends():
    schedule = CosineAnnealing(0.1, min_lr=0.001)
    assert schedule.value(0, 11) == pytest.approx(0.1)
    assert schedule.value(10, 11) == pytest.approx(0.001)
    middle = schedule.value(5, 11)
    assert middle == pytest.approx(0.001 + 0.5 * 0.099 * (1 + math.cos(math.pi * 0.5)))


def test_cosine_annealing_is_monotonic():
    schedule = CosineAnnealing(0.5)
    values = [schedule.value(epoch, 20) for epoch in range(20)]
    assert all(later <= earlier + 1e-12 for earlier, later in zip(values, values[1:]))


def test_registry_and_validation():
    assert set(available_schedules()) == {"constant", "step", "exponential", "cosine"}
    assert isinstance(get_schedule("cosine", 0.1), CosineAnnealing)
    assert isinstance(get_schedule(None, 0.1), Constant)
    with pytest.raises(ValueError):
        get_schedule("linear", 0.1)
    with pytest.raises(ValueError):
        StepDecay(0.1, drop=1.5)
    with pytest.raises(ValueError):
        CosineAnnealing(0.1, min_lr=0.5)
