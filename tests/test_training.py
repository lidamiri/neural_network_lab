import json

import numpy as np
import pytest

from nnlab.builder import build_model
from nnlab.config import ExperimentConfig, from_dict, load_config, save_config
from nnlab.datasets import load_dataset
from nnlab.experiment import run_benchmark, run_experiment
from nnlab.optimizers import Adam
from nnlab.schedules import CosineAnnealing
from nnlab.serialization import load_model, save_model
from nnlab.trainer import EarlyStopping, Trainer, evaluate


def silent_trainer(model, optimizer, **kwargs):
    options = {"epochs": 5, "batch_size": 16, "verbose": False, "seed": 0}
    options.update(kwargs)
    return Trainer(model, optimizer, **options)


def test_trainer_learns_xor():
    dataset = load_dataset("xor", {"samples": 400, "target": "binary"}, seed=3)
    model = build_model(
        [{"type": "dense", "units": 8, "activation": "tanh"}, {"type": "dense", "units": 1, "activation": "sigmoid"}],
        dataset.input_dim,
        loss="binary_crossentropy",
        seed=3,
    )
    trainer = silent_trainer(model, Adam(learning_rate=0.05), epochs=120, batch_size=32)
    result = trainer.fit(dataset)
    report = evaluate(model, dataset, "test")
    assert result.epochs_run == 120
    assert report["accuracy"] > 0.9
    assert len(result.history) == 120
    assert "val_accuracy" in result.history[0]


def test_regression_history_reports_r2():
    dataset = load_dataset("regression", {"samples": 300}, seed=4)
    model = build_model([{"type": "dense", "units": 16, "activation": "tanh"}, {"type": "dense", "units": 1}], 1, loss="mse", seed=4)
    result = silent_trainer(model, Adam(learning_rate=0.01), epochs=10).fit(dataset)
    assert "val_r2" in result.history[-1]


def test_early_stopping_halts_training_and_restores_best():
    dataset = load_dataset("moons", {"samples": 200}, seed=6)
    model = build_model([{"type": "dense", "units": 4, "activation": "relu"}, {"type": "dense", "units": 2, "activation": "softmax"}], 2, loss="cce", seed=6)
    trainer = silent_trainer(model, Adam(learning_rate=0.0001), epochs=500, patience=3, min_delta=1e-2)
    result = trainer.fit(dataset)
    assert result.stopped_early
    assert result.epochs_run < 500
    assert result.best_epoch >= 1
    final_loss = model.total_loss(dataset.x_val, dataset.y_val)
    assert final_loss == pytest.approx(result.best_val_loss, rel=1e-6)


def test_schedule_updates_optimizer_learning_rate():
    dataset = load_dataset("moons", {"samples": 120}, seed=7)
    model = build_model([{"type": "dense", "units": 4, "activation": "tanh"}, {"type": "dense", "units": 2, "activation": "softmax"}], 2, loss="cce", seed=7)
    optimizer = Adam(learning_rate=0.1)
    trainer = silent_trainer(model, optimizer, epochs=4, schedule=CosineAnnealing(0.1, min_lr=0.01))
    result = trainer.fit(dataset)
    rates = [record["learning_rate"] for record in result.history]
    assert rates[0] == pytest.approx(0.1)
    assert rates[-1] == pytest.approx(0.01)


def test_early_stopping_counter():
    stopper = EarlyStopping(patience=2, min_delta=0.1)
    assert stopper.update(1.0)
    assert not stopper.update(0.95)
    assert not stopper.update(0.94)
    assert stopper.should_stop()


def test_trainer_validates_arguments():
    model = build_model([{"type": "dense", "units": 1}], 1, loss="mse", seed=0)
    with pytest.raises(ValueError):
        Trainer(model, Adam(), epochs=0)
    with pytest.raises(ValueError):
        Trainer(model, Adam(), batch_size=0)


def test_model_round_trip_through_archive(tmp_path, rng):
    model = build_model(
        [{"type": "dense", "units": 5, "activation": "relu"}, {"type": "batchnorm"}, {"type": "dense", "units": 2, "activation": "softmax"}],
        3,
        loss="cce",
        l2=0.01,
        seed=8,
    )
    x = rng.normal(size=(9, 3))
    path = save_model(model, tmp_path / "model")
    restored = load_model(path)
    np.testing.assert_allclose(model.predict(x), restored.predict(x))
    assert restored.loss.name == "categorical_crossentropy"
    assert restored.regularizer.coefficient == pytest.approx(0.01)


def test_load_model_rejects_foreign_archives(tmp_path):
    path = tmp_path / "other.npz"
    np.savez(path, weights=np.zeros(3))
    with pytest.raises(ValueError):
        load_model(path)


def test_config_defaults_and_validation():
    config = from_dict({"architecture": [{"type": "dense", "units": 2}]})
    assert config.optimizer["name"] == "adam"
    assert config.training_options()["epochs"] == 50
    with pytest.raises(ValueError):
        from_dict({"architecture": []})
    with pytest.raises(ValueError):
        from_dict({"architecture": [{"type": "dense", "units": 2}], "colour": "blue"})
    with pytest.raises(ValueError):
        from_dict({"architecture": [{"type": "dense", "units": 2}], "training": {"epochs_total": 3}})


def test_config_round_trip(tmp_path):
    config = ExperimentConfig(name="demo", architecture=[{"type": "dense", "units": 2}], dataset={"name": "xor"})
    path = save_config(config, tmp_path / "demo.json")
    loaded = load_config(path)
    assert loaded.name == "demo"
    assert loaded.architecture == config.architecture


def test_run_experiment_writes_artifacts(tmp_path):
    config = from_dict(
        {
            "name": "tiny",
            "seed": 2,
            "dataset": {"name": "xor", "options": {"samples": 200, "target": "binary"}},
            "architecture": [
                {"type": "dense", "units": 4, "activation": "tanh"},
                {"type": "dense", "units": 1, "activation": "sigmoid"},
            ],
            "loss": "binary_crossentropy",
            "optimizer": {"name": "adam", "learning_rate": 0.05},
            "training": {"epochs": 5, "batch_size": 16, "verbose": False},
        }
    )
    summary = run_experiment(config, tmp_path / "run", verbose=False)
    for name in ["model.npz", "history.json", "summary.json", "metrics.txt"]:
        assert (tmp_path / "run" / name).exists()
    stored = json.loads((tmp_path / "run" / "summary.json").read_text())
    assert stored["epochs_run"] == 5
    assert summary["primary_metric"] == "accuracy"


def test_benchmark_runs_every_configuration(tmp_path):
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    base = {
        "dataset": {"name": "xor", "options": {"samples": 120, "target": "binary"}},
        "architecture": [{"type": "dense", "units": 2, "activation": "tanh"}, {"type": "dense", "units": 1, "activation": "sigmoid"}],
        "loss": "binary_crossentropy",
        "training": {"epochs": 2, "batch_size": 16, "verbose": False},
    }
    for name in ["alpha", "beta"]:
        payload = dict(base, name=name)
        (config_dir / f"{name}.json").write_text(json.dumps(payload))
    rows = run_benchmark(config_dir, tmp_path / "bench", logger=lambda message: None)
    assert [row["name"] for row in rows] == ["alpha", "beta"]
    assert (tmp_path / "bench" / "benchmark.txt").exists()
