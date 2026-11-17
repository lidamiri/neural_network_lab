import json
import pathlib

import numpy as np

from nnlab.cli import main
from nnlab.serialization import save_model
from nnlab.builder import build_model


ROOT = pathlib.Path(__file__).resolve().parent.parent


def write_config(path):
    payload = {
        "name": "cli-xor",
        "seed": 1,
        "dataset": {"name": "xor", "options": {"samples": 160, "target": "binary"}},
        "architecture": [
            {"type": "dense", "units": 4, "activation": "tanh"},
            {"type": "dense", "units": 1, "activation": "sigmoid"},
        ],
        "loss": "binary_crossentropy",
        "optimizer": {"name": "adam", "learning_rate": 0.05},
        "training": {"epochs": 3, "batch_size": 16, "verbose": False},
    }
    path.write_text(json.dumps(payload))
    return path


def test_datasets_command_lists_builtins(capsys):
    assert main(["datasets"]) == 0
    output = capsys.readouterr().out
    assert "xor" in output and "spirals" in output


def test_gradcheck_command_succeeds(capsys):
    assert main(["gradcheck"]) == 0
    assert "all gradient checks passed" in capsys.readouterr().out


def test_train_then_evaluate_and_predict(tmp_path, capsys):
    config_path = write_config(tmp_path / "config.json")
    output = tmp_path / "run"
    assert main(["train", "--config", str(config_path), "--output", str(output), "--quiet"]) == 0
    assert (output / "model.npz").exists()

    report_path = tmp_path / "report.json"
    assert main(["evaluate", "--model", str(output / "model.npz"), "--config", str(config_path), "--output", str(report_path)]) == 0
    report = json.loads(report_path.read_text())
    assert "accuracy" in report

    features = tmp_path / "features.csv"
    np.savetxt(features, np.array([[0.5, 0.5], [-0.5, 0.5]]), delimiter=",")
    predictions = tmp_path / "predictions.json"
    assert main(["predict", "--model", str(output / "model.npz"), "--input", str(features), "--output", str(predictions)]) == 0
    rows = json.loads(predictions.read_text())
    assert len(rows) == 2
    assert rows[0]["label"] in (0, 1)


def test_predict_reports_regression_values(tmp_path):
    model = build_model([{"type": "dense", "units": 1, "activation": "linear"}], 2, loss="mse", seed=0)
    model_path = save_model(model, tmp_path / "reg")
    features = tmp_path / "features.csv"
    np.savetxt(features, np.array([[1.0, 2.0]]), delimiter=",")
    output = tmp_path / "out.json"
    assert main(["predict", "--model", model_path, "--input", str(features), "--output", str(output)]) == 0
    rows = json.loads(output.read_text())
    assert "values" in rows[0]


def test_missing_configuration_returns_error_code(tmp_path):
    assert main(["train", "--config", str(tmp_path / "absent.json"), "--output", str(tmp_path / "x")]) == 2


def test_shipped_configurations_are_valid():
    from nnlab.config import load_config

    paths = sorted((ROOT / "configs").glob("*.json"))
    assert len(paths) >= 5
    for path in paths:
        config = load_config(path)
        assert config.architecture
