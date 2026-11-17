from dataclasses import dataclass, field

from .utils import read_json, write_json

TRAINING_KEYS = {"epochs", "batch_size", "shuffle", "patience", "min_delta", "restore_best", "verbose"}
TOP_LEVEL_KEYS = {
    "name",
    "seed",
    "dataset",
    "architecture",
    "loss",
    "l2",
    "optimizer",
    "schedule",
    "training",
}


@dataclass
class ExperimentConfig:
    name: str = "experiment"
    seed: int = 0
    dataset: dict = field(default_factory=lambda: {"name": "xor", "options": {}})
    architecture: list = field(default_factory=list)
    loss: object = "mse"
    l2: float = 0.0
    optimizer: object = field(default_factory=lambda: {"name": "adam", "learning_rate": 0.001})
    schedule: object = field(default_factory=lambda: {"name": "constant"})
    training: dict = field(default_factory=dict)

    def validate(self):
        if not self.architecture:
            raise ValueError("configuration must define at least one layer in architecture")
        if float(self.l2) < 0.0:
            raise ValueError("l2 must be non-negative")
        unknown = set(self.training) - TRAINING_KEYS
        if unknown:
            raise ValueError(f"unknown training options: {sorted(unknown)}")
        if "name" not in self.dataset:
            raise ValueError("dataset section must define a name")
        return self

    def training_options(self):
        options = {
            "epochs": 50,
            "batch_size": 32,
            "shuffle": True,
            "patience": None,
            "min_delta": 1e-5,
            "restore_best": True,
            "verbose": True,
        }
        options.update(self.training)
        return options

    def to_dict(self):
        return {
            "name": self.name,
            "seed": self.seed,
            "dataset": self.dataset,
            "architecture": self.architecture,
            "loss": self.loss,
            "l2": self.l2,
            "optimizer": self.optimizer,
            "schedule": self.schedule,
            "training": self.training,
        }


def from_dict(data):
    unknown = set(data) - TOP_LEVEL_KEYS
    if unknown:
        raise ValueError(f"unknown configuration keys: {sorted(unknown)}")
    defaults = ExperimentConfig()
    config = ExperimentConfig(
        name=data.get("name", defaults.name),
        seed=int(data.get("seed", defaults.seed)),
        dataset=data.get("dataset", defaults.dataset),
        architecture=list(data.get("architecture", defaults.architecture)),
        loss=data.get("loss", defaults.loss),
        l2=float(data.get("l2", defaults.l2)),
        optimizer=data.get("optimizer", defaults.optimizer),
        schedule=data.get("schedule", defaults.schedule),
        training=dict(data.get("training", {})),
    )
    return config.validate()


def load_config(path):
    return from_dict(read_json(path))


def save_config(config, path):
    return write_json(path, config.to_dict())
