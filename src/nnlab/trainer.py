import time
from dataclasses import dataclass, field

import numpy as np

from .metrics import classification_metrics, predictions_to_labels, regression_metrics
from .preprocessing import iterate_minibatches
from .schedules import Constant
from .utils import format_seconds, make_rng


@dataclass
class TrainingResult:
    history: list = field(default_factory=list)
    best_epoch: int = 0
    best_val_loss: float = float("inf")
    stopped_early: bool = False
    epochs_run: int = 0


class EarlyStopping:
    def __init__(self, patience=None, min_delta=0.0):
        self.patience = patience
        self.min_delta = float(min_delta)
        self.best = float("inf")
        self.wait = 0

    def update(self, value):
        if value < self.best - self.min_delta:
            self.best = value
            self.wait = 0
            return True
        self.wait += 1
        return False

    def should_stop(self):
        return self.patience is not None and self.wait >= self.patience


class Trainer:
    def __init__(
        self,
        model,
        optimizer,
        schedule=None,
        epochs=50,
        batch_size=32,
        shuffle=True,
        patience=None,
        min_delta=1e-5,
        restore_best=True,
        verbose=True,
        seed=0,
        logger=print,
    ):
        if int(epochs) < 1:
            raise ValueError("epochs must be at least 1")
        if int(batch_size) < 1:
            raise ValueError("batch_size must be at least 1")
        self.model = model
        self.optimizer = optimizer
        self.schedule = schedule if schedule is not None else Constant(optimizer.learning_rate)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.early_stopping = EarlyStopping(patience=patience, min_delta=min_delta)
        self.restore_best = bool(restore_best)
        self.verbose = bool(verbose)
        self.logger = logger
        self.rng = make_rng(seed)

    def _log(self, message):
        if self.verbose and self.logger is not None:
            self.logger(message)

    def _validation_score(self, dataset):
        if dataset.task == "classification":
            return "val_accuracy", self._accuracy(dataset, "val")
        return "val_r2", self._r2(dataset, "val")

    def _accuracy(self, dataset, split):
        outputs = self.model.predict(getattr(dataset, f"x_{split}"), batch_size=self.batch_size)
        labels = predictions_to_labels(outputs)
        truth = getattr(dataset, f"labels_{split}")
        return float(np.mean(labels == truth)) if len(truth) else 0.0

    def _r2(self, dataset, split):
        outputs = self.model.predict(getattr(dataset, f"x_{split}"), batch_size=self.batch_size)
        truth = getattr(dataset, f"y_{split}")
        return regression_metrics(truth, outputs)["r2"] if len(truth) else 0.0

    def fit(self, dataset):
        result = TrainingResult()
        best_state = None
        for epoch in range(self.epochs):
            learning_rate = self.schedule.value(epoch, self.epochs)
            self.optimizer.learning_rate = float(learning_rate)
            started = time.time()
            total_loss = 0.0
            seen = 0
            for batch_x, batch_y in iterate_minibatches(
                dataset.x_train, dataset.y_train, self.batch_size, rng=self.rng, shuffle=self.shuffle
            ):
                batch_loss = self.model.train_on_batch(batch_x, batch_y, self.optimizer)
                total_loss += batch_loss * len(batch_x)
                seen += len(batch_x)
            train_loss = total_loss / max(seen, 1)
            if len(dataset.x_val):
                val_loss = self.model.evaluate_loss(dataset.x_val, dataset.y_val, batch_size=self.batch_size)
            else:
                val_loss = train_loss
            score_name, score = self._validation_score(dataset) if len(dataset.x_val) else (None, None)
            elapsed = time.time() - started
            record = {
                "epoch": epoch + 1,
                "learning_rate": float(learning_rate),
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
                "seconds": float(elapsed),
            }
            if score_name is not None:
                record[score_name] = float(score)
            result.history.append(record)
            result.epochs_run = epoch + 1
            detail = f"{score_name}={score:.4f} " if score_name is not None else ""
            self._log(
                f"epoch {epoch + 1:>4}/{self.epochs}  lr={learning_rate:.5f}  "
                f"train_loss={train_loss:.5f}  val_loss={val_loss:.5f}  {detail}({format_seconds(elapsed)})"
            )
            improved = self.early_stopping.update(val_loss)
            if improved:
                best_state = self.model.state_dict()
                result.best_epoch = epoch + 1
                result.best_val_loss = float(val_loss)
            if self.early_stopping.should_stop():
                result.stopped_early = True
                self._log(f"early stopping at epoch {epoch + 1}, best epoch {result.best_epoch}")
                break
        if self.restore_best and best_state is not None:
            self.model.load_state_dict(best_state)
        return result


def evaluate(model, dataset, split="test", batch_size=512):
    x = getattr(dataset, f"x_{split}")
    y = getattr(dataset, f"y_{split}")
    outputs = model.predict(x, batch_size=batch_size)
    report = {"loss": float(model.loss.forward(outputs, y) + model.regularizer.penalty(model.layers))}
    if dataset.task == "classification":
        labels = getattr(dataset, f"labels_{split}")
        scores = outputs if dataset.num_classes > 2 else None
        report.update(classification_metrics(labels, predictions_to_labels(outputs), dataset.num_classes, scores))
    else:
        truth = y
        prediction = outputs
        if dataset.target_scaler is not None:
            truth = dataset.target_scaler.inverse_transform(y)
            prediction = dataset.target_scaler.inverse_transform(outputs)
        report.update(regression_metrics(truth, prediction))
    return report
