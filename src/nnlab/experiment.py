import pathlib
import time

from .builder import build_model
from .config import load_config
from .datasets import load_dataset
from .optimizers import get_optimizer
from .reporting import benchmark_table, classification_report, regression_report
from .schedules import get_schedule
from .serialization import save_model
from .trainer import Trainer, evaluate
from .utils import ensure_dir, write_json


def _write_text(path, text):
    target = pathlib.Path(path)
    ensure_dir(target.parent)
    target.write_text(text + "\n", encoding="utf-8")
    return target


def _primary_metric(report):
    if "accuracy" in report:
        return "accuracy", report["accuracy"]
    return "r2", report["r2"]


def run_experiment(config, output_dir, verbose=True, logger=print):
    started = time.time()
    dataset = load_dataset(config.dataset["name"], config.dataset.get("options", {}), seed=config.seed)
    model = build_model(config.architecture, dataset.input_dim, loss=config.loss, l2=config.l2, seed=config.seed)
    optimizer = get_optimizer(config.optimizer)
    schedule = get_schedule(config.schedule, optimizer.learning_rate)
    options = config.training_options()
    options["verbose"] = bool(verbose and options["verbose"])
    trainer = Trainer(model, optimizer, schedule=schedule, seed=config.seed, logger=logger, **options)
    if options["verbose"]:
        logger(f"dataset {dataset.describe()}")
        logger(model.summary())
    result = trainer.fit(dataset)
    validation = evaluate(model, dataset, "val")
    test = evaluate(model, dataset, "test")
    metric_name, metric_value = _primary_metric(test)
    output = ensure_dir(output_dir)
    save_model(model, output / "model.npz")
    write_json(output / "history.json", result.history)
    if dataset.task == "classification":
        names = [str(index) for index in range(dataset.num_classes)]
        text = classification_report(test, names)
    else:
        text = regression_report(test)
    _write_text(output / "metrics.txt", text)
    summary = {
        "name": config.name,
        "dataset": dataset.describe(),
        "config": config.to_dict(),
        "epochs_run": result.epochs_run,
        "best_epoch": result.best_epoch,
        "stopped_early": result.stopped_early,
        "parameters": model.num_parameters(),
        "validation": validation,
        "test": test,
        "primary_metric": metric_name,
        "primary_value": metric_value,
        "seconds": time.time() - started,
    }
    write_json(output / "summary.json", summary)
    if verbose:
        logger(f"test {metric_name}: {metric_value:.4f}")
    return summary


def run_config_file(path, output_dir, verbose=True, logger=print):
    config = load_config(path)
    return run_experiment(config, output_dir, verbose=verbose, logger=logger)


def run_benchmark(config_dir, output_dir, logger=print):
    directory = pathlib.Path(config_dir)
    rows = []
    for path in sorted(directory.glob("*.json")):
        config = load_config(path)
        summary = run_experiment(config, pathlib.Path(output_dir) / config.name, verbose=False, logger=logger)
        rows.append(
            {
                "name": config.name,
                "dataset": config.dataset["name"],
                "parameters": summary["parameters"],
                "epochs": summary["epochs_run"],
                "test_loss": summary["test"]["loss"],
                "metric": summary["primary_metric"],
                "value": summary["primary_value"],
            }
        )
        logger(f"{config.name}: {summary['primary_metric']}={summary['primary_value']:.4f}")
    output = ensure_dir(output_dir)
    write_json(output / "benchmark.json", rows)
    table = benchmark_table(rows)
    _write_text(output / "benchmark.txt", table)
    return rows
