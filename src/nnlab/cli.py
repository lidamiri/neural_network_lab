import argparse
import json
import sys

import numpy as np

from .config import load_config
from .datasets import available_datasets, load_dataset
from .experiment import run_benchmark, run_config_file
from .gradcheck import run_gradcheck
from .metrics import predictions_to_labels
from .serialization import load_model
from .trainer import evaluate
from .utils import write_json

CLASSIFICATION_LOSSES = {"binary_crossentropy", "categorical_crossentropy"}


def build_parser():
    parser = argparse.ArgumentParser(prog="nnlab", description="Feedforward neural network toolkit")
    commands = parser.add_subparsers(dest="command", required=True)

    train = commands.add_parser("train", help="train a model from a configuration file")
    train.add_argument("--config", required=True)
    train.add_argument("--output", default="runs/latest")
    train.add_argument("--quiet", action="store_true")
    train.set_defaults(handler=cmd_train)

    evaluate_parser = commands.add_parser("evaluate", help="evaluate a saved model on a dataset split")
    evaluate_parser.add_argument("--model", required=True)
    evaluate_parser.add_argument("--config", required=True)
    evaluate_parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    evaluate_parser.add_argument("--output")
    evaluate_parser.set_defaults(handler=cmd_evaluate)

    predict = commands.add_parser("predict", help="run a saved model on a CSV file of feature rows")
    predict.add_argument("--model", required=True)
    predict.add_argument("--input", required=True)
    predict.add_argument("--output")
    predict.add_argument("--delimiter", default=",")
    predict.set_defaults(handler=cmd_predict)

    check = commands.add_parser("gradcheck", help="compare analytic and numerical gradients")
    check.add_argument("--seed", type=int, default=0)
    check.add_argument("--tolerance", type=float, default=1e-5)
    check.set_defaults(handler=cmd_gradcheck)

    benchmark = commands.add_parser("benchmark", help="train every configuration in a directory")
    benchmark.add_argument("--config-dir", default="configs")
    benchmark.add_argument("--output", default="runs/benchmark")
    benchmark.set_defaults(handler=cmd_benchmark)

    datasets = commands.add_parser("datasets", help="list the built-in datasets")
    datasets.set_defaults(handler=cmd_datasets)

    return parser


def cmd_train(args):
    summary = run_config_file(args.config, args.output, verbose=not args.quiet)
    print(f"saved run to {args.output}")
    print(f"test {summary['primary_metric']}: {summary['primary_value']:.4f}")
    return 0


def cmd_evaluate(args):
    model = load_model(args.model)
    config = load_config(args.config)
    dataset = load_dataset(config.dataset["name"], config.dataset.get("options", {}), seed=config.seed)
    report = evaluate(model, dataset, split=args.split)
    if args.output:
        write_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True, default=float))
    return 0


def cmd_predict(args):
    model = load_model(args.model)
    features = np.loadtxt(args.input, delimiter=args.delimiter, ndmin=2)
    outputs = model.predict(features)
    rows = []
    if model.loss.name in CLASSIFICATION_LOSSES:
        labels = predictions_to_labels(outputs)
        for index, label in enumerate(labels):
            rows.append({"index": index, "label": int(label), "scores": outputs[index].tolist()})
    else:
        for index in range(outputs.shape[0]):
            rows.append({"index": index, "values": outputs[index].tolist()})
    if args.output:
        write_json(args.output, rows)
    for row in rows:
        print(json.dumps(row, sort_keys=True))
    return 0


def cmd_gradcheck(args):
    results = run_gradcheck(seed=args.seed, tolerance=args.tolerance)
    failed = [result for result in results if not result["passed"]]
    for result in results:
        status = "PASS" if result["passed"] else "FAIL"
        print(f"{status}  {result['name']:<28} max relative error {result['max_relative_error']:.3e}")
    if failed:
        print(f"{len(failed)} case(s) failed")
        return 1
    print("all gradient checks passed")
    return 0


def cmd_benchmark(args):
    rows = run_benchmark(args.config_dir, args.output)
    print(f"benchmarked {len(rows)} configuration(s), report written to {args.output}")
    return 0


def cmd_datasets(args):
    for name in available_datasets():
        print(name)
    return 0


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.handler(args)
    except (ValueError, FileNotFoundError, KeyError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
