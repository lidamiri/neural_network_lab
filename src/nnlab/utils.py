import json
import os
import pathlib
import random

import numpy as np


def make_rng(seed=None):
    return np.random.default_rng(seed)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    return make_rng(seed)


def ensure_dir(path):
    directory = pathlib.Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def to_jsonable(value):
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, pathlib.PurePath):
        return str(value)
    return value


def write_json(path, data):
    target = pathlib.Path(path)
    ensure_dir(target.parent)
    with open(target, "w", encoding="utf-8") as handle:
        json.dump(to_jsonable(data), handle, indent=2, sort_keys=True)
        handle.write(os.linesep)
    return target


def read_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def format_seconds(seconds):
    seconds = float(seconds)
    if seconds < 60.0:
        return f"{seconds:.1f}s"
    minutes, remainder = divmod(seconds, 60.0)
    if minutes < 60.0:
        return f"{int(minutes)}m{remainder:04.1f}s"
    hours, minutes = divmod(minutes, 60.0)
    return f"{int(hours)}h{int(minutes):02d}m"


def chunked(items, size):
    for start in range(0, len(items), size):
        yield items[start:start + size]
