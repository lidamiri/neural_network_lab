import json
import os

import numpy as np

from .builder import build_model

FORMAT_VERSION = 1
META_KEY = "__meta__"


def save_model(model, path):
    arrays = dict(model.state_dict())
    meta = {"format": FORMAT_VERSION, "config": model.get_config()}
    arrays[META_KEY] = np.array(json.dumps(meta, sort_keys=True))
    target = os.fspath(path)
    if not target.endswith(".npz"):
        target = f"{target}.npz"
    np.savez_compressed(target, **arrays)
    return target


def load_model(path):
    with np.load(os.fspath(path), allow_pickle=False) as archive:
        if META_KEY not in archive.files:
            raise ValueError(f"{path} is not a nnlab model archive")
        meta = json.loads(str(archive[META_KEY]))
        if meta.get("format") != FORMAT_VERSION:
            raise ValueError(f"unsupported model format {meta.get('format')}")
        state = {name: archive[name] for name in archive.files if name != META_KEY}
    config = meta["config"]
    model = build_model(
        config["layers"],
        config["input_dim"],
        loss=config["loss"],
        l2=config["l2"],
    )
    model.load_state_dict(state)
    return model
