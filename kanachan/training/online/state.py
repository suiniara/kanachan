"""Atomic, portable snapshots; no executable module objects in checkpoints."""

import copy
import hashlib
import os
from pathlib import Path
import random

import torch


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def cpu_state(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_state(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_state(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_state(item) for item in value)
    return value


def atomic_save(path, state):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        torch.save(state, stream)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def export_model(template, model):
    result = copy.deepcopy(template)

    # Constructors in older exports may request CUDA or half precision.
    def normalize(value):
        if isinstance(value, dict):
            if "__kanachan__" in value:
                kwargs = value.get("kwargs", {})
                if "device" in kwargs:
                    kwargs["device"] = torch.device("cpu")
                if "dtype" in kwargs:
                    kwargs["dtype"] = torch.float32
                value.pop("state_dict", None)
            for item in value.values():
                normalize(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                normalize(item)

    normalize(result)
    result["state_dict"] = cpu_state(model.state_dict())
    return result


def capture_rng():
    result = {
        "python": random.getstate(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all()
        if torch.cuda.is_available()
        else [],
    }
    # NumPy is optional; store its MT state as tensors/primitives so the
    # checkpoint remains loadable with weights_only=True.
    try:
        import numpy as np
    except ImportError:
        pass
    else:
        state = np.random.get_state()
        result["numpy"] = (
            state[0], torch.tensor(state[1].astype("int64")),
            state[2], state[3], state[4],
        )
    return result


def restore_rng(state):
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"].cpu())
    if "numpy" in state:
        import numpy as np
        name, keys, position, gaussian, cached = state["numpy"]
        np.random.set_state((
            name, keys.cpu().numpy().astype("uint32"), position, gaussian, cached
        ))
    if state["cuda"]:
        if (
            not torch.cuda.is_available()
            or len(state["cuda"]) != torch.cuda.device_count()
        ):
            raise ValueError(
                "Exact RNG restoration needs the original CUDA device count."
            )
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])
