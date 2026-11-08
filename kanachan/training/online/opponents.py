"""Immutable, hash-addressed fixed and promoted opponents."""

from pathlib import Path
import random
import shutil

from .state import atomic_save, file_digest


class OpponentPool:
    def __init__(self, root, max_history, state=None):
        self.root = Path(root)
        (self.root / "opponents").mkdir(parents=True, exist_ok=True)
        self.max_history = max_history
        self.fixed = [] if state is None else list(state["fixed"])
        self.history = [] if state is None else list(state["history"])
        self.registry = (
            []
            if state is None
            else list(state.get("registry", self.fixed + self.history))
        )
        for entry in self.fixed + self.history:
            self.path(entry)

    def path(self, entry):
        identity = entry["id"]
        if len(identity) != 64 or any(
            ch not in "0123456789abcdef" for ch in identity
        ):
            raise ValueError("Invalid opponent identity.")
        path = self.root / "opponents" / f"{identity}.kanachan"
        if not path.is_file() or file_digest(path) != identity:
            raise ValueError(f"Missing or modified opponent: {path}")
        return path

    def add_fixed(self, path):
        identity = file_digest(path)
        entry = {"id": identity, "version": 0}
        destination = self.root / "opponents" / f"{identity}.kanachan"
        if not destination.exists():
            shutil.copyfile(path, destination)
        self.path(entry)
        if entry not in self.fixed:
            self.fixed.append(entry)
        if entry not in self.registry:
            self.registry.append(entry)

    def promote(self, model_state, version):
        temporary = self.root / "opponents" / "candidate.kanachan"
        atomic_save(temporary, model_state)
        identity = file_digest(temporary)
        destination = temporary.with_name(f"{identity}.kanachan")
        if destination.exists():
            temporary.unlink()
        else:
            temporary.replace(destination)
        entry = {"id": identity, "version": version}
        if entry not in self.registry:
            self.registry.append(entry)
        if self.max_history and all(
            item["id"] != identity for item in self.fixed + self.history
        ):
            self.history.append(entry)
            self.history = self.history[-self.max_history :]
        return entry

    def choose(self):
        if not self.fixed:
            raise ValueError("An online run needs a fixed baseline.")
        return random.choice(self.fixed + self.history)

    def state_dict(self):
        return {
            "fixed": self.fixed.copy(),
            "history": self.history.copy(),
            "registry": self.registry.copy(),
        }
