import json
import logging
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Protocol

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrainingState:
    num_samples: int
    optimizer_steps: int = 0
    data_position_known: bool = True
    grad_scaler_state: dict | None = None


class SnapshotWriter(Protocol):
    def __call__(
        self,
        num_samples: int | None,
        *,
        resume_num_samples: int | None = None,
        optimizer_steps: int = 0,
        cursor_is_exact: bool = True,
        grad_scaler_state: dict | None = None,
    ) -> None: ...


def save_training_state(
    prefix: Path, infix: str, state: TrainingState
) -> None:
    """Write the data cursor after the corresponding model and optimizer files."""
    if state.num_samples < 0 or state.optimizer_steps < 0:
        raise ValueError(state)
    path = prefix / f"training-state{infix}.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(
            {
                "version": 1,
                "num_samples": state.num_samples,
                "optimizer_steps": state.optimizer_steps,
                "data_position_known": state.data_position_known,
                "grad_scaler_state": state.grad_scaler_state,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def load_training_state(
    prefix: Path,
    infix: str,
    fallback_samples: int,
    *,
    restart_data: bool = False,
    start_new_epoch: bool = False,
) -> TrainingState:
    path = prefix / f"training-state{infix}.json"
    if restart_data:
        logger.warning(
            "Restarting data, optimizer history and AMP scaling from the beginning."
        )
        return TrainingState(0)
    if not path.exists():
        logger.warning(
            "%s: No saved training cursor. Using %d samples from the legacy "
            "snapshot name; optimizer update history is unknown.",
            path,
            fallback_samples,
        )
        return TrainingState(0 if start_new_epoch else fallback_samples)
    try:
        contents = json.loads(path.read_text(encoding="utf-8"))
        if contents["version"] != 1:
            raise ValueError("Unknown training-state version.")
        for field in ("num_samples", "optimizer_steps"):
            if type(contents[field]) is not int or contents[field] < 0:
                raise ValueError(f"Invalid {field}.")
        position_known = contents.get("data_position_known", True)
        if type(position_known) is not bool:
            raise ValueError("Invalid data_position_known.")
        scaler_state = contents.get("grad_scaler_state")
        if scaler_state is not None:
            if not isinstance(scaler_state, dict):
                raise ValueError("Invalid AMP scaler state.")
            for key in ("scale", "growth_factor", "backoff_factor"):
                if (
                    not math.isfinite(scaler_state[key])
                    or scaler_state[key] <= 0
                ):
                    raise ValueError("Invalid AMP scaler state.")
            for key in ("growth_interval", "_growth_tracker"):
                if type(scaler_state[key]) is not int or scaler_state[key] < 0:
                    raise ValueError("Invalid AMP scaler state.")
            if scaler_state["growth_interval"] == 0:
                raise ValueError("Invalid AMP growth interval.")
        state = TrainingState(
            contents["num_samples"],
            contents["optimizer_steps"],
            position_known,
            scaler_state,
        )
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError(f"{path}: Invalid training state.") from error
    if start_new_epoch:
        logger.info(
            "Starting a new data epoch after %d optimizer steps.",
            state.optimizer_steps,
        )
        return replace(state, num_samples=0, data_position_known=True)
    if not state.data_position_known and state.num_samples > 0:
        raise RuntimeError(
            f"{path}: The checkpoint has no exact input position (replay buffer "
            "or unfinished multi-worker accumulation). Use start_new_epoch=true "
            "to retain optimizer history while rereading data, or optimizer.initialize=true "
            "to reset optimizer history as well."
        )
    logger.info(
        "Resuming at sample %d after %d optimizer steps.",
        state.num_samples,
        state.optimizer_steps,
    )
    return state
