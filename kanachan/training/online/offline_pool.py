"""An immutable, bounded train-data pool, separate from the online FIFO."""

from pathlib import Path
import time

import torch

from kanachan.training.core.offline_rl import DataLoader
from kanachan.training.core.rl.rewards import PlacementReward
from kanachan.training.core.rl.discrete_cql import OBSERVATIONS
from .replay import ReplayBuffer
from .reservoir import TransitionReservoir
from .state import atomic_save, file_digest


POOL_POLICY = "full_stream_reservoir_r_v1"
_CHECKPOINT_SECONDS = 60.0


def is_pool_preparation(directory):
    """A failed initial pool build may be continued without overwriting a run."""
    names = {path.name for path in Path(directory).iterdir()}
    return "offline-pool.pt" in names and names <= {
        "offline-pool.pt", "offline-pool.pt.tmp",
    }


TRAINING_KEYS = (
    *OBSERVATIONS, "action",
    *(("next", name) for name in OBSERVATIONS),
    ("next", "reward"), ("next", "done"), ("next", "discount"),
)


def training_columns(data):
    return data.select(*TRAINING_KEYS).clone(False)


def prepare_offline_pool(config, resume_state=None):
    if config.offline_data is None:
        return None, None
    replay = ReplayBuffer(config.offline_capacity, config.seed ^ 0x05C01)
    if resume_state is not None:
        saved = resume_state.get("offline_replay")
        if saved is None:
            raise ValueError("Offline pool missing from the online checkpoint.")
        replay.load_state_dict(saved)
        if not replay.metadata:
            replay.metadata = {"policy": "fixed_prefix_uniform_with_replacement"}
        return replay, resume_state["offline_data_sha256"]
    path = Path(config.offline_data)
    identity = file_digest(path)
    provenance = {
        "policy": POOL_POLICY, "path": str(path.resolve()), "sha256": identity,
        "capacity": config.offline_capacity, "seed": config.seed,
    }
    checkpoint = Path(config.output_dir) / "offline-pool.pt"
    reservoir = TransitionReservoir(
        config.offline_capacity, config.seed ^ 0x5A17, config.seed ^ 0x05C01
    )
    complete = False
    reuse = getattr(config, "offline_pool_cache", None)
    cache_path = checkpoint if checkpoint.exists() or reuse is None else Path(reuse)
    if cache_path.exists():
        saved = torch.load(cache_path, map_location="cpu", weights_only=True)
        if saved.get("identity") != provenance or saved.get("status") not in (
            "building", "complete"
        ):
            raise ValueError("Offline pool build identity changed; use a new output directory.")
        reservoir.load_state_dict(saved["reservoir"])
        complete = saved["status"] == "complete"
        if cache_path != checkpoint and not complete:
            raise ValueError("A reused offline pool cache must be complete.")
    elif reuse is not None:
        raise ValueError("The requested completed offline pool cache does not exist.")

    def save(status):
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        atomic_save(checkpoint, {
            "identity": provenance, "status": status,
            "reservoir": reservoir.state_dict(),
        })

    if not complete:
        save("building")
        last_save = time.monotonic()
        # DataLoader's iterator seed must not consume the learner RNG, including
        # when an interrupted build restarts at a saved source row.
        with torch.random.fork_rng(devices=[]):
            reader = DataLoader(
                path=path, num_skip_samples=reservoir.seen,
                rewrite_rooms=None, rewrite_grades=None,
                get_reward=PlacementReward(), dtype=torch.float32,
                batch_size=4096, num_workers=0,
            )
            for batch in reader:
                batch["next", "discount"] = torch.ones(len(batch))
                reservoir.consume(training_columns(batch))
                if time.monotonic() - last_save >= _CHECKPOINT_SECONDS:
                    save("building")
                    last_save = time.monotonic()
        if file_digest(path) != identity:
            raise ValueError("Offline source bytes changed while building the pool.")
    replay = reservoir.replay
    if not len(replay):
        raise ValueError("Offline training data contains no transitions.")
    replay.metadata = {
        "policy": POOL_POLICY, "source_rows": reservoir.seen,
        "source_indices": reservoir.indices[:len(replay)].clone(),
    }
    if not complete:
        save("complete")
    return replay, identity


def mixed_batch(online, offline, config, online_alpha):
    if offline is None:
        return online.sample(config.batch_size), online_alpha, 0
    if len(online) < config.learning_starts:
        return offline.sample(config.batch_size), config.offline_cql_alpha, config.batch_size
    offline_count = min(
        config.batch_size - 1,
        max(1, round(config.batch_size * config.offline_fraction)),
    )
    left = offline.sample(offline_count)
    right = training_columns(online.sample(config.batch_size - offline_count))
    batch = torch.cat((left, right), 0)
    alpha = torch.cat((
        torch.full((offline_count,), config.offline_cql_alpha),
        torch.full((len(right),), online_alpha),
    ))
    return batch, alpha, offline_count
