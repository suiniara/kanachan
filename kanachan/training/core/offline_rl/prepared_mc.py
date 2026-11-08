"""Bounded RAM cache of canonical complete-return MC batches and afterstates."""

import hashlib
import json
import random

from kanachan.training.core.bc.prefetch import CandidateFeaturePrefetch
from kanachan.training.online.state import capture_rng, restore_rng


class PreparedMCCache:
    """Prepare one bounded input pass before updates, then shuffle batch order.

    MC returns and first-epoch batches come from the canonical replay reader.
    Later epochs reuse those batches with a dedicated deterministic permutation.
    Recovery regenerates and hashes the cache instead of serializing it.
    The byte cap covers retained tensors, excluding transient replay/worker RAM.
    """

    can_checkpoint = True

    def __init__(
        self,
        source_factory,
        *,
        workers,
        max_bytes,
        seed,
        epochs=1,
        pin_memory=False,
    ):
        if max_bytes <= 0 or epochs < 1:
            raise ValueError(
                "Prepared MC memory budget and epochs must be positive."
            )
        self.batches = []
        self.position = 0
        self.pin_memory = pin_memory
        self.bytes = 0
        digest = hashlib.sha256()
        rng = capture_rng()
        try:
            random.seed(seed)
            source = source_factory()
            with CandidateFeaturePrefetch(
                source, workers=workers, batches=2 * workers
            ) as pipeline:
                for batch in pipeline:
                    # Release IPC/shared-memory storages instead of retaining
                    # the whole epoch in /dev/shm or keeping one fd per result.
                    batch = batch.clone()
                    assert "mc_return" in batch
                    for key in batch.keys(
                        include_nested=True, leaves_only=True
                    ):
                        tensor = batch[key]
                        self.bytes += tensor.numel() * tensor.element_size()
                        digest.update(str(key).encode())
                        digest.update(tensor.contiguous().numpy().tobytes())
                    if self.bytes > max_bytes:
                        raise MemoryError(
                            "Prepared MC batches exceed the explicit RAM budget."
                        )
                    self.batches.append(batch)
            provenance = source.state_dict()
        finally:
            restore_rng(rng)
        if not self.batches:
            raise ValueError("Prepared MC cache is empty.")
        self.orders = []
        for epoch in range(epochs):
            order = list(range(len(self.batches)))
            if epoch:
                random.Random(seed + epoch).shuffle(order)
            self.orders.extend(order)
        self.identity = {
            "version": "prepared_mc_afterstate_v1",
            "source": provenance["source"],
            "reward_identity": provenance["reward_identity"],
            "discount_factor": provenance["discount_factor"],
            "batch_size": provenance["batch_size"],
            "max_size": provenance["max_size"],
            "seed": seed,
            "epochs": epochs,
            "cache_sha256": digest.hexdigest(),
            "order_sha256": hashlib.sha256(
                json.dumps(self.orders).encode()
            ).hexdigest(),
        }
        self.rows = sum(len(batch) for batch in self.batches)

    def __iter__(self):
        return self

    def __next__(self):
        if self.position == len(self.orders):
            raise StopIteration
        batch = self.batches[self.orders[self.position]].clone(recurse=False)
        self.position += 1
        return batch.pin_memory() if self.pin_memory else batch

    def iter_with_last(self):
        while self.position < len(self.orders):
            batch = next(self)
            yield batch, self.position == len(self.orders)

    def state_dict(self):
        return {
            "format_version": 3,
            "input_samples": 0,
            "prepared_mc": self.identity,
            "batch_position": self.position,
        }

    def load_state_dict(self, state):
        if (
            state.get("format_version") != 3
            or state.get("prepared_mc") != self.identity
        ):
            raise ValueError(
                "Prepared MC checkpoint differs from its regenerated cache."
            )
        cursor = state["batch_position"]
        if type(cursor) is not int or not 0 <= cursor <= len(self.orders):
            raise ValueError("Invalid prepared MC batch position.")
        self.position = cursor

    def close(self):
        self.batches.clear()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
