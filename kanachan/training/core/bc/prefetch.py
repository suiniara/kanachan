"""Ordered, bounded CPU afterstate prefetch for the sequential BC loader."""

import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from queue import Empty, Full, Queue
from threading import Event, Thread

import torch


def _initialize_worker():
    torch.set_num_threads(1)


def _derive(
    sparse,
    numeric,
    progression,
    candidates,
    mode="discard_v1",
    enable_early_shape_hint=False,
    next_inputs=None,
):
    if mode == "structured_v2":
        from kanachan.structured_v2 import derive_structured_v2_batch

        cur = derive_structured_v2_batch(
            sparse, numeric, progression, candidates,
            enable_early_shape_hint=enable_early_shape_hint,
            include_room_level=False,
        )
        if next_inputs is not None:
            from kanachan.training.iql.joint_v2 import derive_next_structured_v2_batch

            n_sp, n_nu, n_pr, n_ca, done = next_inputs
            nxt = derive_next_structured_v2_batch(
                {"sparse": n_sp, "numeric": n_nu, "progression": n_pr, "candidates": n_ca},
                done,
                enable_early_shape_hint=enable_early_shape_hint,
            )
            return (cur, nxt)
        return cur
    if mode == "none":
        return None
    from kanachan.candidate_features import derive_candidate_features

    return derive_candidate_features(sparse, progression, candidates)


def _attach_features(batch, result, mode):
    if mode == "structured_v2":
        if isinstance(result, tuple):
            cur, nxt = result
            for key, value in cur.items():
                batch[key] = value
            for key, value in nxt.items():
                batch["next", key] = value
        else:
            for key, value in result.items():
                batch[key] = value
    elif result is not None:
        batch["candidate_afterstate"] = result
    return batch


def _parse_and_derive(lines, rewrite_rooms, rewrite_grades, mode="discard_v1", enable_early_shape_hint=False):
    from kanachan.training.core.bc.data_iterator import DataIterator
    from kanachan.training.core.bc.data_loader import DataLoader

    parser = DataIterator(
        path=Path(os.devnull),
        num_skip_samples=0,
        rewrite_rooms=rewrite_rooms,
        rewrite_grades=rewrite_grades,
        local_rank=0,
    )
    rows = [parser.parse_line(line) for line in lines]
    batch = DataLoader.to_tensordict(torch.utils.data.default_collate(rows))
    feats = _derive(
        batch["sparse"], batch["numeric"], batch["progression"], batch["candidates"],
        mode=mode, enable_early_shape_hint=enable_early_shape_hint,
    )
    return _attach_features(batch, feats, mode)


class CandidateFeaturePrefetch:
    """Keep batch order while parsing and deriving public-observation features.

    Raw BC parsing preserves labels; feature derivation receives only sparse
    observations, history and candidates. Live training sources must be
    deterministic, without sampling RNG. MC preparation may consume a replay
    source only before training, with its RNG captured and restored by the
    caller. Construct the source before the trainer restores checkpoint RNG.
    """

    def __init__(
        self,
        source,
        *,
        workers=4,
        batches=8,
        pin_memory=False,
        raw_batches=False,
        rewrite_rooms=None,
        rewrite_grades=None,
        mode="discard_v1",
        enable_early_shape_hint=False,
    ):
        if workers < 1 or batches < 1:
            raise ValueError(
                "Feature workers and prefetch batches must be positive."
            )
        if mode not in ("none", "discard_v1", "structured_v2"):
            raise ValueError(f"Unsupported prefetch mode: {mode}")
        self.source = iter(source)
        self.pin_memory = pin_memory
        self.raw_batches = raw_batches
        self.rewrite_rooms = rewrite_rooms
        self.rewrite_grades = rewrite_grades
        self.mode = mode
        self.enable_early_shape_hint = bool(enable_early_shape_hint)
        self.queue = Queue(maxsize=batches)
        self.stopped = Event()
        self.closed = False
        self.pool = ProcessPoolExecutor(
            max_workers=workers,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=_initialize_worker,
        )
        self.thread = Thread(target=self._produce, daemon=True)
        self.thread.start()

    def _put(self, item):
        while not self.stopped.is_set():
            try:
                self.queue.put(item, timeout=0.1)
                return
            except Full:
                pass

    def _produce(self):
        try:
            for batch in self.source:
                if self.stopped.is_set():
                    break
                if self.raw_batches:
                    future = self.pool.submit(
                        _parse_and_derive,
                        batch,
                        self.rewrite_rooms,
                        self.rewrite_grades,
                        self.mode,
                        self.enable_early_shape_hint,
                    )
                    self._put((None, future))
                else:
                    next_inputs = None
                    if (
                        self.mode == "structured_v2"
                        and "next" in batch
                        and "sparse" in batch["next"]
                        and "done" in batch["next"]
                    ):
                        nxt = batch["next"]
                        next_inputs = (
                            nxt["sparse"],
                            nxt["numeric"],
                            nxt["progression"],
                            nxt["candidates"],
                            nxt["done"],
                        )
                    future = self.pool.submit(
                        _derive,
                        batch["sparse"],
                        batch["numeric"],
                        batch["progression"],
                        batch["candidates"],
                        self.mode,
                        self.enable_early_shape_hint,
                        next_inputs,
                    )
                    self._put((batch, future))
            self._put(None)
        except BaseException as error:  # noqa: BLE001
            # Every producer failure must reach the consumer to avoid a hang.
            self._put(error)

    def __iter__(self):
        return self

    def __next__(self):
        if self.closed:
            raise StopIteration
        item = self.queue.get()
        if item is None:
            self.close()
            raise StopIteration
        if isinstance(item, BaseException):
            self.close()
            raise item
        batch, future = item
        try:
            if batch is None:
                batch = future.result()
            else:
                batch = batch.clone(recurse=False)
                _attach_features(batch, future.result(), self.mode)
            if self.pin_memory:
                batch = batch.pin_memory()
            return batch
        except BaseException:
            self.close()
            raise

    def close(self):
        if self.closed:
            return
        self.closed = True
        self.stopped.set()
        self.thread.join(timeout=5)
        self.pool.shutdown(wait=True, cancel_futures=True)
        src = getattr(self, "source", None)
        self.source = None
        if src is not None and hasattr(src, "close"):
            try:
                src.close()
            except Exception:
                pass
        while True:
            try:
                self.queue.get_nowait()
            except Empty:
                break

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
