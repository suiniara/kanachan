"""Bounded MC prefetch with isolated reader RNG and exact queued-batch recovery."""

import io
import multiprocessing
import os
import random
import traceback
from collections import deque
from threading import Condition, Thread

import torch
from tensordict import TensorDict


def _send(connection, kind, value):
    stream = io.BytesIO()
    torch.save({"kind": kind, "value": value}, stream)
    connection.send_bytes(stream.getvalue())


def _pack_replay(state):
    """Avoid serializing one tiny tensor/storage per transition field."""
    rows = state.pop("replay")
    state["packed_replay"] = (
        tuple(torch.stack(column) for column in zip(*rows)) if rows else ()
    )
    return state


def _unpack_replay(state):
    state = dict(state)
    columns = state.pop("packed_replay")
    state["replay"] = list(zip(*(column.unbind(0) for column in columns)))
    return state


def _reader(connection, arguments, seed, saved, threads):
    # Spawn starts a separate interpreter; the input worker never uses CUDA.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    try:
        from kanachan.candidate_features import derive_candidate_features
        from kanachan.training.core.offline_rl import EpisodeReplayBuffer

        torch.set_num_threads(threads)
        torch.manual_seed(seed)
        random.seed(seed)
        import numpy as np

        np.random.seed(seed)
        source = EpisodeReplayBuffer(**arguments)
        if saved is not None:
            source.load_state_dict(_unpack_replay(saved))
        while True:
            command = connection.recv()
            if command == "next":
                try:
                    batch = next(source)
                except StopIteration:
                    _send(connection, "end", None)
                    continue
                batch["candidate_afterstate"] = derive_candidate_features(
                    batch["sparse"], batch["progression"], batch["candidates"]
                )
                _send(connection, "batch", batch.to_dict())
            elif command == "state":
                _send(connection, "state", _pack_replay(source.state_dict()))
            elif command == "close":
                return
            else:
                raise ValueError(f"Unknown MC reader command: {command}")
    except (EOFError, BrokenPipeError):
        pass
    except BaseException:  # noqa: BLE001
        # Propagate every worker failure instead of leaving the trainer waiting.
        try:
            _send(connection, "error", traceback.format_exc())
        except (EOFError, BrokenPipeError, OSError):
            pass
    finally:
        connection.close()


class StreamingMCPrefetch:
    """Stream canonical replay in one CPU process, with a bounded ready queue.

    The reader owns its RNG. At a checkpoint, a barrier finishes its current
    request and stops further requests. The snapshot contains packed canonical
    replay state, ready batches and the trainer iterator's lookahead batch.
    No sampled transition is lost or regenerated under the learner's RNG.
    """

    can_checkpoint = True

    def __init__(
        self, arguments, *, seed, batches=8, threads=2, pin_memory=False
    ):
        if batches < 1 or threads < 1:
            raise ValueError(
                "MC prefetch capacity and CPU threads must be positive."
            )
        path = arguments["training_data"]
        stat = path.stat()
        self.identity = {
            "version": "streaming_mc_afterstate_v1",
            "source": (str(path.resolve()), stat.st_size, stat.st_mtime_ns),
            "seed": seed,
            "batch_size": arguments["batch_size"],
            "max_size": arguments["max_size"],
            "reward_identity": arguments["reward_identity"],
            "discount_factor": arguments["discount_factor"],
            "rewrite": (
                arguments["rewrite_rooms"],
                arguments["rewrite_grades"],
            ),
        }
        self.arguments = dict(arguments, pin_memory=False)
        self.seed, self.capacity, self.threads = seed, batches, threads
        self.pin_memory = pin_memory
        self.condition = Condition()
        self.ready = deque()
        self.pending = None
        self.saved = None
        self.process = self.thread = self.connection = None
        self.in_flight = self.paused = self.closed = self.exhausted = False
        self.error = None

    def _start(self):
        if self.closed:
            raise RuntimeError("MC prefetch is closed.")
        if self.process is not None:
            return
        context = multiprocessing.get_context("spawn")
        self.connection, child = context.Pipe()
        self.process = context.Process(
            target=_reader,
            args=(child, self.arguments, self.seed, self.saved, self.threads),
            daemon=True,
        )
        self.process.start()
        child.close()
        self.saved = None
        self.thread = Thread(target=self._produce, daemon=True)
        self.thread.start()

    def _rpc(self, command):
        self.connection.send(command)
        if not self.connection.poll(300):
            raise TimeoutError(f"MC reader timed out during {command}.")
        packet = torch.load(
            io.BytesIO(self.connection.recv_bytes()),
            weights_only=True,
            map_location="cpu",
        )
        if packet["kind"] == "error":
            raise RuntimeError("MC reader failed:\n" + packet["value"])
        return packet["kind"], packet["value"]

    @staticmethod
    def _batch(value):
        return TensorDict(value, [len(value["action"])])

    def _produce(self):
        try:
            while True:
                with self.condition:
                    self.condition.wait_for(
                        lambda: (
                            self.closed
                            or (
                                not self.paused
                                and len(self.ready) < self.capacity
                            )
                        )
                    )
                    if self.closed or self.exhausted:
                        return
                    self.in_flight = True
                kind, value = self._rpc("next")
                with self.condition:
                    if kind == "end":
                        self.exhausted = True
                    elif kind == "batch":
                        self.ready.append(self._batch(value))
                    else:
                        raise RuntimeError("Invalid MC reader batch response.")
                    self.in_flight = False
                    self.condition.notify_all()
                    if self.exhausted:
                        return
        except BaseException as error:  # noqa: BLE001
            with self.condition:
                self.error = error
                self.in_flight = False
                self.condition.notify_all()

    def __iter__(self):
        return self

    def __next__(self):
        self._start()
        with self.condition:
            self.condition.wait_for(
                lambda: (
                    self.ready or self.error or self.exhausted or self.closed
                )
            )
            if self.error is not None:
                raise self.error
            if self.closed or not self.ready:
                raise StopIteration
            batch = self.ready.popleft()
            self.condition.notify_all()
        return batch.pin_memory() if self.pin_memory else batch

    def iter_with_last(self):
        if self.pending is None:
            try:
                self.pending = next(self)
            except StopIteration:
                return
        while True:
            current = self.pending
            try:
                self.pending = next(self)
            except StopIteration:
                self.pending = None
                yield current, True
                return
            yield current, False

    def state_dict(self):
        self._start()
        with self.condition:
            self.paused = True
            self.condition.wait_for(lambda: not self.in_flight)
        try:
            if self.error is not None:
                raise self.error
            kind, worker = self._rpc("state")
            if kind != "state":
                raise RuntimeError("Invalid MC reader state response.")
            return {
                "format_version": 4,
                "streaming_mc": dict(self.identity),
                "input_samples": worker["input_samples"],
                "worker": worker,
                "ready": [batch.to_dict() for batch in self.ready],
                "pending": None
                if self.pending is None
                else self.pending.to_dict(),
                "exhausted": self.exhausted,
            }
        finally:
            with self.condition:
                self.paused = False
                self.condition.notify_all()

    def load_state_dict(self, state):
        if self.process is not None:
            raise ValueError(
                "Load MC prefetch state before starting the reader."
            )
        if (
            state.get("format_version") != 4
            or state.get("streaming_mc") != self.identity
        ):
            raise ValueError(
                "Streaming MC checkpoint differs from its source or recipe."
            )
        if state["input_samples"] != self.arguments["num_skip_samples"]:
            raise ValueError("Streaming MC input cursor differs.")
        self.saved = state["worker"]
        self.ready = deque(self._batch(value) for value in state["ready"])
        value = state["pending"]
        self.pending = None if value is None else self._batch(value)
        self.exhausted = state["exhausted"]

    def close(self):
        with self.condition:
            self.closed = True
            self.condition.notify_all()
        if self.process is not None:
            self.thread.join(timeout=5)
            if not self.thread.is_alive() and self.process.is_alive():
                try:
                    self.connection.send("close")
                except (BrokenPipeError, EOFError, OSError):
                    pass
            self.process.join(timeout=5)
            if self.process.is_alive():
                # Only this adapter's reader is signalled after a stuck request.
                self.process.terminate()
                self.process.join(timeout=5)
            if self.process.is_alive():
                self.process.kill()
                self.process.join(timeout=5)
            self.connection.close()
            self.thread.join(timeout=5)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
