from pathlib import Path
import logging
import random
from typing import Any
from tqdm import tqdm
import torch
from torch import Tensor
import torch.utils.data
from torch.distributed import ProcessGroup, new_group, broadcast
from tensordict import TensorDict  # type: ignore
from kanachan.constants import (
    NUM_TYPES_OF_SPARSE_FEATURES,
    MAX_NUM_ACTIVE_SPARSE_FEATURES,
    NUM_NUMERIC_FEATURES,
    NUM_TYPES_OF_PROGRESSION_FEATURES,
    MAX_LENGTH_OF_PROGRESSION_FEATURES,
    NUM_TYPES_OF_ACTIONS,
    MAX_NUM_ACTION_CANDIDATES,
    NUM_TYPES_OF_ROUND_SUMMARY,
    MAX_NUM_ROUND_SUMMARY,
    NUM_RESULTS,
)
from kanachan.training.common import get_distributed_environment
from kanachan.training.core.rl import RewardFunction
from kanachan.training.core.offline_rl.dataset import Dataset


_INTERNAL_BATCH_SIZE = 1024
_LOGGER = logging.getLogger(__name__)


_BUFFER_ELEMENT = tuple[
    Tensor,  # sparse
    Tensor,  # numeric
    Tensor,  # progression
    Tensor,  # candidates
    Tensor,  # action
    Tensor,  # action_value
    Tensor,  # next_sparse
    Tensor,  # next_numeric
    Tensor,  # next_progression
    Tensor,  # next_candidates
    Tensor,  # round_summary
    Tensor,  # results
    Tensor,  # end_of_round
    Tensor,  # end_of_game
    Tensor,  # done
    Tensor,  # reward
    Tensor,  # discount
    Tensor,  # steps_to_next_obs
]


class EpisodeReplayBuffer:
    def __init__(
        self,
        *,
        training_data: Path,
        contiguous_training_data: bool,
        num_skip_samples: int,
        rewrite_rooms: int | None,
        rewrite_grades: int | None,
        get_reward: RewardFunction,
        discount_factor: float,
        dtype: torch.dtype,
        max_size: int,
        batch_size: int,
        num_workers: int,
        pin_memory: bool,
        ranks: list[int] | None = None,
        main_rank: int | None = None,
        reward_identity: str | None = None,
        data_cache: Path | None = None,
        data_cache_bytes: int = 256 * 1024 * 1024,
    ) -> None:
        if not training_data.exists():
            raise RuntimeError(f"{training_data}: Does not exist.")
        if not training_data.is_file():
            raise RuntimeError(f"{training_data}: Not a file.")
        if num_skip_samples < 0:
            raise ValueError(num_skip_samples)
        if rewrite_rooms is not None:
            if rewrite_rooms < 0 or rewrite_rooms > 4:
                raise ValueError(rewrite_rooms)
        if rewrite_grades is not None:
            if rewrite_grades < 0 or rewrite_grades > 15:
                raise ValueError(rewrite_grades)
        if max_size <= 0:
            raise ValueError(max_size)
        if batch_size <= 0:
            raise ValueError(batch_size)
        if batch_size > max_size:
            raise ValueError(f"{batch_size} > {max_size}")
        if num_workers < 0:
            raise ValueError(num_workers)
        world_size, rank, _ = get_distributed_environment()
        if ranks is None:
            ranks = list(range(world_size))
        if len(ranks) == 0 or len(set(ranks)) != len(ranks):
            raise ValueError(ranks)
        if any(r < 0 or r >= world_size for r in ranks) or rank not in ranks:
            raise ValueError(ranks)
        ranks = sorted(ranks)
        if main_rank is None:
            main_rank = ranks[0]
        if main_rank not in ranks:
            raise ValueError(main_rank)

        # Worker batches interleave strided subsets of the file. That order is
        # unsuitable for episode returns and reward functions using continuity.
        if contiguous_training_data and num_workers > 0:
            _LOGGER.warning(
                "Contiguous replay requires sequential input; using num_workers=0."
            )
            num_workers = 0

        dataset = Dataset(
            path=training_data,
            num_skip_samples=num_skip_samples,
            rewrite_rooms=rewrite_rooms,
            rewrite_grades=rewrite_grades,
        )
        self.__data_loader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=_INTERNAL_BATCH_SIZE,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        self.__data_iter = (
            iter(self.__data_loader) if rank == main_rank else None
        )
        self.__batch_cache = None
        if data_cache is not None:
            if num_workers != 0:
                raise ValueError("Parsed batch caching requires sequential, zero-worker input.")
            if rank == main_rank:
                from .batch_cache import CachedBatches

                def reopen(cursor):
                    source = Dataset(
                        path=training_data, num_skip_samples=cursor,
                        rewrite_rooms=rewrite_rooms, rewrite_grades=rewrite_grades,
                    )
                    return iter(torch.utils.data.DataLoader(
                        source, batch_size=_INTERNAL_BATCH_SIZE, num_workers=0,
                    ))

                self.__batch_cache = CachedBatches(
                    path=data_cache, source=training_data, start=num_skip_samples,
                    batch_size=_INTERNAL_BATCH_SIZE, byte_budget=data_cache_bytes,
                    rewrite_rooms=rewrite_rooms, rewrite_grades=rewrite_grades,
                    first_reader=self.__data_iter, reader_factory=reopen,
                )
                self.__data_iter = self.__batch_cache
        self.__contiguous = contiguous_training_data
        self.__sparse: Tensor = torch.empty(
            0, MAX_NUM_ACTIVE_SPARSE_FEATURES, device="cpu", dtype=torch.int32
        )
        self.__numeric: Tensor = torch.empty(
            0, NUM_NUMERIC_FEATURES, device="cpu", dtype=torch.int32
        )
        self.__progression: Tensor = torch.empty(
            0,
            MAX_LENGTH_OF_PROGRESSION_FEATURES,
            device="cpu",
            dtype=torch.int32,
        )
        self.__candidates: Tensor = torch.empty(
            0, MAX_NUM_ACTION_CANDIDATES, device="cpu", dtype=torch.int32
        )
        self.__action: Tensor = torch.empty(0, device="cpu", dtype=torch.int32)
        self.__next_sparse: Tensor = torch.empty(
            0, MAX_NUM_ACTIVE_SPARSE_FEATURES, device="cpu", dtype=torch.int32
        )
        self.__next_numeric: Tensor = torch.empty(
            0, NUM_NUMERIC_FEATURES, device="cpu", dtype=torch.int32
        )
        self.__next_progression: Tensor = torch.empty(
            0,
            MAX_LENGTH_OF_PROGRESSION_FEATURES,
            device="cpu",
            dtype=torch.int32,
        )
        self.__next_candidates: Tensor = torch.empty(
            0, MAX_NUM_ACTION_CANDIDATES, device="cpu", dtype=torch.int32
        )
        self.__round_summary: Tensor = torch.empty(
            0, MAX_NUM_ROUND_SUMMARY, device="cpu", dtype=torch.int32
        )
        self.__results: Tensor = torch.empty(
            0, NUM_RESULTS, device="cpu", dtype=torch.int32
        )
        self.__end_of_round: Tensor = torch.empty(
            0, device="cpu", dtype=torch.bool
        )
        self.__done: Tensor = torch.empty(0, device="cpu", dtype=torch.bool)
        self.__get_reward = get_reward
        self.__discount_factor = discount_factor
        self.__dtype = dtype
        self.__replay_buffer: list[_BUFFER_ELEMENT] = []
        self.__batch_size = batch_size
        self.__max_size = max_size
        self.__group: ProcessGroup | None = None
        if len(ranks) >= 2:
            self.__group = new_group(
                ranks=ranks, backend="gloo", use_local_synchronization=True
            )
        self.__main_rank = main_rank
        self.__rank = rank
        self.__rank_in_group = ranks.index(rank)
        self.__num_processes = len(ranks)
        self.__finished = False
        self.__first_iteration = True
        self.__can_checkpoint = num_workers == 0 and len(ranks) == 1
        self.__input_samples = num_skip_samples
        self.__source_path = training_data
        stat = training_data.stat()
        self.__source_signature = (str(training_data.resolve()), stat.st_size, stat.st_mtime_ns)
        self.__pending = None
        self.__reward_identity = reward_identity
        self.__rewrite_config = (rewrite_rooms, rewrite_grades)
        self.__sampling_policy = "swap_v2"

    def __iter__(self) -> "EpisodeReplayBuffer":
        return self

    @property
    def can_checkpoint(self) -> bool:
        return self.__can_checkpoint

    @property
    def cache_statistics(self):
        return None if self.__batch_cache is None else dict(self.__batch_cache.stats)

    def iter_with_last(self):
        """Keep lookahead inside the checkpoint instead of a generator local."""
        if self.__pending is None:
            try:
                self.__pending = next(self)
            except StopIteration:
                return
        while True:
            current = self.__pending
            try:
                self.__pending = next(self)
            except StopIteration:
                self.__pending = None
                yield current, True
                return
            yield current, False

    def state_dict(self):
        if not self.can_checkpoint:
            raise ValueError("Exact offline replay recovery requires one rank and sequential input.")
        from kanachan.training.online.state import capture_rng, cpu_state

        names = ("sparse", "numeric", "progression", "candidates", "action",
                 "next_sparse", "next_numeric", "next_progression", "next_candidates",
                 "round_summary", "results", "end_of_round", "done")
        result = {
            "format_version": 1 if self.__sampling_policy == "ordered_v1" else 2,
            "source": self.__source_signature,
            "input_samples": self.__input_samples,
            "batch_size": self.__batch_size, "max_size": self.__max_size,
            "contiguous": self.__contiguous, "discount_factor": self.__discount_factor,
            "dtype": self.__dtype,
            "reward_identity": self.__reward_identity, "rewrite_config": self.__rewrite_config,
            "buffers": {name: getattr(self, "_EpisodeReplayBuffer__" + name) for name in names},
            "replay": self.__replay_buffer, "finished": self.__finished,
            "pending": None if self.__pending is None else self.__pending.to_dict(),
            "rng": capture_rng(),
        }
        if self.__sampling_policy != "ordered_v1":
            result["sampling_policy"] = self.__sampling_policy
        return cpu_state(result)

    def load_state_dict(self, state):
        version = state.get("format_version")
        policy = "ordered_v1" if version == 1 else state.get("sampling_policy")
        if (not self.can_checkpoint or version not in (1, 2)
                or (version == 2 and policy != "swap_v2")
                or tuple(state["source"]) != self.__source_signature
                or state["batch_size"] != self.__batch_size
                or state["max_size"] != self.__max_size
                or state["contiguous"] != self.__contiguous
                or state["discount_factor"] != self.__discount_factor
                or state["dtype"] != self.__dtype
                or state.get("reward_identity") != self.__reward_identity
                or tuple(state.get("rewrite_config", (None, None))) != self.__rewrite_config
                or state["input_samples"] != self.__input_samples):
            raise ValueError("Offline replay checkpoint does not match its source or configuration.")
        from kanachan.training.online.state import restore_rng

        self.__sampling_policy = policy
        for name, value in state["buffers"].items():
            setattr(self, "_EpisodeReplayBuffer__" + name, value.cpu().clone())
        self.__replay_buffer = list(state["replay"])
        self.__finished = state["finished"]
        self.__first_iteration = False
        pending = state["pending"]
        self.__pending = None if pending is None else TensorDict(pending, [len(pending["action"])])
        restore_rng(state["rng"])

    def _append(self, episode: TensorDict) -> None:
        for i in range(episode.batch_size[0]):
            td = episode[i].to_tensordict()  # type: ignore
            sparse = td["sparse"]
            numeric = td["numeric"]
            progression = td["progression"]
            candidates = td["candidates"]
            action = td["action"]
            action_value = td["action_value"]
            next_sparse = td["next", "sparse"]
            next_numeric = td["next", "numeric"]
            next_progression = td["next", "progression"]
            next_candidates = td["next", "candidates"]
            round_summary = td["next", "round_summary"]
            results = td["next", "results"]
            end_of_round = td["next", "end_of_round"]
            end_of_game = td["next", "end_of_game"]
            done = td["next", "done"]
            reward = td["next", "reward"]
            discount = td["next", "discount"]
            steps = td["next", "steps_to_next_obs"]
            self.__replay_buffer.append(
                (
                    sparse,
                    numeric,
                    progression,
                    candidates,
                    action,
                    action_value,
                    next_sparse,
                    next_numeric,
                    next_progression,
                    next_candidates,
                    round_summary,
                    results,
                    end_of_round,
                    end_of_game,
                    done,
                    reward,
                    discount,
                    steps,
                )
            )

    def _sample(self) -> _BUFFER_ELEMENT:
        if len(self.__replay_buffer) == 0:
            errmsg = "The replay buffer is empty."
            raise RuntimeError(errmsg)
        idx = random.randrange(len(self.__replay_buffer))
        if self.__sampling_policy == "ordered_v1":
            return self.__replay_buffer.pop(idx)
        result = self.__replay_buffer[idx]
        self.__replay_buffer[idx] = self.__replay_buffer[-1]
        self.__replay_buffer.pop()
        return result

    def _sample_batch(self, batch_size: int) -> _BUFFER_ELEMENT:
        num_processes = self.__num_processes

        if len(self.__replay_buffer) < batch_size * num_processes:
            errmsg = "The replay buffer is too small."
            raise RuntimeError(errmsg)

        batch: list[list[Tensor]] = [[] for _ in range(18)]
        for _ in range(batch_size * num_processes):
            t = self._sample()
            for i in range(18):
                batch[i].append(t[i])

        return (
            torch.stack(batch[0]),
            torch.stack(batch[1]),
            torch.stack(batch[2]),
            torch.stack(batch[3]),
            torch.stack(batch[4]),
            torch.stack(batch[5]),
            torch.stack(batch[6]),
            torch.stack(batch[7]),
            torch.stack(batch[8]),
            torch.stack(batch[9]),
            torch.stack(batch[10]),
            torch.stack(batch[11]),
            torch.stack(batch[12]),
            torch.stack(batch[13]),
            torch.stack(batch[14]),
            torch.stack(batch[15]),
            torch.stack(batch[16]),
            torch.stack(batch[17]),
        )

    def _broadcast_status(self, status: int = 0) -> int:
        header = torch.tensor(status, device="cpu", dtype=torch.int64)
        if self.__group is not None:
            broadcast(header, src=self.__main_rank, group=self.__group)
        return int(header.item())

    def _broadcast(
        self, batch: _BUFFER_ELEMENT | None, total_size: int
    ) -> TensorDict:
        if batch is None:
            shapes = (
                (MAX_NUM_ACTIVE_SPARSE_FEATURES,),
                (NUM_NUMERIC_FEATURES,),
                (MAX_LENGTH_OF_PROGRESSION_FEATURES,),
                (MAX_NUM_ACTION_CANDIDATES,),
                (),
                (),
                (MAX_NUM_ACTIVE_SPARSE_FEATURES,),
                (NUM_NUMERIC_FEATURES,),
                (MAX_LENGTH_OF_PROGRESSION_FEATURES,),
                (MAX_NUM_ACTION_CANDIDATES,),
                (MAX_NUM_ROUND_SUMMARY,),
                (NUM_RESULTS,),
                (),
                (),
                (),
                (),
            )
            shapes = (*shapes, (), ())
            dtypes = [torch.int32] * 18
            dtypes[5] = dtypes[15] = dtypes[16] = self.__dtype
            dtypes[17] = torch.int64
            dtypes[12] = dtypes[13] = dtypes[14] = torch.bool
            tensors = [
                torch.empty((total_size, *shape), device="cpu", dtype=dtype)
                for shape, dtype in zip(shapes, dtypes, strict=True)
            ]
        else:
            tensors = list(batch)

        batch_size = total_size // self.__num_processes
        if self.__group is not None:
            first = batch_size * self.__rank_in_group
            last = first + batch_size
            for i, tensor in enumerate(tensors):
                broadcast(tensor, src=self.__main_rank, group=self.__group)
                tensors[i] = tensor[first:last]

        keys = (
            "sparse",
            "numeric",
            "progression",
            "candidates",
            "action",
            "action_value",
            ("next", "sparse"),
            ("next", "numeric"),
            ("next", "progression"),
            ("next", "candidates"),
            ("next", "round_summary"),
            ("next", "results"),
            ("next", "end_of_round"),
            ("next", "end_of_game"),
            ("next", "done"),
            ("next", "reward"),
            ("next", "discount"),
            ("next", "steps_to_next_obs"),
        )
        result = TensorDict({}, batch_size=batch_size, device="cpu")
        for key, tensor in zip(keys, tensors, strict=True):
            result[key] = tensor
        if self.__contiguous:
            # Keep fixed, complete-trajectory labels separate from network
            # output. Deriving them after broadcast preserves old wire and
            # checkpoint layouts while avoiding shared mutable storage.
            result["mc_return"] = result["action_value"].detach().clone()
        return result

    def __next__(self) -> TensorDict:
        if self.__finished:
            raise StopIteration

        batch = None
        total_size = 0
        if self.__rank == self.__main_rank:
            try:
                batch = self._next_main()
                total_size = batch[0].size(0)
            except StopIteration:
                pass
            except Exception:
                # Every reader must leave the same collective when the source
                # fails, rather than hanging in a tensor broadcast forever.
                self.__finished = True
                self.__data_iter = None
                self._broadcast_status(-1)
                raise
        total_size = self._broadcast_status(total_size)
        if total_size <= 0:
            self.__finished = True
            self.__data_iter = None
            if total_size < 0:
                raise RuntimeError(
                    "Replay buffer reader failed on the main rank."
                )
            raise StopIteration
        return self._broadcast(batch, total_size)

    def _next_main(self) -> _BUFFER_ELEMENT:
        num_processes = self.__num_processes
        _, _, local_rank = get_distributed_environment()
        assert self.__data_iter is not None

        if (
            len(self.__replay_buffer)
            >= self.__max_size + self.__batch_size * num_processes
        ):
            return self._sample_batch(self.__batch_size)

        progress: tqdm | None = None
        if self.__first_iteration:
            progress = tqdm(
                desc="Loading data to replay buffer...",
                total=self.__max_size,
                maxinterval=0.1,
                disable=(local_rank != 0),
                unit=" samples",
                smoothing=0.0,
            )
            self.__first_iteration = False

        while True:
            try:
                data = next(self.__data_iter)
            except StopIteration:
                if progress is not None:
                    progress.close()
                if self.__done.numel() > 0:
                    raise RuntimeError(
                        "Incomplete episode at the end of contiguous replay data."
                    ) from None
                batch_size = min(
                    self.__batch_size,
                    len(self.__replay_buffer) // num_processes,
                )
                if batch_size > 0:
                    return self._sample_batch(batch_size)
                if self.__replay_buffer:
                    _LOGGER.warning(
                        "Dropping %d replay samples: fewer than the number of ranks (%d).",
                        len(self.__replay_buffer),
                        num_processes,
                    )
                    self.__replay_buffer.clear()
                raise

            assert isinstance(data, list)
            assert len(data) == 13
            input_batch_size = data[0].size(0)
            assert input_batch_size > 0
            self.__input_samples += input_batch_size

            sparse: Tensor = data[0]
            assert isinstance(sparse, Tensor)
            assert sparse.device == torch.device("cpu")
            assert sparse.dtype == torch.int32
            assert sparse.dim() == 2
            assert sparse.size(0) == input_batch_size
            assert sparse.size(1) == MAX_NUM_ACTIVE_SPARSE_FEATURES
            assert torch.all(sparse >= 0).item()
            assert torch.all(sparse <= NUM_TYPES_OF_SPARSE_FEATURES).item()
            self.__sparse = torch.cat((self.__sparse, sparse), dim=0)

            numeric: Tensor = data[1]
            assert isinstance(numeric, Tensor)
            assert numeric.device == torch.device("cpu")
            assert numeric.dtype == torch.int32
            assert numeric.dim() == 2
            assert numeric.size(0) == input_batch_size
            assert numeric.size(1) == NUM_NUMERIC_FEATURES
            self.__numeric = torch.cat((self.__numeric, numeric), dim=0)

            progression: Tensor = data[2]
            assert isinstance(progression, Tensor)
            assert progression.device == torch.device("cpu")
            assert progression.dtype == torch.int32
            assert progression.dim() == 2
            assert progression.size(0) == input_batch_size
            assert progression.size(1) == MAX_LENGTH_OF_PROGRESSION_FEATURES
            assert torch.all(progression >= 0).item()
            assert torch.all(
                progression <= NUM_TYPES_OF_PROGRESSION_FEATURES
            ).item()
            self.__progression = torch.cat(
                (self.__progression, progression), dim=0
            )

            candidates: Tensor = data[3]
            assert isinstance(candidates, Tensor)
            assert candidates.dtype == torch.int32
            assert candidates.device == torch.device("cpu")
            assert candidates.dim() == 2
            assert candidates.size(0) == input_batch_size
            assert candidates.size(1) == MAX_NUM_ACTION_CANDIDATES
            assert torch.all(candidates >= 0).item()
            assert torch.all(candidates <= NUM_TYPES_OF_ACTIONS).item()
            self.__candidates = torch.cat(
                (self.__candidates, candidates), dim=0
            )

            action: Tensor = data[4]
            assert isinstance(action, Tensor)
            assert action.dtype == torch.int32
            assert action.dim() == 1
            assert action.size(0) == input_batch_size
            assert torch.all(action >= 0).item()
            assert torch.all(action < MAX_NUM_ACTION_CANDIDATES).item()
            self.__action = torch.cat((self.__action, action), dim=0)

            next_sparse: Tensor = data[5]
            assert isinstance(next_sparse, Tensor)
            assert next_sparse.device == torch.device("cpu")
            assert next_sparse.dtype == torch.int32
            assert next_sparse.dim() == 2
            assert next_sparse.size(0) == input_batch_size
            assert next_sparse.size(1) == MAX_NUM_ACTIVE_SPARSE_FEATURES
            assert torch.all(next_sparse >= 0).item()
            assert torch.all(
                next_sparse <= NUM_TYPES_OF_SPARSE_FEATURES
            ).item()
            self.__next_sparse = torch.cat(
                (self.__next_sparse, next_sparse), dim=0
            )

            next_numeric: Tensor = data[6]
            assert isinstance(next_numeric, Tensor)
            assert next_numeric.device == torch.device("cpu")
            assert next_numeric.dtype == torch.int32
            assert next_numeric.dim() == 2
            assert next_numeric.size(0) == input_batch_size
            assert next_numeric.size(1) == NUM_NUMERIC_FEATURES
            self.__next_numeric = torch.cat(
                (self.__next_numeric, next_numeric), dim=0
            )

            next_progression: Tensor = data[7]
            assert isinstance(next_progression, Tensor)
            assert next_progression.device == torch.device("cpu")
            assert next_progression.dtype == torch.int32
            assert next_progression.dim() == 2
            assert next_progression.size(0) == input_batch_size
            assert (
                next_progression.size(1) == MAX_LENGTH_OF_PROGRESSION_FEATURES
            )
            assert torch.all(next_progression >= 0).item()
            assert torch.all(
                next_progression <= NUM_TYPES_OF_PROGRESSION_FEATURES
            ).item()
            self.__next_progression = torch.cat(
                (self.__next_progression, next_progression), dim=0
            )

            next_candidates: Tensor = data[8]
            assert isinstance(next_candidates, Tensor)
            assert next_candidates.device == torch.device("cpu")
            assert next_candidates.dtype == torch.int32
            assert next_candidates.dim() == 2
            assert next_candidates.size(0) == input_batch_size
            assert next_candidates.size(1) == MAX_NUM_ACTION_CANDIDATES
            assert torch.all(next_candidates >= 0).item()
            assert torch.all(next_candidates <= NUM_TYPES_OF_ACTIONS).item()
            self.__next_candidates = torch.cat(
                (self.__next_candidates, next_candidates), dim=0
            )

            round_summary: Tensor = data[9]
            assert isinstance(round_summary, Tensor)
            assert round_summary.device == torch.device("cpu")
            assert round_summary.dtype == torch.int32
            assert round_summary.dim() == 2
            assert round_summary.size(0) == input_batch_size
            assert round_summary.size(1) == MAX_NUM_ROUND_SUMMARY
            assert torch.all(round_summary >= 0).item()
            assert torch.all(
                round_summary <= NUM_TYPES_OF_ROUND_SUMMARY
            ).item()
            self.__round_summary = torch.cat(
                (self.__round_summary, round_summary), dim=0
            )

            results: Tensor = data[10]
            assert isinstance(results, Tensor)
            assert results.device == torch.device("cpu")
            assert results.dtype == torch.int32
            assert results.dim() == 2
            assert results.size(0) == input_batch_size
            assert results.size(1) == NUM_RESULTS
            self.__results = torch.cat((self.__results, results), dim=0)

            end_of_round: Tensor = data[11]
            assert isinstance(end_of_round, Tensor)
            assert end_of_round.device == torch.device("cpu")
            assert end_of_round.dtype == torch.bool
            assert end_of_round.dim() == 1
            assert end_of_round.size(0) == input_batch_size
            self.__end_of_round = torch.cat(
                (self.__end_of_round, end_of_round), dim=0
            )

            done: Tensor = data[12]
            assert isinstance(done, Tensor)
            assert done.device == torch.device("cpu")
            assert done.dtype == torch.bool
            assert done.dim() == 1
            assert done.size(0) == input_batch_size
            self.__done = torch.cat((self.__done, done), dim=0)

            if self.__contiguous:
                terminal_indices = self.__done.nonzero().flatten()
                if terminal_indices.numel() == 0:
                    continue
                length = int(terminal_indices[-1].item()) + 1
            else:
                # Shuffled rows are independent transitions, not episodes.
                length = self.__done.size(0)

            source: dict[str, Any] = {
                "sparse": self.__sparse[:length],
                "numeric": self.__numeric[:length],
                "progression": self.__progression[:length],
                "candidates": self.__candidates[:length],
                "action": self.__action[:length],
                "next": {
                    "sparse": self.__next_sparse[:length],
                    "numeric": self.__next_numeric[:length],
                    "progression": self.__next_progression[:length],
                    "candidates": self.__next_candidates[:length],
                    "round_summary": self.__round_summary[:length],
                    "results": self.__results[:length],
                    "end_of_game": self.__done[:length],
                    "end_of_round": self.__end_of_round[:length],
                    "done": self.__done[:length].clone(),
                },
            }
            episode = TensorDict(
                source,
                batch_size=length,
                device=torch.device("cpu"),
            )

            with torch.no_grad():
                self.__get_reward(episode, self.__contiguous)
            if episode.get(("next", "reward"), None) is None:  # type: ignore
                errmsg = (
                    '`get_reward` did not set the `("next", "reward")` tensor.'
                )
                raise RuntimeError(errmsg)
            reward: Tensor = episode["next", "reward"]
            if not isinstance(reward, Tensor):
                errmsg = "The `reward` is not a tensor."
                raise RuntimeError(errmsg)
            if reward.dim() not in (1, 2):
                errmsg = "An invalid shape of the `reward` tensor."
                raise RuntimeError(errmsg)
            if reward.dim() == 2:
                if reward.size(1) != 1:
                    errmsg = "An invalid shape of the `reward` tensor."
                    raise RuntimeError(errmsg)
                reward.squeeze_(1)
            if reward.size(0) != length:
                errmsg = "An invalid shape of the `reward` tensor."
                raise RuntimeError(errmsg)
            if reward.dtype not in (
                torch.float64,
                torch.float32,
                torch.float16,
            ):
                errmsg = "An invalid `dtype` of the `reward` tensor."
                raise RuntimeError(errmsg)
            episode["next", "reward"] = (
                reward.to(self.__dtype).detach().clone()
            )

            discount = episode.get(("next", "discount"), None)
            if discount is None:
                discount = torch.full_like(episode["next", "reward"], self.__discount_factor)
            if discount.shape != (length,) or not discount.isfinite().all() or not ((discount >= 0) & (discount <= 1)).all():
                raise RuntimeError("Invalid per-transition discount in replay data.")
            episode["next", "discount"] = discount.to(dtype=self.__dtype).detach().clone()
            steps = episode.get(("next", "steps_to_next_obs"), None)
            if steps is None:
                steps = torch.ones(length, dtype=torch.int64)
            if steps.shape != (length,) or steps.dtype not in (torch.int32, torch.int64) or not (steps > 0).all():
                raise RuntimeError("Invalid transition horizon in replay data.")
            episode["next", "steps_to_next_obs"] = steps.to(dtype=torch.int64).detach().clone()

            episode["action_value"] = episode["next", "reward"].clone()
            if self.__contiguous:
                for i in range(length - 1, -1, -1):
                    if not episode["next", "done"][i].item():
                        following = i + int(steps[i].item())
                        if following >= length:
                            raise RuntimeError("The reward function removed the episode's terminal flag or supplied an invalid horizon.")
                        episode["action_value"][i] += (
                            discount[i] * episode["action_value"][following]
                        )

            if progress is not None:
                if len(self.__replay_buffer) + length <= self.__max_size:
                    progress.update(length)
                elif len(self.__replay_buffer) < self.__max_size:
                    progress.update(
                        self.__max_size - len(self.__replay_buffer)
                    )

            self._append(episode)
            self.__sparse = self.__sparse[length:]
            self.__numeric = self.__numeric[length:]
            self.__progression = self.__progression[length:]
            self.__candidates = self.__candidates[length:]
            self.__action = self.__action[length:]
            self.__next_sparse = self.__next_sparse[length:]
            self.__next_numeric = self.__next_numeric[length:]
            self.__next_progression = self.__next_progression[length:]
            self.__next_candidates = self.__next_candidates[length:]
            self.__round_summary = self.__round_summary[length:]
            self.__results = self.__results[length:]
            self.__end_of_round = self.__end_of_round[length:]
            self.__done = self.__done[length:]

            if (
                len(self.__replay_buffer)
                >= self.__max_size + self.__batch_size * num_processes
            ):
                if progress is not None:
                    progress.close()
                return self._sample_batch(self.__batch_size)

        raise RuntimeError("Never reach here.")

    def __len__(self) -> int:
        return len(self.__replay_buffer)
