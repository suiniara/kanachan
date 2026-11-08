import datetime
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

import torch
from torch import distributed

from kanachan.training.core.offline_rl import EpisodeReplayBuffer


def _worker(root: Path, mode: str, rank: int, world_size: int) -> None:
    os.environ["LOCAL_RANK"] = str(rank)
    torch.set_num_threads(1)
    print(f"rank {rank}: initializing Gloo", file=sys.stderr, flush=True)
    distributed.init_process_group(
        backend="gloo",
        init_method=os.environ.get(
            "KANACHAN_TEST_RENDEZVOUS", (root / "store").as_uri()
        ),
        rank=rank,
        world_size=world_size,
        timeout=datetime.timedelta(seconds=30),
    )
    print(f"rank {rank}: Gloo ready", file=sys.stderr, flush=True)
    result = {"rank": rank, "ids": [], "batches": []}
    if mode != "subgroup" or rank != 0:

        def reward(data, contiguous):
            data["next", "reward"] = torch.ones(data.batch_size)

        kwargs = (
            {"ranks": [1, 2], "main_rank": 2} if mode == "subgroup" else {}
        )
        replay = EpisodeReplayBuffer(
            training_data=root / "data.tsv",
            contiguous_training_data=True,
            num_skip_samples=0,
            rewrite_rooms=None,
            rewrite_grades=None,
            get_reward=reward,
            discount_factor=0.9,
            dtype=torch.float32,
            max_size=8,
            batch_size=4,
            num_workers=0,
            pin_memory=False,
            **kwargs,
        )
        print(f"rank {rank}: replay ready", file=sys.stderr, flush=True)
        try:
            for batch in replay:
                result["batches"].append(batch.batch_size[0])
                result["ids"].extend(batch["numeric"][:, 0].tolist())
        except RuntimeError as error:
            result["error"] = str(error)
        # All ranks must remember EOF/error without entering another collective.
        try:
            next(replay)
        except StopIteration:
            result["finished"] = True
    distributed.barrier()
    distributed.destroy_process_group()
    print("REPLAY_RESULT: " + json.dumps(result), flush=True)


@unittest.skipUnless(
    distributed.is_gloo_available()
    and os.environ.get("KANACHAN_TEST_DISTRIBUTED") == "1",
    "Set KANACHAN_TEST_DISTRIBUTED=1 in an environment supporting Gloo rendezvous",
)
class ReplayDistributedTests(unittest.TestCase):
    def test_distributed_eof_failure_and_nonzero_rank_subgroup(self):
        from test_training_pipeline import rl_line

        for mode, world_size in (
            ("complete", 2),
            ("error", 2),
            ("subgroup", 3),
        ):
            with (
                self.subTest(mode=mode),
                tempfile.TemporaryDirectory() as directory,
            ):
                root = Path(directory)
                text = "".join(rl_line(i, terminal=True) for i in range(13))
                if mode == "error":
                    text += "invalid\n"
                (root / "data.tsv").write_text(text)
                environment = dict(
                    os.environ,
                    PYTHONDONTWRITEBYTECODE="1",
                    OMP_NUM_THREADS="1",
                    MKL_NUM_THREADS="1",
                    TORCHDYNAMO_DISABLE="1",
                )
                processes = [
                    subprocess.Popen(
                        [
                            sys.executable,
                            "-B",
                            "-m",
                            "test.training.test_replay_distributed",
                            "--worker",
                            str(root),
                            mode,
                            str(rank),
                            str(world_size),
                        ],
                        cwd=Path(__file__).resolve().parents[2],
                        env=environment,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                        encoding="utf-8",
                        errors="replace",
                    )
                    for rank in range(world_size)
                ]
                results = []
                try:
                    for process in processes:
                        try:
                            stdout, stderr = process.communicate(timeout=45)
                        except subprocess.TimeoutExpired as error:
                            self.fail(
                                f"Replay worker timed out: {error.stdout!r} {error.stderr!r}"
                            )
                        self.assertEqual(
                            process.returncode, 0, stdout + stderr
                        )
                        line = next(
                            line
                            for line in stdout.splitlines()
                            if line.startswith("REPLAY_RESULT: ")
                        )
                        results.append(
                            json.loads(line.removeprefix("REPLAY_RESULT: "))
                        )
                finally:
                    for process in processes:
                        if process.poll() is None:
                            process.terminate()
                            process.communicate(timeout=10)
                participants = [
                    result
                    for result in results
                    if mode != "subgroup" or result["rank"] != 0
                ]
                self.assertTrue(
                    all(result["finished"] for result in participants)
                )
                if mode == "error":
                    self.assertTrue(
                        all("error" in result for result in participants),
                        results,
                    )
                else:
                    self.assertTrue(
                        all(
                            result["batches"] == [4, 2]
                            for result in participants
                        ),
                        results,
                    )
                    ids = [
                        index
                        for result in participants
                        for index in result["ids"]
                    ]
                    self.assertEqual(len(ids), 12)
                    self.assertEqual(len(set(ids)), 12)
                    self.assertTrue(set(ids) <= set(range(13)))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--worker":
        _worker(
            Path(sys.argv[2]), sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
        )
    else:
        unittest.main()
