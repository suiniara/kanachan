"""Validate a bounded sample before selecting a BC or RL training preset."""

import argparse
import json
from collections import Counter
from itertools import islice
from pathlib import Path

from kanachan.constants import NUM_TYPES_OF_ACTIONS
from kanachan.training.core.bc.data_iterator import DataIterator as BCIterator
from kanachan.training.core.offline_rl.data_iterator import (
    DataIterator as RLIterator,
)


def audit(path: Path, data_format: str, max_samples: int = 10000) -> dict:
    if data_format not in ("bc", "rl") or max_samples < 0:
        raise ValueError("Invalid format or sample limit.")
    reader = (BCIterator if data_format == "bc" else RLIterator)(
        path=path,
        num_skip_samples=0,
        rewrite_rooms=None,
        rewrite_grades=None,
        local_rank=0,
    )
    actions = Counter()
    candidate_total = 0
    terminal = 0
    round_end = 0
    count = 0
    for row in islice(reader, max_samples or None):
        candidate_total += int((row[3] < NUM_TYPES_OF_ACTIONS).sum().item())
        actions[int(row[3][row[4].long()].item())] += 1
        count += 1
        if data_format == "rl":
            round_end += int(row[-2].item())
            terminal += int(row[-1].item())
    if not count:
        raise RuntimeError("The data contains no samples.")
    warnings = []
    if data_format == "rl" and terminal == count:
        warnings.append(
            "Every sampled transition is terminal. BC annotations have the same "
            "eight-column layout: convert complete games with bin/annotation/rl/rl.py "
            "before RL training. A terminal-only dataset cannot teach bootstrapping."
        )
    return {
        "format": data_format,
        "num_samples": count,
        "sample_limit": max_samples,
        "average_legal_actions": candidate_total / count,
        "most_common_action_types": actions.most_common(10),
        **(
            {
                "end_of_game_fraction": terminal / count,
                "end_of_round_fraction": round_end / count,
            }
            if data_format == "rl"
            else {}
        ),
        "warnings": warnings,
    }


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--format", choices=("bc", "rl"), required=True)
    parser.add_argument("--max-samples", type=int, default=10000)
    arguments = parser.parse_args()
    if arguments.max_samples < 0:
        parser.error(
            "max-samples must be non-negative; zero checks the whole file."
        )
    print(
        json.dumps(
            audit(arguments.data, arguments.format, arguments.max_samples),
            indent=2,
        )
    )


if __name__ == "__main__":
    _main()
