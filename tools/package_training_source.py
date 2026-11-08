"""Package reproducible Python training sources, examples and test fixtures."""

import argparse
import hashlib
import json
from pathlib import Path
import zipfile


REQUIRED = (
    "pyproject.toml",
    "hatch_build.py",
    "LICENSE",
    "README.md",
    "kanachan/training/iql/train.py",
    "kanachan/training/awr/train.py",
    "kanachan/training/iql/decision-decoder.md",
    "examples/training/iql-placement-reward.py",
    "tools/verify_mc_cql.py",
    "kanachan/candidate_features.py",
    "src/features/candidate_features.cpp",
    "src/simulation/shoupai.cpp",
    "tools/features/CMakeLists.txt",
    "tools/features/README.md",
)


def package(root: Path, output: Path):
    paths = {root / name for name in REQUIRED}
    for pattern in (
        "kanachan/**/*.py",
        "kanachan/**/*.md",
        "test/training/*.py",
        "examples/**/*.py",
        "examples/**/*.yaml",
        "examples/**/*.yml",
        "examples/**/*.md",
        "tools/*.py",
    ):
        paths.update(root.glob(pattern))
    missing = [name for name in REQUIRED if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Required training bundle files are missing: {missing}"
        )
    contents = {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(paths)
    }
    manifest = {
        "format_version": 1,
        "required": list(REQUIRED),
        "files": {
            name: hashlib.sha256(data).hexdigest()
            for name, data in contents.items()
        },
        "scope": "Training sources, candidate-feature native source/build instructions, documentation, examples and tests; native binaries and datasets are supplied separately.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(
        output, "x", compression=zipfile.ZIP_DEFLATED
    ) as archive:
        for name, data in contents.items():
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            archive.writestr(info, data, compress_type=zipfile.ZIP_DEFLATED)
        archive.writestr(
            zipfile.ZipInfo(
                "training-source-manifest.json",
                date_time=(1980, 1, 1, 0, 0, 0),
            ),
            json.dumps(manifest, indent=2) + "\n",
            compress_type=zipfile.ZIP_DEFLATED,
        )
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    manifest = package(Path(__file__).resolve().parents[1], arguments.output)
    print(
        f"Packaged {len(manifest['files'])} source files: {arguments.output.resolve()}"
    )
