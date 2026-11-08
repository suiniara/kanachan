"""Tag wheels containing local CPython extensions for their build interpreter."""

from importlib.machinery import EXTENSION_SUFFIXES
from pathlib import Path
import subprocess
import sys

from hatchling.builders.hooks.plugin.interface import BuildHookInterface


NATIVE_MODULES = (
    "kanachan/_candidate_features",
    "kanachan/simulation/_simulation",
)


class CustomBuildHook(BuildHookInterface):
    def initialize(self, version, build_data):
        root = Path(self.root)
        binaries = []
        for module in NATIVE_MODULES:
            path = root / module
            matches = sorted(
                candidate
                for extension in ("so", "pyd")
                for candidate in path.parent.glob(f"{path.name}*.{extension}")
                if candidate.is_file()
            )
            if len(matches) > 1:
                raise RuntimeError(
                    f"Multiple native binaries for {module}; keep one matching this interpreter."
                )
            for binary in matches:
                if binary.name not in {
                    path.name + suffix for suffix in EXTENSION_SUFFIXES
                }:
                    raise RuntimeError(
                        f"Native binary {binary.name} does not match this interpreter's extension suffixes; rebuild it here."
                    )
                # Bypass kanachan's runtime imports in the isolated build env.
                # A separate process contains loader failures or ABI crashes.
                probe = subprocess.run(
                    [
                        sys.executable,
                        "-I",
                        "-c",
                        "import importlib.util, sys; "
                        "s = importlib.util.spec_from_file_location(sys.argv[1], sys.argv[2]); "
                        "m = importlib.util.module_from_spec(s); s.loader.exec_module(m)",
                        path.name,
                        str(binary.resolve()),
                    ],
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                if probe.returncode:
                    raise RuntimeError(
                        f"Native binary {binary.name} cannot load with the build interpreter; rebuild it here.\n{probe.stderr.strip()}"
                    )
                binaries.append(binary)
        if binaries:
            build_data["pure_python"] = False
            build_data["infer_tag"] = True
            for binary in binaries:
                build_data["force_include"][str(binary)] = binary.relative_to(
                    root
                ).as_posix()
