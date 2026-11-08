"""Build real wheels and check metadata, pip compatibility and native imports."""

from email.parser import Parser
import hashlib
import importlib.machinery
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
import unittest
import zipfile


ROOT = Path(__file__).resolve().parents[2]
BUILD_DEPS = all(
    importlib.util.find_spec(name)
    for name in ("build", "hatchling", "pip", "packaging")
)


@unittest.skipUnless(
    BUILD_DEPS, "Install build, hatchling, pip and packaging for wheel tests"
)
class WheelPackagingTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source = self.root / "source"
        self.source.mkdir()
        for name in (
            "pyproject.toml",
            "hatch_build.py",
            "README.md",
            "LICENSE",
        ):
            shutil.copyfile(ROOT / name, self.source / name)
        (self.source / "kanachan").mkdir()
        (self.source / "kanachan/__init__.py").write_text("", encoding="utf-8")
        self.dist = self.root / "dist"

    def run_python(self, *args, cwd=None):
        return subprocess.run(
            [sys.executable, "-I", *map(str, args)],
            cwd=cwd or self.root,
            capture_output=True,
            text=True,
            timeout=60,
        )

    def build(self, kind="wheel", source=None, success=True):
        result = self.run_python(
            "-m",
            "build",
            f"--{kind}",
            "--no-isolation",
            "--outdir",
            self.dist,
            source or self.source,
        )
        self.assertEqual(
            result.returncode == 0, success, result.stdout + result.stderr
        )
        return result

    def metadata(self, wheel):
        with zipfile.ZipFile(wheel) as archive:
            path = next(
                name
                for name in archive.namelist()
                if name.endswith(".dist-info/WHEEL")
            )
            return Parser().parsestr(
                archive.read(path).decode()
            ), archive.namelist()

    def native_fixture(self):
        binary = os.environ.get("KANACHAN_TEST_NATIVE_FEATURES_PATH")
        if not binary:
            self.skipTest(
                "Set KANACHAN_TEST_NATIVE_FEATURES_PATH to a compiled candidate extension"
            )
        target = self.source / "kanachan" / Path(binary).name
        shutil.copyfile(binary, target)
        return target

    def test_pure_wheel_remains_portable_without_native_payload(self):
        self.build()
        (wheel,) = self.dist.glob("*.whl")
        metadata, entries = self.metadata(wheel)
        self.assertTrue(wheel.name.endswith("-py3-none-any.whl"))
        self.assertEqual(metadata["Root-Is-Purelib"], "true")
        self.assertEqual(metadata.get_all("Tag"), ["py3-none-any"])
        self.assertFalse(
            any(name.endswith((".so", ".pyd")) for name in entries)
        )

    def test_native_wheel_tags_install_and_isolated_import(self):
        from packaging.tags import sys_tags
        from packaging.utils import parse_wheel_filename

        native = self.native_fixture()
        self.build()
        (wheel,) = self.dist.glob("*.whl")
        metadata, entries = self.metadata(wheel)
        tags = parse_wheel_filename(wheel.name)[3]
        self.assertTrue(tags.intersection(sys_tags()))
        self.assertTrue(
            all(tag.abi != "none" and tag.platform != "any" for tag in tags)
        )
        self.assertEqual(metadata["Root-Is-Purelib"], "false")
        self.assertEqual(
            set(metadata.get_all("Tag")), {str(tag) for tag in tags}
        )
        payload = "kanachan/" + native.name
        self.assertIn(payload, entries)
        with zipfile.ZipFile(wheel) as archive:
            self.assertEqual(
                hashlib.sha256(archive.read(payload)).digest(),
                hashlib.sha256(native.read_bytes()).digest(),
            )
        installed = self.root / "installed"
        result = self.run_python(
            "-m",
            "pip",
            "install",
            "--no-deps",
            "--no-index",
            "--target",
            installed,
            wheel,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        result = self.run_python(
            "-c",
            "\n".join(
                (
                    "import json, pathlib, struct, sys",
                    "sys.path.insert(0, sys.argv[1])",
                    "from kanachan import _candidate_features as f",
                    "assert pathlib.Path(f.__file__).resolve().is_relative_to(pathlib.Path(sys.argv[1]).resolve())",
                    "hand = [0] * 37",
                    "for t in [1,2,3,11,12,13,21,22,23,31,31,31,32,32]: hand[t] += 1",
                    "visible = [0] * 34",
                    "for t,n in enumerate(hand):",
                    "    j = t // 10 * 9 + (4 if t % 10 == 0 else t % 10 - 1) if t < 30 else t - 3",
                    "    visible[j] += n",
                    "values = struct.unpack('46f', f.discard_features(hand, visible, [0]*34, [128], 0))",
                    "assert f.SCHEMA_ID == 'discard_afterstate_v1' and values[0] == values[2] == 1",
                    "print(json.dumps({'native': f.__file__, 'width': len(values)}))",
                )
            ),
            installed,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(json.loads(result.stdout)["width"], 46)

    def test_native_wheel_is_rejected_for_foreign_abi_and_platform(self):
        from packaging.utils import parse_wheel_filename

        self.native_fixture()
        self.build()
        (wheel,) = self.dist.glob("*.whl")
        (tag,) = parse_wheel_filename(wheel.name)[3]
        other_platform = (
            "linux_x86_64" if sys.platform == "win32" else "win_amd64"
        )
        for version, abi, platform in (
            ("3.9", "cp39", tag.platform),
            (
                f"{sys.version_info.major}.{sys.version_info.minor}",
                tag.abi,
                other_platform,
            ),
        ):
            with self.subTest(version=version, platform=platform):
                result = self.run_python(
                    "-m",
                    "pip",
                    "install",
                    "--dry-run",
                    "--no-index",
                    "--no-deps",
                    "--only-binary=:all:",
                    "--python-version",
                    version,
                    "--implementation",
                    "cp",
                    "--abi",
                    abi,
                    "--platform",
                    platform,
                    wheel,
                )
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("not a supported wheel", result.stderr)

    def test_invalid_native_binary_fails_build(self):
        binary = (
            self.source
            / "kanachan"
            / (
                "_candidate_features"
                + importlib.machinery.EXTENSION_SUFFIXES[0]
            )
        )
        binary.write_bytes(b"not a compiled extension")
        result = self.build(success=False)
        self.assertIn("cannot load with the build interpreter", result.stderr)
        self.assertFalse(list(self.dist.glob("*.whl")))

    def test_foreign_extension_suffix_fails_build(self):
        (
            self.source / "kanachan/_candidate_features.cpython-999-foreign.so"
        ).write_bytes(b"foreign")
        result = self.build(success=False)
        self.assertIn("does not match this interpreter", result.stderr)

    def test_multiple_native_variants_fail_build(self):
        for suffix in (".so", ".pyd"):
            (
                self.source / f"kanachan/_candidate_features{suffix}"
            ).write_bytes(b"ambiguous")
        result = self.build(success=False)
        self.assertIn("Multiple native binaries", result.stderr)

    def test_sdist_retains_hook_and_excludes_prebuilt_extensions(self):
        (self.source / "kanachan/_candidate_features.so").write_bytes(
            b"local binary"
        )
        self.build("sdist")
        (sdist,) = self.dist.glob("*.tar.gz")
        with tarfile.open(sdist) as archive:
            names = archive.getnames()
            self.assertTrue(
                any(name.endswith("/hatch_build.py") for name in names)
            )
            self.assertFalse(
                any(name.endswith((".so", ".pyd")) for name in names)
            )
            extracted = self.root / "unpacked"
            archive.extractall(extracted, filter="data")
        (source,) = extracted.iterdir()
        self.build(source=source)
        (wheel,) = self.dist.glob("*.whl")
        self.assertTrue(wheel.name.endswith("-py3-none-any.whl"))


if __name__ == "__main__":
    unittest.main()
