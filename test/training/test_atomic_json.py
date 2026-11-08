"""Atomic metadata publication under real Windows sharing conflicts."""

import ctypes
from ctypes import wintypes
import json
import os
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from kanachan.data.catalog import atomic_json


@unittest.skipUnless(os.name == "nt", "Requires Windows file-sharing semantics")
class AtomicJsonSharingTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.target = self.root / "progress.json"
        atomic_json(self.target, {"version": 1})
        self.kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        self.kernel.CreateFileW.argtypes = [wintypes.LPCWSTR,wintypes.DWORD,wintypes.DWORD,wintypes.LPVOID,wintypes.DWORD,wintypes.DWORD,wintypes.HANDLE]
        self.kernel.CreateFileW.restype = wintypes.HANDLE
        self.kernel.CloseHandle.argtypes = [wintypes.HANDLE]

    def hold(self):
        handle = self.kernel.CreateFileW(str(self.target),0x80000000,3,None,3,0x80,None)
        if handle == wintypes.HANDLE(-1).value:
            raise ctypes.WinError(ctypes.get_last_error())
        lock = threading.Lock()
        closed = False

        def release():
            nonlocal closed
            with lock:
                if not closed:
                    self.kernel.CloseHandle(handle)
                    closed = True
        self.addCleanup(release)
        return release

    def test_transient_reader_lock_preserves_exact_utf8_serialization(self):
        release = self.hold()
        timer = threading.Timer(0.15,release)
        timer.start()
        value = {"版本": 2, "tiles": ["东", None, 123]}
        try:
            atomic_json(self.target,value)
        finally:
            timer.join(timeout=2)
            release()
        expected = (json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+"\n").replace("\n",os.linesep).encode("utf-8")
        self.assertEqual(self.target.read_bytes(),expected)
        self.assertFalse(self.target.with_name("progress.json.tmp").exists())

    def test_persistent_reader_lock_is_bounded_and_retains_old_complete_file(self):
        old = self.target.read_bytes()
        release = self.hold()
        begin = time.perf_counter()
        try:
            with self.assertRaises(PermissionError):
                atomic_json(self.target,{"version":2})
        finally:
            release()
        self.assertLess(time.perf_counter()-begin,8)
        self.assertEqual(self.target.read_bytes(),old)
        self.assertEqual(json.loads(self.target.with_name("progress.json.tmp").read_bytes()),{"version":2})


class AtomicJsonErrorTest(unittest.TestCase):
    def test_unrelated_permission_failure_is_not_hidden_or_retried(self):
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary)/"state.json"
            error = PermissionError(13,"permanent ACL failure")
            with patch("pathlib.Path.replace",side_effect=error) as replace:
                with self.assertRaises(PermissionError) as caught:
                    atomic_json(target,{"value":1})
            self.assertIs(caught.exception,error)
            self.assertEqual(replace.call_count,1)
            self.assertFalse(target.exists())

    def test_non_json_value_preserves_previous_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary)/"state.json"
            atomic_json(target,{"value":1})
            before = target.read_bytes()
            with self.assertRaises(ValueError):
                atomic_json(target,{"value":float("nan")})
            self.assertEqual(target.read_bytes(),before)


if __name__ == "__main__":
    unittest.main()
