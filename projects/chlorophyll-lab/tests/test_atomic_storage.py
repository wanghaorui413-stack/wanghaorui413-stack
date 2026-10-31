import tempfile
import unittest
from pathlib import Path
from chlorophyll.common import atomic_directory, write_json, read_json


class AtomicStorageTests(unittest.TestCase):
    def test_failure_can_be_retried_and_success_is_immutable(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "result"
            with self.assertRaises(RuntimeError):
                with atomic_directory(target) as stage:
                    (stage / "partial").write_text("incomplete")
                    raise RuntimeError("simulated interrupted operation")
            self.assertFalse(target.exists())
            with atomic_directory(target) as stage:
                write_json(stage / "manifest.json", {"complete": True})
            self.assertTrue(read_json(target / "manifest.json")["complete"])
            with self.assertRaises(FileExistsError):
                with atomic_directory(target):
                    pass

    def test_failed_json_encoding_preserves_previous_content(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            write_json(path, {"old": True})
            with self.assertRaises(ValueError):
                write_json(path, {"invalid": float("nan")})
            self.assertEqual(read_json(path), {"old": True})
