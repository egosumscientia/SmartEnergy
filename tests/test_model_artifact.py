from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from filelock import FileLock, Timeout
from core.model_artifact import load_model_bundle, model_read_lock, save_model_bundle


class ModelArtifactTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "model.pkl"
        self.bundle = {"model": "fixture", "scaler": [1, 2], "features": ["x"]}
        save_model_bundle(self.bundle, self.path)

    def test_roundtrip_atomic_replacement_and_stable_lock(self):
        old_inode = self.path.stat().st_ino
        lock_inode = Path(str(self.path) + ".lock").stat().st_ino
        replacement = dict(self.bundle, model="replacement")
        save_model_bundle(replacement, self.path)
        self.assertNotEqual(old_inode, self.path.stat().st_ino)
        self.assertEqual(lock_inode, Path(str(self.path) + ".lock").stat().st_ino)
        self.assertEqual(load_model_bundle(self.path), replacement)

    def test_failed_write_preserves_original_and_cleans_temporary(self):
        before = self.path.read_bytes()
        with patch("core.model_artifact.joblib.dump", side_effect=RuntimeError("interrupted write")):
            with self.assertRaises(RuntimeError):
                save_model_bundle({"model": "broken"}, self.path)
        self.assertEqual(self.path.read_bytes(), before)
        self.assertEqual(list(self.path.parent.glob("*.tmp")), [])
        self.assertEqual(load_model_bundle(self.path), self.bundle)

    def test_readers_share_lock_and_block_writer(self):
        before = self.path.read_bytes()
        with model_read_lock(self.path):
            self.assertEqual(load_model_bundle(self.path, timeout=0), self.bundle)
            with self.assertRaises(Timeout):
                save_model_bundle({}, self.path, timeout=0)
        self.assertEqual(self.path.read_bytes(), before)

    def test_exclusive_writer_blocks_reader(self):
        with FileLock(str(self.path) + ".lock"):
            with self.assertRaises(Timeout):
                load_model_bundle(self.path, timeout=0)


if __name__ == "__main__":
    unittest.main()
