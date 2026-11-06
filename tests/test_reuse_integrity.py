"""Negative checks for artifacts transferred across the repository boundary."""

from hashlib import sha256
import importlib.util
from pathlib import Path
import tempfile
import unittest

from spot_perp_basis.data.ingest import checksum_value
from spot_perp_basis.artifacts import ROOT

spec = importlib.util.spec_from_file_location("reuse", ROOT / "scripts/reuse_spot_archives.py")
reuse = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reuse)


class TransferIntegrityTests(unittest.TestCase):
    def test_official_checksum_rejects_wrong_filename_and_malformed_hash(self):
        digest = "a" * 64
        self.assertEqual(checksum_value(f"{digest.upper()} *bars.zip\n".encode(), "bars.zip"), digest)
        for body in (f"{digest} wrong.zip".encode(), b"abcd bars.zip", f"{digest} bars.zip extra".encode()):
            with self.subTest(body=body), self.assertRaises(ValueError):
                checksum_value(body, "bars.zip")

    def test_corrupt_source_is_rejected_before_destination_is_created(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, destination = root / "source", root / "destination"
            source.write_bytes(b"corrupt source")
            with self.assertRaisesRegex(ValueError, "Source checksum"):
                reuse.copy_verified(source, destination, sha256(b"expected").hexdigest())
            self.assertFalse(destination.exists())

    def test_independent_copy_and_subsequent_tampering(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, destination = root / "source", root / "copied.zip"
            source.write_bytes(b"original artifact")
            digest = sha256(source.read_bytes()).hexdigest()
            reuse.copy_verified(source, destination, digest)
            self.assertNotEqual(source.stat().st_ino, destination.stat().st_ino)
            check = root / "copied.zip.CHECKSUM"
            check.write_text(f"{digest} copied.zip\n")
            inventory = {"files": [{"path": destination.name, "checksum_path": check.name,
                                    "filename": destination.name, "sha256": digest,
                                    "bytes": destination.stat().st_size}], "sidecars": []}
            reuse.verify_inventory(root, inventory)
            source.unlink()
            reuse.verify_inventory(root, inventory)
            destination.write_bytes(b"changed bytes")
            with self.assertRaisesRegex(ValueError, "Local archive mismatch"):
                reuse.verify_inventory(root, inventory)

    def test_paths_and_symlinks_cannot_escape_source(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "artifacts"
            root.mkdir()
            (root / "outside").symlink_to(root.parent)
            for relative in ("../secret", "/tmp/secret", "outside/secret"):
                with self.subTest(relative=relative), self.assertRaises(ValueError):
                    reuse.within(root, relative)


if __name__ == "__main__":
    unittest.main()
