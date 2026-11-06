"""Frozen acquisition receipts must survive restores and reject changed artifacts."""

from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from spot_perp_basis.artifacts import ROOT

spec = importlib.util.spec_from_file_location("download_pilot", ROOT / "scripts/download_pilot.py")
download = importlib.util.module_from_spec(spec)
spec.loader.exec_module(download)


class AcquisitionReceiptTests(unittest.TestCase):
    def exercise(self, remote_body, existing_body=None):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "receipt.json"
            receipt = {"markets": {}, "documents": {"schema": {
                "url": "https://example.invalid/schema", "path": "schema.md",
                "sha256": sha256(b"frozen document").hexdigest(),
            }}}
            manifest.write_text(json.dumps(receipt))
            original = manifest.read_bytes()
            path = root / "schema.md"
            if existing_body is not None:
                path.write_bytes(existing_body)
            with patch.object(download, "ROOT", root), patch.object(download, "MANIFEST", manifest), \
                    patch.object(download, "fetch", return_value=(remote_body, {})) as fetch:
                if remote_body != b"frozen document" or existing_body == b"tampered":
                    with self.assertRaises(ValueError):
                        download.main()
                else:
                    download.main()
                    self.assertEqual(path.read_bytes(), b"frozen document")
                self.assertEqual(manifest.read_bytes(), original)
                if existing_body is not None:
                    fetch.assert_not_called()
                elif remote_body != b"frozen document":
                    self.assertFalse(path.exists())

    def test_missing_file_is_restored_without_changing_receipt(self):
        self.exercise(b"frozen document")

    def test_changed_remote_file_is_rejected_before_writing(self):
        self.exercise(b"changed remote")

    def test_corrupt_existing_cache_is_rejected_without_silent_repair(self):
        self.exercise(b"frozen document", existing_body=b"tampered")


if __name__ == "__main__":
    unittest.main()
