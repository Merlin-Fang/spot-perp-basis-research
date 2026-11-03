"""Generic provenance helpers reused from volatility_prediction/cryptovol/common.py."""

from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def load_config(path="configs/experiment.yaml"):
    """Read a YAML file; relative paths are resolved from the repository root."""
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path
    return yaml.safe_load(path.read_text())


def file_sha256(path):
    """Hash large artifacts without reading the entire file into memory."""
    hasher = sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def write_json(path, value):
    """Write strict JSON atomically so interrupted writes cannot replace a receipt."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)
