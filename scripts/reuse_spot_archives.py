"""Copy verified legacy spot artifacts, or recheck the copy without the old repo.

This performs byte-integrity checks only. It never opens kline CSVs or computes
price statistics, labels, forecasting scores or trading outcomes.
"""

import argparse
import json
from pathlib import Path
import re
import shutil

from spot_perp_basis.data.ingest import checksum_value
from spot_perp_basis.artifacts import ROOT, file_sha256, utc_now, write_json

INVENTORY = ROOT / "manifests/reused_spot_archives.json"
SPOT_URL = "https://data.binance.vision/data/spot/monthly/klines"


def within(root, relative):
    """Reject absolute paths, traversal and symlinks escaping an artifact root."""
    relative = Path(relative)
    if relative.is_absolute():
        raise ValueError(f"Expected a relative artifact path: {relative}")
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"Artifact escapes its root: {relative}")
    return path


def copy_verified(source, destination, expected_sha):
    if file_sha256(source) != expected_sha:
        raise ValueError(f"Source checksum mismatch: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        temporary = destination.with_suffix(destination.suffix + ".partial")
        shutil.copy2(source, temporary)
        if file_sha256(temporary) != expected_sha:
            raise ValueError(f"Copied checksum mismatch: {temporary}")
        temporary.replace(destination)
    if file_sha256(destination) != expected_sha:
        raise ValueError(f"Destination checksum mismatch: {destination}")


def verify_inventory(root, inventory):
    for row in inventory["files"]:
        path = within(root, row["path"])
        check = within(root, row["checksum_path"])
        if file_sha256(path) != row["sha256"] or path.stat().st_size != row["bytes"]:
            raise ValueError(f"Local archive mismatch: {path}")
        if checksum_value(check.read_bytes(), row["filename"]) != row["sha256"]:
            raise ValueError(f"Local checksum mismatch: {check}")
    for row in inventory["sidecars"]:
        if file_sha256(within(root, row["path"])) != row["sha256"]:
            raise ValueError(f"Local sidecar mismatch: {row['path']}")


def import_archives(source_root):
    source_root = source_root.resolve()
    existing = json.loads(INVENTORY.read_text()) if INVENTORY.exists() else None
    raw = source_root / "data/raw/binance"
    manifest_path = within(source_root, (raw / "LATEST_full").read_text().strip())
    source_manifest = json.loads(manifest_path.read_text())
    if existing and file_sha256(manifest_path) != existing["source_manifest_sha256"]:
        raise ValueError("Source manifest differs from the recorded import")
    rows, sidecars = [], []

    def copy_sidecar(source, destination):
        digest = file_sha256(source)
        copy_verified(source, destination, digest)
        sidecars.append({"path": str(destination.relative_to(ROOT)), "sha256": digest})

    for key, info in sorted(source_manifest["files"].items()):
        symbol, month = info["symbol"], info["month"]
        if symbol not in {"BTCUSDT", "ETHUSDT"} or not re.fullmatch(r"\d{4}-(0[1-9]|1[0-2])", month):
            raise ValueError(f"Unexpected symbol or month: {key}")
        filename = f"{symbol}-1m-{month}.zip"
        url = f"{SPOT_URL}/{symbol}/1m/{filename}"
        if key != f"{symbol}/{month}" or info["filename"] != filename or info["url"] != url:
            raise ValueError(f"Unexpected spot archive identity: {key}")
        if info["checksum_metadata"]["url"] != url + ".CHECKSUM":
            raise ValueError(f"Unexpected checksum source: {key}")
        source = within(source_root, info["path"])
        source_check = within(source_root, info["checksum_path"])
        digest = checksum_value(source_check.read_bytes(), filename)
        if digest != info["sha256"] or source.stat().st_size != info["bytes"]:
            raise ValueError(f"Legacy receipt mismatch: {key}")
        metadata = source.parent / "metadata.json"
        if json.loads(metadata.read_text()) != info:
            raise ValueError(f"Legacy metadata mismatch: {key}")
        folder = ROOT / "data/raw/binance/spot" / symbol / month / digest
        destination = folder / filename
        copy_verified(source, destination, digest)
        copy_sidecar(source_check, folder / source_check.name)
        copy_sidecar(metadata, folder / "metadata.json")
        rows.append({
            "market": "spot", "symbol": symbol, "month": month, "interval": "1m",
            "filename": filename, "sha256": digest, "bytes": info["bytes"],
            "url": url, "checksum_url": url + ".CHECKSUM",
            "original_downloaded_at_utc": info["downloaded_at_utc"],
            "path": str(destination.relative_to(ROOT)),
            "checksum_path": str((folder / source_check.name).relative_to(ROOT)),
            "metadata_path": str((folder / "metadata.json").relative_to(ROOT)),
            "timestamp_unit": "not_inspected_at_phase0",
        })
    for info in source_manifest["documentation"].values():
        source = within(source_root, info["path"])
        if file_sha256(source) != info["sha256"]:
            raise ValueError(f"Documentation checksum mismatch: {source}")
        copy_sidecar(source, ROOT / info["path"])
    copy_sidecar(manifest_path, ROOT / "data/legacy" / manifest_path.name)
    for name in ("index.json", "LATEST_full", "LATEST_smoke"):
        copy_sidecar(raw / name, ROOT / "data/legacy" / name)
    inventory = {
        "schema_version": "reused-spot-inventory-v1", "imported_at_utc": utc_now(),
        "source_repository": str(source_root),
        "source_manifest_sha256": file_sha256(manifest_path),
        "verification": "SHA-256 and receipt identity only; kline contents not parsed",
        "metadata_path_semantics": "Copied legacy receipts retain original relative paths; use this inventory for current paths",
        "files": rows, "sidecars": sidecars,
    }
    verify_inventory(ROOT, inventory)
    if existing:
        if rows != existing["files"] or sidecars != existing["sidecars"]:
            raise ValueError("Restored artifacts differ from the recorded import")
        return existing  # Keep the original provenance receipt on restore/recheck.
    write_json(INVENTORY, inventory)
    return inventory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--source", type=Path, help="Existing volatility repository (read only)")
    group.add_argument("--verify", action="store_true", help="Recheck local files without the old repository")
    args = parser.parse_args()
    if args.verify:
        inventory = json.loads(INVENTORY.read_text())
        verify_inventory(ROOT, inventory)
    else:
        inventory = import_archives(args.source)
    print(f"Verified {len(inventory['files'])} spot archives, "
          f"{sum(row['bytes'] for row in inventory['files']):,} archive bytes, "
          f"{len(inventory['sidecars'])} provenance sidecars")


if __name__ == "__main__":
    main()
