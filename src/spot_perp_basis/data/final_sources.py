"""Fixed-scope official raw inputs for Task 3; no statistical outcome access.

The new data-QA authority does not alter any development/model access guard.
All cache paths are relative to an explicitly supplied, versioned data root.
"""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from urllib.error import HTTPError

import pandas as pd

from .ingest import checksum_value, fetch


VERSION = "final-inputs-v1"
ASSETS = ("BTCUSDT", "ETHUSDT", "DOGEUSDT", "ADAUSDT")
START = pd.Timestamp("2023-01-01", tz="UTC")
END = pd.Timestamp("2026-01-01", tz="UTC")
WARMUP = pd.Timestamp("2022-12-01", tz="UTC")
MONTHS = tuple(str(m) for m in pd.period_range("2022-12", "2025-12", freq="M"))
SOURCES = ("spot", "usdm_perpetual", "funding")
BASE = "https://data.binance.vision/data/"


def digest(path):
    h = sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def json_bytes(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def atomic(path, body):
    """Immutable publication; interrupted temporary files are never valid inputs."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != body:
            raise ValueError(f"Refusing to overwrite different artifact: {path.name}")
        return
    temporary = path.with_name(path.name + ".partial")
    try:
        temporary.write_bytes(body)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


@dataclass(frozen=True, order=True)
class Source:
    symbol: str
    month: str
    source: str

    def __post_init__(self):
        if self.symbol not in ASSETS or self.month not in MONTHS or self.source not in SOURCES:
            raise ValueError("Outside authorized four-asset raw-input scope")

    @property
    def filename(self):
        kind = "fundingRate" if self.source == "funding" else "1m"
        return f"{self.symbol}-{kind}-{self.month}.zip"

    @property
    def url(self):
        if self.source == "funding":
            return f"{BASE}futures/um/monthly/fundingRate/{self.symbol}/{self.filename}"
        venue = "spot" if self.source == "spot" else "futures/um"
        return f"{BASE}{venue}/monthly/klines/{self.symbol}/1m/{self.filename}"

    @property
    def relative(self):
        return Path("raw") / self.source / self.symbol / self.month / self.filename

    @property
    def key(self):
        return f"{self.symbol}/{self.month}/{self.source}"


def requirements():
    return [Source(a, m, s) for a in ASSETS for m in MONTHS for s in SOURCES]


def candidates(reuse_root, specs):
    """Names/paths only first; do not open unrelated years, assets or results.

    Existing manifests are not required. Ambiguous duplicate content is a
    conflict, never an invitation to silently choose a preferred vintage.
    """
    result = {s.key: [] for s in specs}
    if reuse_root is None:
        return result
    lookup = {(s.filename, s.source): s for s in specs}
    for p in Path(reuse_root).rglob("*.zip"):
        for source in SOURCES:
            if source in p.parts and (p.name, source) in lookup:
                result[lookup[p.name, source].key].append(p)
    return result


def check_source(root, spec):
    path = Path(root) / spec.relative
    receipt = json.loads(path.with_suffix(".receipt.json").read_text())
    check = path.with_name(path.name + ".CHECKSUM")
    if (receipt["key"] != spec.key or receipt["url"] != spec.url
            or receipt["path"] != str(spec.relative)
            or checksum_value(check.read_bytes(), spec.filename) != receipt["sha256"]
            or digest(check) != receipt["checksum_sha256"]
            or digest(path) != receipt["sha256"] or path.stat().st_size != receipt["bytes"]):
        raise ValueError(f"Corrupt or wrong-identity source: {spec.key}")
    return receipt


def acquire_one(root, spec, existing=(), *, offline=False, request=fetch):
    root = Path(root)
    path = root / spec.relative
    receipt_path = path.with_suffix(".receipt.json")
    if receipt_path.exists():
        return check_source(root, spec)
    if offline:
        raise FileNotFoundError(f"Unverified offline source: {spec.key}")
    check_body, check_http = request(spec.url + ".CHECKSUM")
    expected = checksum_value(check_body, spec.filename)
    pool = list(existing) + ([path] if path.exists() else [])
    by_hash = {}
    for item in pool:
        by_hash.setdefault(digest(item), []).append(item)
    if set(by_hash) - {expected}:
        # Original conflicts remain untouched; quarantine records their identity.
        conflict = {"key": spec.key, "official_sha256": expected,
                    "conflicting_sha256": sorted(set(by_hash) - {expected}),
                    "candidate_count": len(pool), "action": "originals_retained_no_replacement"}
        atomic(root / "conflicts" / (spec.key.replace("/", "_") + ".json"), json_bytes(conflict))
        raise ValueError(f"Source conflict quarantined: {spec.key}")
    if expected in by_hash:
        body = by_hash[expected][0].read_bytes()
        http = {"url": spec.url, "acquisition": "reused_verified", "duplicate_copies": len(pool)}
    else:
        body, http = request(spec.url)
        http["acquisition"] = "downloaded_verified"
    if sha256(body).hexdigest() != expected:
        raise ValueError(f"Official checksum mismatch: {spec.key}")
    atomic(path, body)
    atomic(path.with_name(path.name + ".CHECKSUM"), check_body)
    receipt = {**http, "key": spec.key, "url": spec.url, "symbol": spec.symbol,
               "month": spec.month, "source": spec.source, "path": str(spec.relative),
               "sha256": expected, "bytes": len(body), "official_checksum_verified": True,
               "checksum_sha256": sha256(check_body).hexdigest(), "checksum_http": check_http,
               "schema_version": VERSION}
    atomic(receipt_path, json_bytes(receipt))
    return check_source(root, spec)


EFFR_URL = ("https://markets.newyorkfed.org/api/rates/unsecured/effr/search.json"
            "?startDate=2022-12-01&endDate=2025-12-31")


def effr_table(raw):
    """QA-only extension of the frozen EFFR conversion, without model guards."""
    f = pd.DataFrame(raw["refRates"])
    if f.empty or not f.type.eq("EFFR").all():
        raise ValueError("Require official EFFR records")
    times = pd.DatetimeIndex(pd.to_datetime(f.effectiveDate, utc=True)).as_unit("ns")
    values = pd.to_numeric(f.percentRate, errors="raise").to_numpy(float)
    import numpy as np
    if (times.hasnans or times.has_duplicates or (times < WARMUP).any() or (times >= END).any()
            or (times != times.normalize()).any() or not np.isfinite(values).all()):
        raise ValueError("Invalid/duplicate/out-of-scope EFFR effective records")
    return pd.DataFrame({"observation_time_utc": times, "reported_percent": values,
                         "annual_rate": values / 100,
                         "available_at_utc": times + pd.Timedelta(days=7),
                         "source": "NYFed_EFFR_current_vintage_percent_ACT365_proxy"}
                        ).sort_values("available_at_utc").reset_index(drop=True)


def acquire_effr(root, *, offline=False, request=fetch):
    path = Path(root) / "raw/effr/2022-12_2025-12.json"
    rp = path.with_suffix(".receipt.json")
    if rp.exists():
        r = json.loads(rp.read_text())
        if r["url"] != EFFR_URL or digest(path) != r["sha256"] or path.stat().st_size != r["bytes"]:
            raise ValueError("Corrupt EFFR source")
        frame = effr_table(json.loads(path.read_text()))
        if (frame.observation_time_utc.min() > START - pd.Timedelta(days=7)
                or frame.observation_time_utc.max() < END - pd.Timedelta(days=7)):
            raise ValueError("EFFR lacks boundary source coverage")
        return r
    if offline:
        raise FileNotFoundError("Missing verified offline EFFR source")
    body, http = request(EFFR_URL)
    frame = effr_table(json.loads(body))
    if frame.observation_time_utc.min() > START - pd.Timedelta(days=7):
        raise ValueError("EFFR lacks causal boundary lookback")
    if frame.observation_time_utc.max() < END - pd.Timedelta(days=7):
        raise ValueError("EFFR source history unexpectedly truncated")
    atomic(path, body)
    r = {**http, "url": EFFR_URL, "path": str(path.relative_to(root)), "bytes": len(body),
         "sha256": sha256(body).hexdigest(), "records": len(frame), "official_checksum": None,
         "validation": "official_HTTPS_current_vintage_no_publisher_checksum",
         "schema_version": VERSION}
    atomic(rp, json_bytes(r))
    return r


def inventory(root, *, acquire=False, offline=False, reuse_root=None, workers=4):
    """Persist exact missing/failing partition identities even after partial success."""
    specs = requirements()
    cache = candidates(reuse_root, specs) if acquire and not offline else {}
    initial = [{"key": s.key, "status": "verified_cached" if (Path(root)/s.relative.with_suffix(".receipt.json")).exists()
                else "existing_unverified" if cache.get(s.key) else "missing"} for s in specs]
    initial_path = Path(root) / "inventory_before.json"
    if not initial_path.exists():
        atomic(initial_path, json_bytes(initial))
    def job(s):
        try:
            r = acquire_one(root, s, cache.get(s.key, ()), offline=offline or not acquire)
            return {"key": s.key, "status": r.get("acquisition", "verified_cached"), "receipt": r}
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as e:
            return {"key": s.key, "status": "unavailable_upstream" if isinstance(e, HTTPError) and e.code == 404
                    else "corrupt_or_conflict" if isinstance(e, ValueError) else "missing_or_transfer_failure",
                    "error": f"{type(e).__name__}: {e}"}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(job, specs))
    try:
        effr = {"status": "verified", "receipt": acquire_effr(root, offline=offline or not acquire)}
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as e:
        effr = {"status": "missing_or_invalid", "error": f"{type(e).__name__}: {e}"}
    report = {"schema_version": VERSION, "files": rows, "effr": effr,
              "complete": all("receipt" in r for r in rows) and "receipt" in effr}
    # Inventory is resumable progress, unlike immutable sources/build receipts.
    p = Path(root) / "inventory.json"
    body = json_bytes(report)
    if not p.exists() or p.read_bytes() != body:
        temporary = p.with_suffix(".json.partial")
        try:
            temporary.write_bytes(body); temporary.replace(p)
        finally:
            temporary.unlink(missing_ok=True)
    return report
