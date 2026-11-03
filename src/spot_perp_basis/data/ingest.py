"""Official archive acquisition, checksums and strict source-schema parsing.

No model fitting, labels or backtest is triggered by ingestion."""

import csv
import io
import json
import re
import time
import zipfile
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from zipfile import ZipFile

import numpy as np
import pandas as pd

from ..artifacts import ROOT, file_sha256, utc_now, write_json
from .market import MINUTE
from .windows import MINUTE_NS, ns_index


# Official checksum verification

def checksum_value(body, filename):
    parts = body.decode("utf-8").strip().split()
    if len(parts) != 2 or not re.fullmatch(r"[a-fA-F0-9]{64}", parts[0]):
        raise ValueError("Unexpected official SHA256 checksum format")
    if Path(parts[1].lstrip("*")).name != filename:
        raise ValueError("Checksum is for a different archive")
    return parts[0].lower()


# Archive download and immutable receipts

def fetch(url, attempts=3):
    for attempt in range(attempts):
        try:
            with urlopen(Request(url, headers={"User-Agent": "SpotPerpBasisAcademicResearch/0.2"}), timeout=25) as r:
                return r.read(), {"url": url, "final_url": r.url, "status": r.status,
                                  "retrieved_at_utc": utc_now(), "http_headers": dict(r.headers.items())}
        except HTTPError as error:
            if error.code not in {429, 500, 502, 503, 504} or attempt + 1 == attempts:
                raise
        except OSError:
            if attempt + 1 == attempts:
                raise
        time.sleep(2 ** attempt)


def store(body, path):
    path = Path(path)
    digest = sha256(body).hexdigest()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if file_sha256(path) != digest:
            raise ValueError(f"Refusing to overwrite different artifact: {path}")
    else:
        temporary = path.with_suffix(path.suffix + ".partial")
        temporary.write_bytes(body)
        temporary.replace(path)
    return {"path": str(path.relative_to(ROOT)), "sha256": digest, "bytes": len(body)}


def verify_entry(info):
    if file_sha256(ROOT / info["path"]) != info["sha256"]:
        raise ValueError(f"Corrupt archive: {info['path']}")
    check = ROOT / info["checksum_path"]
    if checksum_value(check.read_bytes(), info["filename"]) != info["sha256"]:
        raise ValueError(f"Checksum receipt differs: {check}")
    if "checksum_sha256" in info and file_sha256(check) != info["checksum_sha256"]:
        raise ValueError(f"Checksum receipt bytes changed: {check}")
    if (ROOT / info["path"]).stat().st_size != info["bytes"]:
        raise ValueError("Archive byte count differs")


def acquire_archive(kind, market, symbol, month, base_url, attempts=3):
    filename = f"{symbol}-1m-{month}.zip" if kind == "klines" else f"{symbol}-fundingRate-{month}.zip"
    url = f"{base_url}/{symbol}/1m/{filename}" if kind == "klines" else f"{base_url}/{symbol}/{filename}"
    receipt = ROOT / "data/raw/binance/phase2/receipts" / kind / market / symbol / f"{month}.json"
    if receipt.exists():
        info = json.loads(receipt.read_text())
        if info["url"] != url:
            raise ValueError("Cached receipt identity differs")
        # Restore a missing artifact against its frozen checksum; never refresh silently.
        for path, source_url, expected in ((info["path"], url, info["sha256"]),
                                           (info["checksum_path"], url + ".CHECKSUM", info["checksum_sha256"])):
            if not (ROOT / path).exists():
                body, _ = fetch(source_url, attempts)
                if sha256(body).hexdigest() != expected:
                    raise ValueError("Remote artifact changed from frozen receipt")
                store(body, ROOT / path)
        verify_entry(info)
        return info
    check_body, check_meta = fetch(url + ".CHECKSUM", attempts)
    expected = checksum_value(check_body, filename)
    category = market if kind == "klines" else "funding"
    folder = ROOT / "data/raw/binance" / category / symbol / month / expected
    path = folder / filename
    if path.exists():
        if file_sha256(path) != expected:
            raise ValueError("Corrupt content-addressed cache")
        archive_meta = {"url": url, "acquisition": "existing_content_addressed_copy"}
    else:
        body, archive_meta = fetch(url, attempts)
        if sha256(body).hexdigest() != expected:
            raise ValueError(f"Official checksum mismatch: {url}")
        store(body, path)
    check_info = store(check_body, folder / (filename + ".CHECKSUM"))
    info = {**archive_meta, "url": url, "kind": kind, "market": market, "symbol": symbol,
            "month": month, "interval": "1m" if kind == "klines" else "settlement_events",
            "filename": filename, "path": str(path.relative_to(ROOT)), "sha256": expected,
            "bytes": path.stat().st_size, "checksum_path": check_info["path"],
            "checksum_sha256": check_info["sha256"], "checksum_http": check_meta,
            "timestamp_unit": "pending_per_file_inspection"}
    write_json(receipt, info)
    return info


# Minute kline schema and timestamp normalization

COLUMNS = ["open_time", "open", "high", "low", "close", "volume", "close_time",
           "quote_volume", "trades", "taker_buy_volume", "taker_buy_quote_volume", "ignore"]


def parse_klines(path, symbol, month, market, expected_sha):
    """Detect timestamp units from each file; never deduplicate, fill or drop rows."""
    if market not in {"spot", "usdm_perpetual"}:
        raise ValueError("Unsupported market")
    if file_sha256(path) != expected_sha:
        raise ValueError("Archive checksum mismatch")
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if names != [f"{symbol}-1m-{month}.csv"]:
            raise ValueError(f"Unexpected ZIP contents: {names}")
        rows = list(csv.reader(io.StringIO(archive.read(names[0]).decode("utf-8-sig"))))
    header = None
    if rows and rows[0] and rows[0][0].lower() in {"open_time", "open time", "opentime"}:
        header = rows.pop(0)
        expected_header = COLUMNS.copy()
        # The official USD-M archive calls trade count 'count'.
        expected_header[8] = "count"
        if header not in (COLUMNS, expected_header):
            raise ValueError(f"Unexpected kline header: {header}")
    if not rows or any(len(row) != 12 for row in rows):
        raise ValueError("Empty archive or malformed 12-field row")
    raw = pd.DataFrame(rows, columns=COLUMNS)
    for name in ("open_time", "close_time"):
        if not raw[name].str.fullmatch(r"\d+").all():
            raise ValueError(f"Noninteger {name}")
    numeric = raw.apply(pd.to_numeric, errors="coerce")
    start = pd.Timestamp(month + "-01", tz="UTC")
    end = start + pd.offsets.MonthBegin()
    candidates = []
    for unit in ("s", "ms", "us", "ns"):
        dates = pd.DatetimeIndex(pd.to_datetime(numeric.open_time, unit=unit, utc=True, errors="coerce"))
        if dates.notna().all() and ((dates >= start) & (dates < end)).all():
            candidates.append((unit, dates.as_unit("ns")))
    if len(candidates) != 1:
        raise ValueError("Timestamp units mixed, ambiguous or outside requested month")
    unit, dates = candidates[0]
    if dates.duplicated().any():
        raise ValueError("Duplicate minute timestamps (including identical rows) are forbidden")
    if (dates.asi8 % MINUTE.value != 0).any():
        raise ValueError("Bar openings do not lie on UTC minute boundaries")
    was_sorted = dates.is_monotonic_increasing
    numeric.index = dates.rename("open_time_utc")
    numeric = numeric.sort_index()
    prices = numeric[["open", "high", "low", "close"]]
    activity = numeric[["volume", "quote_volume", "trades", "taker_buy_volume", "taker_buy_quote_volume"]]
    numeric["bad_price"] = (~np.isfinite(prices).all(axis=1) | ~prices.gt(0).all(axis=1)
                            | numeric.high.lt(numeric[["open", "close", "low"]].max(axis=1))
                            | numeric.low.gt(numeric[["open", "close", "high"]].min(axis=1)))
    numeric["bad_activity"] = (~np.isfinite(activity).all(axis=1) | ~activity.ge(0).all(axis=1)
                               | numeric.trades.mod(1).ne(0)
                               | numeric.taker_buy_volume.gt(numeric.volume + 1e-8)
                               | numeric.taker_buy_quote_volume.gt(numeric.quote_volume + 1e-6))
    step = {"s": 60, "ms": 60_000, "us": 60_000_000, "ns": 60_000_000_000}[unit]
    numeric["bad_close_time"] = numeric.close_time.ne(numeric.open_time + step - 1)
    numeric["valid"] = ~numeric[["bad_price", "bad_activity", "bad_close_time"]].any(axis=1)
    numeric["observed"] = True
    grid = pd.date_range(start, end, freq="min", inclusive="left", name="open_time_utc").as_unit("ns")
    bars = numeric.reindex(grid)
    for name in ("valid", "observed", "bad_price", "bad_activity", "bad_close_time"):
        bars[name] = bars[name].fillna(False).astype(bool)
    positive_volume = bars.valid & bars.volume.gt(0)
    implied_price = bars.quote_volume / bars.volume.where(bars.volume.gt(0))
    inconsistent_quote = positive_volume & ((implied_price < bars.low - 1e-4) | (implied_price > bars.high + 1e-4))
    stale_close = bars.valid & bars.valid.shift(1, fill_value=False) & bars.close.eq(bars.close.shift(1))
    audit = {
        "market": market, "symbol": symbol, "month": month, "sha256": expected_sha,
        "timestamp_unit": unit, "raw_header": header, "raw_first_row": rows[0], "raw_last_row": rows[-1],
        "raw_rows": len(rows), "input_was_sorted": was_sorted, "expected_minutes": len(grid),
        "observed_minutes": int(bars.observed.sum()), "missing_minutes": int((~bars.observed).sum()),
        "invalid_minutes": int((bars.observed & ~bars.valid).sum()),
        "bad_price": int(bars.bad_price.sum()), "bad_activity": int(bars.bad_activity.sum()),
        "bad_close_time": int(bars.bad_close_time.sum()),
        "zero_volume_minutes": int((bars.valid & bars.volume.eq(0)).sum()),
        "zero_trade_minutes": int((bars.valid & bars.trades.eq(0)).sum()),
        "unchanged_close_minutes": int(stale_close.sum()),
        "quote_over_base_outside_ohlc_minutes": int(inconsistent_quote.sum()),
        "first_open_utc": str(numeric.index[0]), "last_open_utc": str(numeric.index[-1]),
        "last_bar_available_at_utc": str(numeric.index[-1] + MINUTE),
        "close_time_offset_raw_units": step - 1,
    }
    return bars, audit


# Funding archive and API schemas

def parse_funding_archive(path, symbol, month, expected_sha):
    if file_sha256(path) != expected_sha:
        raise ValueError("Funding archive checksum mismatch")
    with zipfile.ZipFile(path) as z:
        if z.namelist() != [f"{symbol}-fundingRate-{month}.csv"]:
            raise ValueError("Unexpected funding ZIP members")
        frame = pd.read_csv(io.BytesIO(z.read(z.namelist()[0])), dtype=str)
    if list(frame) != ["calc_time", "funding_interval_hours", "last_funding_rate"] or frame.empty:
        raise ValueError("Unexpected funding schema or empty archive")
    if not frame.calc_time.str.fullmatch(r"\d+").all():
        raise ValueError("Noninteger funding timestamp")
    values = pd.to_numeric(frame.calc_time, errors="raise")
    start = pd.Timestamp(month + "-01", tz="UTC")
    end = start + pd.offsets.MonthBegin()
    units = []
    for unit in ("s", "ms", "us", "ns"):
        stamps = pd.DatetimeIndex(pd.to_datetime(values, unit=unit, utc=True, errors="coerce"))
        if stamps.notna().all() and ((stamps >= start) & (stamps < end)).all():
            units.append((unit, stamps.as_unit("ns")))
    if len(units) != 1:
        raise ValueError("Mixed/invalid funding timestamp units")
    unit, stamps = units[0]
    if stamps.duplicated().any():
        raise ValueError("Duplicate funding settlements")
    rates = pd.to_numeric(frame.last_funding_rate, errors="raise").to_numpy(float)
    hours = pd.to_numeric(frame.funding_interval_hours, errors="raise").to_numpy(float)
    if not np.isfinite(rates).all() or not np.isfinite(hours).all() or (hours <= 0).any():
        raise ValueError("Invalid funding rates/intervals")
    result = pd.DataFrame({"symbol": symbol, "settlement_time_utc": stamps,
                           "funding_rate": rates, "funding_interval_hours": hours,
                           "source_month": month, "source_sha256": expected_sha})
    return result.sort_values("settlement_time_utc"), {"symbol": symbol, "month": month,
        "timestamp_unit": unit, "events": len(result), "first_settlement_utc": str(stamps.min()),
        "last_settlement_utc": str(stamps.max()), "interval_hours_values": sorted(set(hours.tolist()))}


def funding_api_pages(symbol, start_ms, end_ms, request=fetch, limit=1000):
    """Documented inclusive-endpoint pagination, for environments with API access.

    Production acquisition uses checksum-protected public archives when the API
    is inaccessible. This adapter is tested synthetically, not asserted live-verified.
    """
    rows, cursor = [], start_ms
    while cursor <= end_ms:
        query = urlencode({"symbol": symbol, "startTime": cursor, "endTime": end_ms, "limit": limit})
        body, _ = request("https://fapi.binance.com/fapi/v1/fundingRate?" + query)
        page = json.loads(body)
        if not isinstance(page, list) or len(page) > limit:
            raise ValueError("Invalid funding API page")
        if not page:
            break
        times = [int(row["fundingTime"]) for row in page]
        if times != sorted(set(times)) or times[0] < cursor or times[-1] > end_ms:
            raise ValueError("Funding pagination overlap, duplicate or non-advancing page")
        if any(row["symbol"] != symbol for row in page):
            raise ValueError("Wrong funding API symbol")
        rows.extend(page)
        cursor = times[-1] + 1
    return rows


# Mark, index and premium reference schemas

def parse_reference(body, kind, month):
    if kind not in {"markPriceKlines", "indexPriceKlines", "premiumIndexKlines"}:
        raise ValueError("Unsupported reference kind")
    with ZipFile(BytesIO(body)) as archive:
        names = archive.namelist()
        if len(names) != 1 or not names[0].endswith(".csv"):
            raise ValueError("Require one CSV")
        raw = pd.read_csv(archive.open(names[0]), header=None, dtype=str)
    if not str(raw.iloc[0, 0]).isdigit():
        raw = raw.iloc[1:].reset_index(drop=True)
    if raw.shape[1] != 12:
        raise ValueError("Reference kline must have twelve fields")
    raw = raw.apply(pd.to_numeric, errors="raise")
    start = pd.Timestamp(month + "-01", tz="UTC")
    end = start + pd.offsets.MonthBegin(1)
    units = []
    for unit in ("s", "ms", "us", "ns"):
        try:
            times = pd.to_datetime(raw[0], unit=unit, utc=True)
            if ((times >= start) & (times < end)).all():
                units.append(unit)
        except (ValueError, OverflowError):
            pass
    if len(units) != 1:
        raise ValueError("Ambiguous timestamp units")
    times = pd.DatetimeIndex(pd.to_datetime(raw[0], unit=units[0], utc=True))
    close = pd.DatetimeIndex(pd.to_datetime(raw[6], unit=units[0], utc=True))
    t = ns_index(times)
    tick = {"s": 10**9, "ms": 10**6, "us": 1000, "ns": 1}[units[0]]
    if (np.any(np.diff(t) <= 0) or np.any(t % MINUTE_NS)
            or np.any(ns_index(close) != t + MINUTE_NS - tick)):
        raise ValueError("Invalid reference time grid/close timestamp")
    values = raw[[1, 2, 3, 4]].to_numpy(float)
    if (not np.isfinite(values).all() or np.any(values[:, 1] < values[:, [0, 2, 3]].max(axis=1))
            or np.any(values[:, 2] > values[:, [0, 1, 3]].min(axis=1))):
        raise ValueError("Invalid reference OHLC")
    if kind != "premiumIndexKlines" and np.any(values <= 0):
        raise ValueError("Nonpositive reference price")
    frame = pd.DataFrame(values, columns=["open", "high", "low", "close"])
    frame["bar_open_time_utc"] = times
    frame["observation_time_utc"] = times + pd.Timedelta(minutes=1)
    frame["feature_available_at_utc"] = times + pd.Timedelta(minutes=2)
    expected = int((end-start)/pd.Timedelta(minutes=1))
    audit = {"kind": kind, "month": month, "timestamp_unit": units[0], "rows": len(frame),
             "expected_minutes": expected, "missing_minutes": expected-len(frame),
             "first_open": str(times[0]), "last_open": str(times[-1]),
             "negative_close_count": int((values[:, 3] < 0).sum()),
             "zero_close_count": int((values[:, 3] == 0).sum()),
             "placeholder_unique_counts": {str(c): int(raw[c].nunique()) for c in [5, 7, 8, 9, 10, 11]},
             "activity_fields_are_observed_trades": False}
    return frame, audit
