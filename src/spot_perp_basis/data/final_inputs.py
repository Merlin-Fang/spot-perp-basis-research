"""Canonical raw-input tables and structural checks, never survival outcomes.

Only pure source parsing/validity and past-availability utilities are reused.
Existing development guards on fitting, feature construction and scoring stay
unchanged. Tables retain the whole minute clock, including invalid/missing rows.
"""

from contextlib import contextmanager
import io
import json
from pathlib import Path
import sys
import threading
from zipfile import ZipFile

import numpy as np
import pandas as pd

from .final_sources import (ASSETS, MONTHS, START, END, WARMUP, VERSION, Source,
                            atomic, check_source, digest, effr_table, json_bytes)
from .funding import join_funding, prepare_funding
from .ingest import COLUMNS, parse_funding_archive, parse_klines
from .market import PRICE_COLUMNS, MINUTE, audit_market, intervals


ROOT = Path(__file__).resolve().parents[3]
CONFIG = "configs/final_data_v1.json"
CODE = ("src/spot_perp_basis/data/final_sources.py", "src/spot_perp_basis/data/final_inputs.py",
        "src/spot_perp_basis/data/ingest.py", "src/spot_perp_basis/data/funding.py",
        "src/spot_perp_basis/data/market.py", "src/spot_perp_basis/data/windows.py",
        "scripts/prepare_final_data.py", CONFIG, "requirements-lock.txt",
        "docs/final_evaluation_protocol.md", "configs/final_evaluation.yaml")
FORBIDDEN = ("spot_perp_basis.models", "spot_perp_basis.training", "spot_perp_basis.evaluation",
             "spot_perp_basis.backtest", "spot_perp_basis.theory", "spot_perp_basis.data.labels",
             "spot_perp_basis.data.features", "spot_perp_basis.economics")


@contextmanager
def structural_only():
    """Fail on accidental model/label/scoring/finance calls in this entry point."""
    def guard(frame, event, arg):
        if event == "call":
            name = frame.f_globals.get("__name__", "")
            if any(name == p or name.startswith(p + ".") for p in FORBIDDEN):
                raise PermissionError("Task 3 authorizes raw structural QA only: " + name)
    old, threaded = sys.getprofile(), threading.getprofile()
    sys.setprofile(guard); threading.setprofile(guard)
    try:
        yield
    finally:
        sys.setprofile(old); threading.setprofile(threaded)


def identity():
    cfg = json.loads((ROOT / CONFIG).read_text())
    if (cfg["assets"] != list(ASSETS) or cfg["start"] != START.isoformat()
            or cfg["end_exclusive"] != END.isoformat() or cfg["schema_version"] != VERSION
            or cfg["authorization"] != "raw_inputs_and_structural_QA_only"
            or cfg["evaluation_authorized"] is not False):
        raise ValueError("Wrong data-QA scope/version")
    return {"schema_version": VERSION, "implementation": {n: digest(ROOT / n) for n in CODE}}


def safe_path(root, relative):
    p = (Path(root) / relative).resolve()
    if not p.is_relative_to(Path(root).resolve()):
        raise ValueError("Data artifact escapes the explicit data root")
    return p


def save_frame(root, relative, frame):
    path = safe_path(root, relative)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".partial")
    try:
        frame.to_parquet(tmp, compression="zstd", index=True)
        if path.exists():
            if digest(path) != digest(tmp):
                raise ValueError("Canonical table differs; use a reviewed new version")
        else:
            tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)
    return {"path": str(path.relative_to(root)), "sha256": digest(path),
            "rows": len(frame), "columns": list(frame.columns)}


def causal_effr(clock, rates):
    """Frozen backward-asof convention, scoped to raw final-period QA inputs."""
    clock = pd.DatetimeIndex(clock).as_unit("ns")
    if (clock.tz is None or clock.hasnans or clock.has_duplicates or not clock.is_monotonic_increasing
            or (clock < START).any() or (clock >= END).any()):
        raise ValueError("Invalid final input decision clock")
    av = pd.DatetimeIndex(rates.available_at_utc).as_unit("ns")
    if (av.has_duplicates or not av.is_monotonic_increasing
            or not rates.available_at_utc.eq(rates.observation_time_utc + pd.Timedelta(days=7)).all()):
        raise ValueError("Invalid EFFR availability")
    ids = np.searchsorted(av.asi8, clock.asi8, side="right") - 1
    known = ids >= 0
    out = pd.DataFrame(index=clock.rename("decision_time_utc"))
    out["effr_annual_rate"] = np.where(known, rates.annual_rate.to_numpy()[np.maximum(ids, 0)], np.nan)
    for col in ("available_at_utc", "observation_time_utc"):
        values = pd.Series(pd.NaT, index=clock, dtype="datetime64[ns, UTC]")
        values.loc[known] = pd.DatetimeIndex(rates[col])[ids[known]]
        out["effr_" + col] = values
    out["zero_rate_annual_rate"] = 0.0
    return out


def funding_inputs(root, symbol):
    parts, audits = [], []
    for month in MONTHS:
        s = Source(symbol, month, "funding"); r = check_source(root, s)
        f, a = parse_funding_archive(Path(root)/s.relative, symbol, month, r["sha256"])
        parts.append(f); audits.append(a)
    events = prepare_funding(pd.concat(parts, ignore_index=True), lag_minutes=1)
    events["interval_change_flag"] = events.funding_interval_hours.ne(events.funding_interval_hours.shift()) & events.index.to_series().gt(0)
    events["interval_mismatch_flag"] = (events.observed_gap_hours - events.funding_interval_hours).abs().gt(1/3600)
    # All authorized inputs are after these assets' established public launch.
    events["quarantined"] = False
    events["eligible_for_causal_use"] = events.available_at_utc.lt(END)
    events["is_warmup"] = events.settlement_time_utc.lt(START)
    events.index.name = "event_row"
    return events, audits


def market_month(root, symbol, month, previous=None):
    previous = previous or {}
    frames, audits = {}, []
    for source, venue in (("spot", "spot"), ("usdm_perpetual", "perp")):
        s = Source(symbol, month, source); r = check_source(root, s)
        bars, audit = parse_klines(Path(root)/s.relative, symbol, month, source, r["sha256"])
        bars, audit = audit_market(bars, audit, previous.get(venue))
        frames[venue] = bars; audits.append(audit)
    spot, perp = frames["spot"], frames["perp"]
    grid = spot.index
    if not grid.equals(perp.index):
        raise ValueError("Cross-market source clocks differ")
    out = pd.DataFrame(index=(grid + MINUTE).rename("decision_time_utc"))
    out["bar_open_time_utc"] = grid
    out["symbol"] = symbol
    for venue, bars in frames.items():
        cols = PRICE_COLUMNS + ["ignore", "observed", "valid", "bad_price", "bad_activity",
                                "bad_close_time", "quote_price_flag", "large_return_flag", "unchanged_close"]
        for col in cols:
            out[venue + "_" + col] = bars[col].to_numpy()
        unit = next(a["timestamp_unit"] for a in audits if a["market"] == ("spot" if venue == "spot" else "usdm_perpetual"))
        out[venue + "_close_time_utc"] = pd.to_datetime(bars.close_time.to_numpy(), unit=unit, utc=True, errors="coerce").as_unit("ns")
    out["pair_observed"] = out.spot_observed & out.perp_observed
    out["pair_valid"] = out.spot_valid & out.perp_valid
    out["pair_active"] = out.pair_valid & out.spot_volume.gt(0) & out.perp_volume.gt(0) & out.spot_trades.gt(0) & out.perp_trades.gt(0)
    # Same pre-existing 100 bps integrity flag; it never deletes observations.
    out["cross_market_price_flag"] = out.pair_valid & (np.log(out.perp_close / out.spot_close).abs().gt(.01))
    gaps = {v: intervals(grid, ~f.observed) for v, f in frames.items()}
    return out.loc[(out.index >= START) & (out.index < END)], audits, {v: f.iloc[-1] for v, f in frames.items()}, gaps


def validate_clock(frame, symbol, expected):
    if (not frame.index.equals(expected.rename("decision_time_utc")) or frame.index.has_duplicates
            or not frame.symbol.eq(symbol).all()
            or not (pd.DatetimeIndex(frame.bar_open_time_utc) + MINUTE).equals(frame.index)):
        raise ValueError("Dropped/duplicated/wrong-symbol or noncausal completed-bar clock")


def verify_build(root, receipt):
    if receipt["identity"] != identity() or receipt["inventory_sha256"] != digest(Path(root)/"inventory.json"):
        raise ValueError("Completed data provenance differs; never silently overwrite")
    for a in receipt["artifacts"]:
        if digest(safe_path(root, a["path"])) != a["sha256"]:
            raise ValueError("Corrupt canonical input table: " + a["path"])


def build(root):
    root = Path(root)
    inv = json.loads((root / "inventory.json").read_text())
    if not inv["complete"]:
        raise ValueError("Cannot build incomplete sources; see exact inventory failures")
    rp = root / "build_receipt.json"
    if rp.exists():
        r = json.loads(rp.read_text()); verify_build(root, r)
        return r
    ident = identity()
    rates = effr_table(json.loads((root / "raw/effr/2022-12_2025-12.json").read_text()))
    artifacts = [save_frame(root, "canonical/effr.parquet", rates)]
    market_audits, funding_audits, gaps = [], [], {}
    for symbol in ASSETS:
        events, audits = funding_inputs(root, symbol); funding_audits.extend(audits)
        artifacts.append(save_frame(root, f"canonical/funding/{symbol}.parquet", events))
        hourly, previous = [], {}
        for month in MONTHS:
            panel, audits, previous, missing = market_month(root, symbol, month, previous)
            market_audits.extend(audits); gaps[f"{symbol}/{month}"] = missing
            panel = join_funding(panel, events, maximum_age_hours=12)
            panel = panel.join(causal_effr(panel.index, rates), validate="one_to_one")
            artifacts.append(save_frame(root, f"canonical/minute/{symbol}/{month}.parquet", panel))
            hourly.append(panel.loc[panel.index == panel.index.floor("h")])
            print(f"Built {symbol} {month}: {len(panel)} minute input rows", flush=True)
        hours = pd.concat(hourly).sort_index()
        for year in (2023, 2024, 2025):
            h = hours.loc[hours.index.year == year]
            clock = pd.date_range(f"{year}-01-01", f"{year+1}-01-01", freq="h", inclusive="left", tz="UTC").as_unit("ns")
            validate_clock(h, symbol, clock)
            artifacts.append(save_frame(root, f"canonical/hourly/{symbol}/{year}.parquet", h))
    r = {"schema_version": VERSION, "identity": ident, "inventory_sha256": digest(root/"inventory.json"),
         "artifacts": artifacts, "market_audits": market_audits, "funding_audits": funding_audits,
         "missing_source_intervals": gaps, "minute_partition_key": "bar_open_month",
         "decision_interval": [START.isoformat(), END.isoformat()], "no_outcomes_constructed": True}
    atomic(rp, json_bytes(r)); return r


def independent_market_check(root, symbol, month, panel, audits):
    """Independently parse every raw row using pandas CSV, then compare all fields.

    This does not use the grid parser's numerical arrays or validity selection.
    It verifies that even invalid/inactive source observations were retained.
    """
    checked = 0
    for source, venue in (("spot", "spot"), ("usdm_perpetual", "perp")):
        spec = Source(symbol, month, source)
        with ZipFile(Path(root)/spec.relative) as z:
            body = z.read(spec.filename.replace(".zip", ".csv"))
        header = int(not body.splitlines()[0].split(b",")[0].isdigit())
        raw = pd.read_csv(io.BytesIO(body), header=None, skiprows=header, names=COLUMNS, float_precision="round_trip")
        unit = next(a["timestamp_unit"] for a in audits if a["market"] == source)
        raw.index = (pd.DatetimeIndex(pd.to_datetime(raw.open_time, unit=unit, utc=True)).as_unit("ns") + MINUTE).rename("decision_time_utc")
        raw = raw.loc[(raw.index >= START) & (raw.index < END)].sort_index()
        if raw.index.has_duplicates or not panel.loc[raw.index, venue+"_observed"].all():
            raise ValueError("Lost/duplicate source keys")
        for col in PRICE_COLUMNS + ["ignore"]:
            np.testing.assert_allclose(panel.loc[raw.index, venue+"_"+col], raw[col], rtol=1e-12, atol=1e-12, equal_nan=True)
        np.testing.assert_array_equal(panel.index[panel[venue+"_observed"]], raw.index)
        closes = pd.DatetimeIndex(pd.to_datetime(raw.close_time, unit=unit, utc=True)).as_unit("ns")
        np.testing.assert_array_equal(pd.DatetimeIndex(panel.loc[raw.index, venue+"_close_time_utc"]).asi8, closes.asi8)
        checked += len(raw)
    return checked


def qa(root):
    root = Path(root)
    receipt = json.loads((root/"build_receipt.json").read_text()); verify_build(root, receipt)
    if (root/"qa_receipt.json").exists():
        old = json.loads((root/"qa_receipt.json").read_text())
        if old["build_sha256"] != digest(root/"build_receipt.json"):
            raise ValueError("Changed QA provenance")
        return old
    cells, funding_summary, boundary_checks = [], [], []
    rates = pd.read_parquet(root/"canonical/effr.parquet")
    raw_rows, independent_funding_events = 0, 0
    for symbol in ASSETS:
        events = pd.read_parquet(root/f"canonical/funding/{symbol}.parquet")
        replay, _ = funding_inputs(root, symbol)
        pd.testing.assert_frame_equal(events, replay)
        for month in MONTHS:
            spec = Source(symbol, month, "funding")
            with ZipFile(root/spec.relative) as z:
                raw = pd.read_csv(io.BytesIO(z.read(spec.filename.replace(".zip", ".csv"))), float_precision="round_trip")
            # Actual archive convention was detected/validated per file by parser;
            # independently reconstruct event membership and every signed rate.
            unit = next(a["timestamp_unit"] for a in receipt["funding_audits"] if a["symbol"] == symbol and a["month"] == month)
            raw.index = pd.DatetimeIndex(pd.to_datetime(raw.calc_time, unit=unit, utc=True)).as_unit("ns")
            e = events.loc[events.source_month.eq(month)].set_index("settlement_time_utc")
            raw = raw.sort_index()
            np.testing.assert_array_equal(e.index.asi8, raw.index.asi8)
            np.testing.assert_allclose(e.funding_rate, raw.last_funding_rate, rtol=1e-12, atol=1e-16)
            np.testing.assert_array_equal(e.funding_interval_hours, raw.funding_interval_hours)
            independent_funding_events += len(raw)
        final_events = events.loc[~events.is_warmup]
        for year in (2023, 2024, 2025):
            e = final_events.loc[final_events.settlement_time_utc.dt.year == year]
            funding_summary.append({"symbol": symbol, "year": year, "events": len(e),
                "first": str(e.settlement_time_utc.min()), "last": str(e.settlement_time_utc.max()),
                "interval_hours": sorted(e.funding_interval_hours.unique().tolist()),
                "interval_changes": int(e.interval_change_flag.sum()), "cadence_gaps": int(e.cadence_gap_flag.sum()),
                "interval_mismatches": int(e.interval_mismatch_flag.sum()),
                "max_observed_gap_hours": float(e.observed_gap_hours.max()),
                "duplicate_settlements": int(e.settlement_time_utc.duplicated().sum())})
        hourly = []
        for month in MONTHS:
            p = pd.read_parquet(root/f"canonical/minute/{symbol}/{month}.parquet")
            start = max(START, pd.Timestamp(month+"-01", tz="UTC") + MINUTE)
            end = min(END, pd.Timestamp(month+"-01", tz="UTC") + pd.offsets.MonthBegin() + MINUTE)
            expected = pd.date_range(start, end, freq="min", inclusive="left").as_unit("ns")
            validate_clock(p, symbol, expected)
            audits = [a for a in receipt["market_audits"] if a["symbol"] == symbol and a["month"] == month]
            raw_rows += independent_market_check(root, symbol, month, p, audits)
            if not (p.pair_valid == (p.spot_valid & p.perp_valid)).all():
                raise ValueError("Wrong paired validity")
            active = p.pair_valid & p.spot_volume.gt(0) & p.perp_volume.gt(0) & p.spot_trades.gt(0) & p.perp_trades.gt(0)
            if not active.equals(p.pair_active):
                raise ValueError("Wrong paired activity")
            replay_join = join_funding(p[["symbol"]], events, maximum_age_hours=12)
            for col in replay_join.columns:
                pd.testing.assert_series_equal(p[col], replay_join[col])
            reff = pd.merge_asof(p[["symbol"]].reset_index(), rates,
                                left_on="decision_time_utc", right_on="available_at_utc",
                                direction="backward", allow_exact_matches=True)
            np.testing.assert_array_equal(p.effr_annual_rate, reff.annual_rate)
            for col in ("observation_time_utc", "available_at_utc"):
                np.testing.assert_array_equal(pd.DatetimeIndex(p["effr_"+col]).asi8, pd.DatetimeIndex(reff[col]).asi8)
            if not p.zero_rate_annual_rate.eq(0).all():
                raise ValueError("Wrong zero financing control")
            for year in sorted(set(p.index.year)):
                y = p.loc[p.index.year == year]
                cells.append({"symbol": symbol, "year": int(year), "source_month": month, "minutes": len(y),
                    "spot_missing": int((~y.spot_observed).sum()), "perp_missing": int((~y.perp_observed).sum()),
                    "spot_invalid": int((y.spot_observed & ~y.spot_valid).sum()),
                    "perp_invalid": int((y.perp_observed & ~y.perp_valid).sum()),
                    "paired_valid": int(y.pair_valid.sum()), "paired_active": int(y.pair_active.sum()),
                    "valid_inactive": int((y.pair_valid & ~y.pair_active).sum()),
                    "funding_unknown": int((~y.funding_known).sum()),
                    "effr_unknown": int(y.effr_annual_rate.isna().sum()),
                    "cross_market_flags": int(y.cross_market_price_flag.sum())})
            hourly.append(p.loc[p.index == p.index.floor("h")])
        h = pd.concat(hourly)
        for year in (2023, 2024, 2025):
            actual = pd.read_parquet(root/f"canonical/hourly/{symbol}/{year}.parquet")
            pd.testing.assert_frame_equal(actual, h.loc[h.index.year == year], check_freq=False)
            if actual.effr_annual_rate.isna().any():
                raise ValueError("Uncovered EFFR causal hourly inputs")
        for date in ("2023-01-01", "2024-01-01", "2024-02-28", "2024-02-29", "2024-03-01", "2025-01-01", "2025-12-31 23:00"):
            t = pd.Timestamp(date, tz="UTC")
            if h.loc[t, "bar_open_time_utc"] != t-MINUTE:
                raise ValueError("Annual/leap completed-minute boundary failed")
            boundary_checks.append({"symbol": symbol, "origin": t.isoformat(), "bar_open": (t-MINUTE).isoformat(),
                                    "pair_valid": bool(h.loc[t, "pair_valid"])})
    pd.testing.assert_frame_equal(rates, effr_table(json.loads((root/"raw/effr/2022-12_2025-12.json").read_text())))
    totals = pd.DataFrame(cells).groupby(["symbol", "year"]).sum(numeric_only=True).reset_index().to_dict("records")
    fatal = [r for r in totals if r["effr_unknown"] or r["minutes"] != (527040 if r["year"] == 2024 else 525600)]
    status = "FINAL_DATA_STRUCTURAL_QA_PARTIAL" if fatal else "FINAL_DATA_STRUCTURAL_QA_READY"
    r = {"schema_version": VERSION, "status": status, "build_sha256": digest(root/"build_receipt.json"),
         "totals": totals, "funding": funding_summary, "boundary_checks": boundary_checks,
         "independent_market_rows_checked": raw_rows, "duplicate_keys": 0,
         "independent_funding_events_checked": independent_funding_events,
         "terminal_cutoff": END.isoformat(), "no_2026_market_sources": True,
         "no_labels_models_forecasts_or_economic_outcomes": True,
         "effr": {"records": len(rates), "first_effective": str(rates.observation_time_utc.min()),
                  "last_effective": str(rates.observation_time_utc.max()), "duplicates": 0,
                  "max_effective_gap_days": float(rates.observation_time_utc.diff().dt.total_seconds().max()/86400),
                  "records_by_effective_year": {str(k): int(v) for k,v in rates.groupby(rates.observation_time_utc.dt.year).size().items()},
                  "weekend_effective_records": int((rates.observation_time_utc.dt.dayofweek>=5).sum()),
                  "percent_to_decimal_divisor": 100, "availability_lag_calendar_days": 7,
                  "official_publisher_checksum": None}, "failures": fatal}
    atomic(root/"qa_receipt.json", json_bytes(r)); return r
