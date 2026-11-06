"""Bounded read-only final/development input catalog and annual readiness plan."""

import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yaml

from .final_inputs import ROOT, structural_only, safe_path
from .final_sources import ASSETS, START, END, Source, check_source, digest
from .ingest import checksum_value
from .forecast_inputs import (RAW, FUNDING, REFERENCE, FINANCING, HISTORY_START,
                              adapt_hourly, observation_masks, reference_history,
                              check_funding_clock, annual_plan)
from .windows import origin_mask


FORMAT = "forecast-input-integration-v1"
HISTORY_MONTHS = tuple(str(m) for m in pd.period_range("2020-12", "2022-12", freq="M"))


def _constant(path, name):
    tree = ast.parse((ROOT/path).read_text())
    return next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == name for t in n.targets))


def contract():
    """Static schema/specification inspection; never import a numerical runner."""
    frozen = _constant("src/spot_perp_basis/training/checkpoints.py", "FROZEN")
    for path, expected in frozen.items():
        if digest(ROOT/path) != expected:
            raise ValueError("Frozen protocol identity changed: "+path)
    cfg = yaml.safe_load((ROOT/"configs/final_evaluation.yaml").read_text())
    if cfg["execution_enabled"] or cfg["holdout_access_authorized"] or cfg["execution"]["enabled"]:
        raise ValueError("Frozen final evaluation flags must remain disabled")
    features = _constant("src/spot_perp_basis/data/features.py", "FEATURES")["minimal"]
    if list(features) != cfg["features"]["primary_order"]:
        raise ValueError("Feature contract differs from frozen specification")
    if (tuple(_constant("src/spot_perp_basis/training/annual.py", "ASSETS")) != ASSETS
            or tuple(_constant("src/spot_perp_basis/training/annual.py", "FINANCING")) != FINANCING):
        raise ValueError("Annual asset/financing registry differs")
    plans = [annual_plan(y) for y in (2023, 2024, 2025)]
    for p, registered in zip(plans, cfg["chronology"]["windows"], strict=True):
        for name in ("fit", "calibration", "evaluation"):
            if [pd.Timestamp(t) for t in p[name]] != [pd.Timestamp(t) for t in registered[name]]:
                raise ValueError("Registered annual calendar mismatch")
    return {"protocol": cfg["protocol_identity"], "feature_order": list(features),
            "frozen_sha256": frozen, "plans": plans,
            "model_registry": list(_constant("src/spot_perp_basis/training/annual.py", "MODELS")),
            "base_input": sorted(_constant("src/spot_perp_basis/training/checkpoints.py", "BASE_INPUT")),
            "optional_input": sorted(_constant("src/spot_perp_basis/training/checkpoints.py", "OPTIONAL_INPUT"))}


def _pinned(root, entry):
    path = safe_path(root, entry["path"])
    if digest(path) != entry["sha256"]:
        raise ValueError("Input provenance/hash mismatch: "+entry["path"])
    return path


def _json(root, entry):
    return json.loads(_pinned(root, entry).read_text())


def _read(root, entry, columns, begin, end):
    path = _pinned(root, entry)
    schema = pq.read_schema(path)
    if not set(columns) <= set(schema.names):
        raise ValueError("Input schema missing required fields: "+entry["path"])
    # Select decision timestamps in the Parquet reader, before materialization.
    return pd.read_parquet(path, columns=columns,
                           filters=[("decision_time_utc", ">=", begin), ("decision_time_utc", "<", end)])


def _digest_set(rows):
    from hashlib import sha256
    return sha256(json.dumps(sorted(rows), separators=(",", ":")).encode()).hexdigest()


def _history_source(root, row):
    spec = Source(row["symbol"], "2022-12", "funding" if row["kind"] == "funding" else row["market"])
    # The fixed official template also applies to permitted historical months.
    expected_url = spec.url.replace("2022-12", row["month"])
    archive, check = safe_path(root, row["path"]), safe_path(root, row["checksum_path"])
    if (row["url"] != expected_url or digest(archive) != row["sha256"]
            or checksum_value(check.read_bytes(), Path(expected_url).name) != row["sha256"]
            or digest(check) != row["checksum_sha256"]):
        raise ValueError("Historical official archive provenance differs")


def _merge_events(old, new, symbol):
    cols = ["symbol", "settlement_time_utc", "available_at_utc", "funding_rate", "funding_interval_hours", "source_sha256"]
    for frame in (old, new):
        if not frame.symbol.eq(symbol).all() or frame.settlement_time_utc.duplicated().any():
            raise ValueError("Mixed asset/duplicate event history")
    overlap = new.loc[new.settlement_time_utc.lt(START), cols].reset_index(drop=True)
    past = old.loc[old.settlement_time_utc.isin(overlap.settlement_time_utc), cols].reset_index(drop=True)
    pd.testing.assert_frame_equal(past, overlap, check_dtype=False)
    events = pd.concat([old.loc[~old.quarantined, cols], new.loc[new.settlement_time_utc.ge(START), cols]], ignore_index=True)
    events = events.sort_values("settlement_time_utc").reset_index(drop=True)
    if events.settlement_time_utc.duplicated().any():
        raise ValueError("Duplicate events at development/final handoff")
    return events, len(overlap)


def _stage(frame, begin, end, *, purge=False):
    begin, end = pd.Timestamp(begin), pd.Timestamp(end)
    expected = pd.date_range(begin, end, freq="h", inclusive="left").as_unit("ns")
    f = frame.loc[(frame.index>=begin)&(frame.index<end)]
    available = f.index.equals(expected.rename(frame.index.name))
    return {"available": available, "expected_clock": len(expected), "present_clock": len(f),
            "absent_clock_rows": len(expected)-len(f),
            "permitted_origin_count_by_calendar_only": int(origin_mask(expected, begin, end, purge=purge).sum()),
            "missing_or_invalid_pairs": int((~(f.spot_valid & f.perp_valid)).sum()),
            "valid_inactive_pairs": int(((f.spot_valid & f.perp_valid)&~f.pair_active).sum()),
            "funding_unknown": int((~f.funding_known).sum())}


def inspect_inputs(data_root, history_root, catalog, public_root=ROOT):
    """Return 24 input cells; no feature, label, fit or prediction entry point."""
    data_root, history_root, public_root = map(Path, (data_root, history_root, public_root))
    with structural_only():
        spec = contract()
        if catalog["schema_version"] != FORMAT:
            raise ValueError("Wrong input catalog version")
        public = _json(public_root, catalog["final_verification"])
        if public["data_schema"] != "final-inputs-v1" or public["status"] != "FINAL_DATA_STRUCTURAL_QA_READY":
            raise ValueError("Require verified final data version")
        inv = _json(data_root, {"path": "inventory.json", "sha256": public["data_receipts_sha256"]["inventory.json"]})
        build = _json(data_root, {"path": "build_receipt.json", "sha256": public["data_receipts_sha256"]["build_receipt.json"]})
        qa = _json(data_root, {"path": "qa_receipt.json", "sha256": public["data_receipts_sha256"]["qa_receipt.json"]})
        if not inv["complete"] or qa["status"] != public["status"] or build["schema_version"] != "final-inputs-v1":
            raise ValueError("Incomplete/wrong final source or build receipt")
        if (build["inventory_sha256"] != public["data_receipts_sha256"]["inventory.json"]
                or qa["build_sha256"] != public["data_receipts_sha256"]["build_receipt.json"]):
            raise ValueError("Final build/QA lineage differs")
        for path, expected in public["implementation_sha256"].items():
            if digest(ROOT/path) != expected:
                raise ValueError("Final input processing identity changed: "+path)
        if _digest_set([[r["key"], r["receipt"]["sha256"]] for r in inv["files"]]+[["NYFED/EFFR/2022-12_2025-12", inv["effr"]["receipt"]["sha256"]]]) != public["canonical_source_set_sha256"]:
            raise ValueError("Final source-set pin differs")
        if _digest_set([[a["path"], a["sha256"]] for a in build["artifacts"]]) != public["canonical_table_set_sha256"]:
            raise ValueError("Final table-set pin differs")
        history = _json(history_root, catalog["history_dataset"])
        sources = _json(history_root, catalog["history_sources"])
        if history["schema_version"] != "five-asset-observations-v1" or sources["schema_version"] != "five-asset-source-v1" or not sources["complete"]:
            raise ValueError("Wrong/incomplete historical source schema")
        if (history["identity"]["acquisition_sha256"] != catalog["history_sources"]["sha256"]
                or history["identity"]["config_sha256"] != sources["identity"]["config_sha256"]):
            raise ValueError("Historical acquisition/config lineage differs")
        effr_raw = _json(history_root, catalog["historical_effr"])
        er = _json(history_root, catalog["historical_effr_receipt"])["sources"]["effr"]
        if (er["sha256"] != catalog["historical_effr"]["sha256"] or er["url"] !=
                "https://markets.newyorkfed.org/api/rates/unsecured/effr/search.json?startDate=2020-01-01&endDate=2022-12-31"):
            raise ValueError("Wrong historical financing source")
        rates = reference_history(effr_raw)
        fa = {a["path"]: a for a in build["artifacts"]}
        final_rates = pd.read_parquet(_pinned(data_root, fa["canonical/effr.parquet"]))
        overlap = final_rates.loc[final_rates.observation_time_utc.lt(START)]
        common = rates.set_index("observation_time_utc").reindex(overlap.observation_time_utc)
        if (not np.array_equal(common.annual_rate, overlap.annual_rate)
                or not pd.DatetimeIndex(common.available_at_utc).equals(pd.DatetimeIndex(overlap.available_at_utc))):
            raise ValueError("Historical/final EFFR overlap conflicts")
        if not final_rates.source.eq("NYFed_EFFR_current_vintage_percent_ACT365_proxy").all():
            raise ValueError("Wrong final financing instrument")
        artifacts = {a["path"]: a for a in history["artifacts"]}
        raw = {(r["symbol"], r["month"], "funding" if r["kind"] == "funding" else r["market"]): r for r in sources["files"]}
        cells, audits, boundaries, verified_history = [], [], [], set()
        dependencies = {catalog[k]["path"]: catalog[k]["sha256"] for k in
                        ("history_dataset", "history_sources", "historical_effr", "historical_effr_receipt")}
        for symbol in ASSETS:
            parts, issues = [], []
            for month in HISTORY_MONTHS:
                name = f"data/processed/five_asset_v1/hourly/{symbol}/{month}.parquet"
                try:
                    for source in ("spot", "usdm_perpetual", "funding"):
                        _history_source(history_root, raw[(symbol, month, source)])
                        verified_history.add((symbol, month, source))
                        row = raw[(symbol, month, source)]
                        dependencies[row["path"]] = row["sha256"]
                        dependencies[row["checksum_path"]] = row["checksum_sha256"]
                    item = artifacts[name]
                    dependencies[item["path"]] = item["sha256"]
                    if item["provenance"]["symbol"] != symbol or item["provenance"]["source_month"] != month:
                        raise ValueError("Wrong historical asset/month identity")
                    for source, prefix in (("spot", "spot"), ("usdm_perpetual", "perp")):
                        if item["provenance"][prefix+"_sha256"] != raw[(symbol, month, source)]["sha256"]:
                            raise ValueError("Hourly/raw source lineage differs")
                    f = _read(history_root, item, RAW+FUNDING+["paired_valid", "active_sample"], HISTORY_START, START)
                    observation_masks(f, symbol)
                    f = f.rename(columns={"paired_valid": "pair_valid", "active_sample": "pair_active"})
                    parts.append(f)
                    audits.append({"asset": symbol, "month": month, "source": "history", "rows": len(f), "sha256": item["sha256"]})
                except (FileNotFoundError, KeyError, ValueError) as e:
                    issues.append({"partition": name, "error": str(e)})
            for year in (2023, 2024, 2025):
                name = f"canonical/hourly/{symbol}/{year}.parquet"
                try:
                    item = fa[name]
                    f = _read(data_root, item, RAW+FUNDING+REFERENCE+["pair_valid", "pair_active"], pd.Timestamp(f"{year}-01-01", tz="UTC"), pd.Timestamp(f"{year+1}-01-01", tz="UTC"))
                    observation_masks(f, symbol)
                    expected = pd.date_range(f"{year}-01-01", f"{year+1}-01-01", tz="UTC", freq="h", inclusive="left").as_unit("ns")
                    if not f.index.equals(expected.rename(f.index.name)):
                        raise ValueError("Annual decision clock missing/duplicated")
                    parts.append(f); audits.append({"asset": symbol, "year": year, "source": "final", "rows": len(f), "sha256": item["sha256"]})
                except (FileNotFoundError, KeyError, ValueError) as e:
                    issues.append({"partition": name, "error": str(e)})
            frame = pd.concat(parts).sort_index() if parts else pd.DataFrame(columns=RAW+FUNDING+["pair_valid", "pair_active"], index=pd.DatetimeIndex([], tz="UTC").as_unit("ns"))
            if len(frame) and frame.index.has_duplicates:
                raise ValueError("Duplicated development/final decision origins")
            # A source-level failure in a monthly catalog is reported, never repaired.
            event_error = None
            try:
                old = pd.read_parquet(_pinned(history_root, artifacts[f"data/processed/five_asset_v1/funding/{symbol}.parquet"]))
                item = artifacts[f"data/processed/five_asset_v1/funding/{symbol}.parquet"]
                dependencies[item["path"]] = item["sha256"]
                new = pd.read_parquet(_pinned(data_root, fa[f"canonical/funding/{symbol}.parquet"]))
                events, overlap_count = _merge_events(old, new, symbol)
                if len(frame): check_funding_clock(frame, events)
                for row in old.loc[old.settlement_time_utc.ge(HISTORY_START-pd.Timedelta(hours=12))].itertuples():
                    if row.source_sha256 != raw[(symbol, row.source_month, "funding")]["sha256"]:
                        raise ValueError("Historical event/raw archive identity differs")
            except (FileNotFoundError, KeyError, ValueError, AssertionError) as e:
                event_error = str(e); overlap_count = None
            # Bind final sources without rebuilding/parsing them a second time.
            for r in inv["files"]:
                if r["receipt"]["symbol"] == symbol:
                    try: check_source(data_root, Source(symbol, r["receipt"]["month"], r["receipt"]["source"]))
                    except (FileNotFoundError, ValueError) as e: issues.append({"partition": r["key"], "error": str(e)})
            masks = observation_masks(frame, symbol) if len(frame) else None
            if len(frame):
                final = frame.loc[frame.index>=START]
                if len(final):
                    av = pd.DatetimeIndex(final_rates.available_at_utc).as_unit("ns")
                    ids = np.searchsorted(av.asi8, final.index.asi8, side="right")-1
                    if (ids<0).any() or not pd.DatetimeIndex(final.effr_available_at_utc).equals(av[ids]):
                        raise ValueError("Final EFFR backward publication clock differs")
            for financing in FINANCING:
                support_error = event_error
                contexts = []
                # Adapters run on homogeneous schema blocks; no final theory is computed.
                for begin, end, kind in ((HISTORY_START, START, "history"), (START, END, "final")):
                    f = frame.loc[(frame.index>=begin)&(frame.index<end)]
                    if not len(f): continue
                    try:
                        if kind == "history": f = f.drop(columns=[c for c in REFERENCE if c in f])
                        context, _ = adapt_hourly(f, symbol, financing, historical_rates=rates)
                        contexts.append(context)
                        allowed = set(spec["base_input"]+spec["optional_input"])
                        if not set(context)<=allowed or not set(spec["base_input"])<=set(context):
                            raise ValueError("Annual input contract mismatch")
                    except ValueError as e: support_error = str(e)
                for plan in spec["plans"]:
                    stages = {n: _stage(frame, *plan[n], purge=n in ("fit", "calibration")) for n in ("warmup", "fit", "calibration", "evaluation")}
                    if contexts:
                        context = pd.concat(contexts).sort_index()
                        for name, stage in stages.items():
                            begin, end = map(pd.Timestamp, plan[name])
                            block = context.loc[(context.index>=begin)&(context.index<end)]
                            stage["financing_unknown"] = int(block.annual_rate.isna().sum())
                    complete = all(s["available"] for s in stages.values()) and not issues and support_error is None
                    cells.append({"asset": symbol, "year": plan["year"], "financing": financing,
                                  "input_status": "available" if complete else "unavailable", "stages": stages,
                                  "diagnostics": issues+([{"error": support_error}] if support_error else []),
                                  "overlap_funding_events_verified": overlap_count,
                                  "statistical_support": "not_evaluated"})
            if len(frame) and not issues:
                for year in (2023, 2024, 2025):
                    t = pd.Timestamp(f"{year}-01-01", tz="UTC")
                    if frame.loc[t, "bar_open_time_utc"] != t-pd.Timedelta(minutes=1):
                        raise ValueError("Previous-month midnight handoff differs")
                    boundaries.append({"asset": symbol, "origin": t.isoformat(), "bar_open": (t-pd.Timedelta(minutes=1)).isoformat()})
                march = _read(data_root, fa[f"canonical/minute/{symbol}/2023-03.parquet"], RAW, pd.Timestamp("2023-03-24 12:40", tz="UTC"), pd.Timestamp("2023-03-24 14:01", tz="UTC"))
                m = observation_masks(march, symbol)
                if (len(m)!=81 or m.spot_state.eq("missing").sum()!=80 or m.spot_state.eq("invalid").sum()!=1
                        or not m.perp_state.eq("active").all()):
                    raise ValueError("Registered March interruption was altered/dropped")
        available = sum(c["input_status"] == "available" for c in cells)
        status = "FINAL_FORECASTING_INPUT_INTEGRATION_READY" if available==24 else "FINAL_FORECASTING_INPUT_INTEGRATION_PARTIAL" if available else "FINAL_FORECASTING_INPUT_INTEGRATION_BLOCKED"
        return {"schema_version": FORMAT, "status": status, "protocol": spec["protocol"],
                "feature_order": spec["feature_order"], "annual_plans": spec["plans"], "cells": cells,
                "model_registry": spec["model_registry"], "input_catalog": catalog,
                "historical_config_sha256": history["identity"]["config_sha256"],
                "required_history_inputs": [{"path": p, "sha256": h} for p,h in sorted(dependencies.items())],
                "available_cells": available, "requested_cells": 24, "partition_audits": audits,
                "historical_raw_archives_verified": len(verified_history), "effr_overlap_records_verified": len(overlap),
                "boundaries": boundaries, "historical_warmup_start": HISTORY_START.isoformat(),
                "final_source_set_sha256": public["canonical_source_set_sha256"],
                "timestamp_conventions": public["timestamp_conventions"],
                "no_final_features_labels_fits_forecasts_scores_or_economics": True,
                "statistical_support": "not_evaluated"}
