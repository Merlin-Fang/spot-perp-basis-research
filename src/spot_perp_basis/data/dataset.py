"""Frozen development dataset IO, provenance and immutable artifact writes.

All access bounds are checked before opening market partitions."""

import json
from pathlib import Path

import pandas as pd

from ..artifacts import ROOT, file_sha256, load_config, write_json
from .funding import funding_history
from .market import MINUTE, hourly_panel, market_quality, minute_panel
from .windows import START, END, bounded


# Frozen dataset provenance and immutable outputs

CONFIG = "configs/five_asset_dataset_v1.yaml"


BASE = ROOT / "experiments/five_asset_dataset"


def study():
    cfg = load_config(CONFIG)
    if not START <= pd.Timestamp(cfg["start"]) < pd.Timestamp(cfg["end"]) <= END:
        raise ValueError("Only 2020–2022 study windows are permitted")
    if cfg["interval"] != "1m" or not cfg["output_namespace"].startswith("five_asset_"):
        raise ValueError("Invalid isolated data namespace")
    return cfg


def allowed_month(symbol, month, cfg):
    if symbol not in cfg["assets"] or not "2020-01" <= month <= "2022-12":
        raise ValueError("Forbidden asset/month before opening source")
    if month < cfg["assets"][symbol]["first_month"]:
        raise ValueError("Prelisting archive month")


def months_for(symbol, cfg):
    return pd.period_range(cfg["assets"][symbol]["first_month"],
        (pd.Timestamp(cfg["end"])-MINUTE).strftime("%Y-%m"),freq="M").astype(str).tolist()


def preserve():
    snapshot = json.loads((BASE / "preservation_receipt.json").read_text())
    for name, digest in snapshot["prior_file_sha256"].items():
        if file_sha256(ROOT / name) != digest:
            raise ValueError(f"Prior research changed: {name}")
    for name, stat in snapshot["prior_data_result_file_stats"].items():
        p = ROOT / name
        if not p.is_file() or p.stat().st_size != stat["bytes"] or p.stat().st_mtime_ns != stat["mtime_ns"]:
            raise ValueError(f"Prior dataset/result changed: {name}")
    return {"content_hashes_unchanged":len(snapshot["prior_file_sha256"]),
            "data_result_file_states_unchanged":len(snapshot["prior_data_result_file_stats"])}


def frozen_json(path, value):
    path = Path(path)
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError(f"Refusing to overwrite different receipt: {path}")
    else:
        write_json(path, value)


def save_frame(frame, path, provenance):
    """Resume verified outputs only; publish no partial Parquet file."""
    path = Path(path)
    receipt = path.with_suffix(path.suffix + ".receipt.json")
    if receipt.exists():
        old = json.loads(receipt.read_text())
        if old["provenance"] != provenance or not path.exists() or file_sha256(path) != old["sha256"]:
            raise ValueError(f"Derived artifact provenance/content changed: {path}")
        return old
    if path.exists():
        raise ValueError(f"Unreceipted derived artifact requires review: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".partial")
    frame.to_parquet(temporary, compression="zstd", index=True)
    temporary.replace(path)
    info = {"path":str(path.relative_to(ROOT)),"sha256":file_sha256(path),
        "bytes":path.stat().st_size,"rows":len(frame),"provenance":provenance}
    frozen_json(receipt,info)
    return info


# Development-only dataset access

def _window(symbol,start,end):
    cfg=study()
    if symbol not in cfg['assets']: raise ValueError('Unknown dataset symbol')
    start,end=pd.Timestamp(start),pd.Timestamp(end)
    if start.tz is None or end.tz is None or not START<=start<end<=END:
        raise ValueError('Require a timezone-aware 2020–2022 observation window')
    manifest=json.loads((ROOT/'manifests/five_asset_v1_dataset.json').read_text())
    if manifest['identity']['config_sha256']!=file_sha256(ROOT/CONFIG): raise ValueError('Configuration changed')
    return cfg,manifest,start,end


def _verified(info):
    path=ROOT/info['path']
    if file_sha256(path)!=info['sha256']: raise ValueError('Dataset artifact changed')
    return path


def minute_partitions(symbol,start=START,end=END):
    """Yield explicit grids, including missing/inactive/prelisting rows; no fill."""
    cfg,manifest,start,end=_window(symbol,start,end)
    for info in manifest['partitions']:
        if info['symbol']!=symbol: continue
        if pd.Timestamp(info['first_decision_utc'])>=end or pd.Timestamp(info['last_decision_utc'])<start: continue
        allowed_month(symbol,info['month'],cfg)
        frame=pd.read_parquet(_verified(info),filters=[('decision_time_utc','>=',start),('decision_time_utc','<',end)])
        bounded(frame)
        if not frame.symbol.eq(symbol).all(): raise ValueError('Cross-symbol contamination')
        yield frame


def hourly_observations(symbol,start=START,end=END):
    """Hourly origins with completed minute/hour inputs; no future targets."""
    _,manifest,start,end=_window(symbol,start,end)
    frame=pd.read_parquet(_verified(manifest['hourly_panel']),filters=[('symbol','==',symbol),
        ('decision_time_utc','>=',start),('decision_time_utc','<',end)])
    bounded(frame)
    if not frame.symbol.eq(symbol).all(): raise ValueError('Cross-symbol contamination')
    return frame


def funding_events(symbol,start=START,end=END,*,purpose='causal'):
    """Quarantine excluded by default; purpose='all' is explicit archival access.

    A causal caller must additionally as-of join on available_at_utc. An
    accounting caller must own the position at (entry, exit] settlements.
    """
    cfg,manifest,start,end=_window(symbol,start,end)
    column={'causal':'eligible_for_causal_use','accounting':'eligible_for_accounting','all':None}.get(purpose)
    if purpose not in ('causal','accounting','all'): raise ValueError('Unknown funding purpose')
    suffix=f'/funding/{symbol}.parquet'
    info=next(row for row in manifest['artifacts'] if row['path'].endswith(suffix))
    filters=[('settlement_time_utc','>=',start),('settlement_time_utc','<',end)]
    if column: filters.append((column,'==',True))
    frame=pd.read_parquet(_verified(info),filters=filters)
    if not frame.symbol.eq(symbol).all(): raise ValueError('Cross-symbol contamination')
    return frame
