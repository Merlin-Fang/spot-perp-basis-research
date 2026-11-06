"""Reproduce Phase 2.7 labels, input availability, opportunity and mark audits."""

import gc
import json
import time

import numpy as np
import pandas as pd

from spot_perp_basis.artifacts import ROOT, file_sha256, load_config, utc_now, write_json
from spot_perp_basis.data.ingest import verify_entry
from spot_perp_basis.data.windows import training_partitions, MINUTE_NS, ns_index
from spot_perp_basis.data.market import Market
from spot_perp_basis.backtest.accounting import fill_cost
from spot_perp_basis.data.funding import reference_asof
from spot_perp_basis.data.labels import LabelEngine
from spot_perp_basis.data.features import historical_features, matured_summaries

OUT = ROOT / "results/phase2_7"


def verify_history():
    frozen = json.loads((ROOT / "manifests/phase2_7_protocol.json").read_text())
    for path, key in (("configs/target_audit.yaml", "config_sha256"),
                      ("docs/phase2_7_protocol.md", "protocol_sha256")):
        # Protocol receipt names are checked explicitly rather than inferred.
        expected = frozen.get(key, frozen.get("protocol_note_sha256") if key == "protocol_sha256" else None)
        if expected is None:
            raise ValueError(f"Missing protocol receipt key: {key}")
        assert file_sha256(ROOT / path) == expected, path
    history = json.loads((ROOT / "manifests/phase2_7_history.json").read_text())
    for group in ("preserved_files", "prior_results"):
        for path, digest in history[group].items():
            assert file_sha256(ROOT / path) == digest, path


def load_training(symbol, cfg, catalog):
    chosen = [p for p in training_partitions(catalog["partitions"], cfg["start"], cfg["end"]) if p["symbol"] == symbol]
    for part in chosen:
        assert part["month"] < "2023-01"
        assert file_sha256(ROOT / part["path"]) == part["sha256"]
    frame = pd.concat([pd.read_parquet(ROOT / p["path"]) for p in chosen]).sort_index()
    frame = frame.loc[(frame.index >= pd.Timestamp(cfg["start"])) & (frame.index < pd.Timestamp(cfg["end"]))]
    fund_path = f"results/phase2_5/{symbol}_funding_notional_proxies.parquet"
    old = json.loads((ROOT / "manifests/phase2_7_history.json").read_text())
    assert file_sha256(ROOT / fund_path) == old["prior_results"][fund_path]
    events = pd.read_parquet(ROOT / fund_path)
    return Market(frame), events, chosen, fund_path


def signal_inventory(market, labels, symbol, cfg):
    times = market.times[market.eligible & (market.basis("close") >= cfg["signals"]["entry_arithmetic_basis"]-1e-12)]
    rows = []
    old = pd.read_parquet(ROOT / "results/phase2_5/trades.parquet")
    trades = old.loc[old.run_id == f"{symbol}_basis_close_1m_d0_zero"].copy()
    trades["entry_signal_utc"] = pd.to_datetime(trades.entry_signal_utc, utc=True)
    for year in (2020, 2021, 2022):
        tt = times[pd.to_datetime(times, utc=True).year == year]
        group = labels.loc[labels.year == year]
        tr = trades.loc[trades.entry_signal_utc.dt.year == year]
        rows.append({"symbol": symbol, "year": year, "signal_timestamps": len(tt),
            "contiguous_runs": int(1+(np.diff(tt) > MINUTE_NS).sum()) if len(tt) else 0,
            "clusters_gap_gt72h": int(1+(np.diff(tt) > cfg["signals"]["cluster_gap_hours"]*60*MINUTE_NS).sum()) if len(tt) else 0,
            "original_completed_trades": int(tr.exit_utc.notna().sum()), "original_open_trades": int(tr.exit_utc.isna().sum()),
            "holding_median_hours": tr.duration_hours.median(), "holding_max_hours": tr.duration_hours.max(),
            "holding_gt72h": int((tr.duration_hours > 72).sum()),
            "fixed72_price_labels": int((group.label_status == "observed").sum()),
            "fixed72_complete_risk_labels": int(group.risk_valid.sum())})
    trades["symbol"] = symbol
    return pd.DataFrame(rows), trades


def mark_sample(market, events, labels, features, symbol, cfg):
    """Strict June-only mark comparison; missing marks never use a trade fallback."""
    sample = cfg["supplementary"]["sample_month"]
    reference = {kind: pd.read_parquet(ROOT / f"data/processed/phase2_7/reference/{kind}/{symbol}/{sample}.parquet")
                 for kind in cfg["supplementary"]["kinds"]}
    mark = reference["markPriceKlines"]
    lo = mark.bar_open_time_utc.min()
    hi = mark.observation_time_utc.max()
    feature_rows = features.loc[(features.decision_time_utc >= lo) & (features.decision_time_utc < hi)].copy()
    extra = feature_rows[["symbol", "decision_time_utc", "basis_bps", "funding_last_bps", "spot_close", "perp_close"]].copy()
    for kind, frame in reference.items():
        values, ages = reference_asof(frame, extra.decision_time_utc, feature=True,
                                      max_age_minutes=cfg["supplementary"]["maximum_asof_age_minutes"])
        extra[kind] = values
        extra[kind+"_age_minutes"] = ages
        # A spread must compare the same observation end, despite the extra
        # feature publication delay. Comparing a lagged mark with today's trade
        # close would also measure the intervening market return.
        obs = ns_index(extra.decision_time_utc)-np.nan_to_num(ages, nan=0).astype('int64')*MINUTE_NS-MINUTE_NS
        pos = np.searchsorted(market.times, obs, side="right")-1
        bounded = np.maximum(0, pos)
        exact = (pos >= 0) & (market.times[bounded] == obs) & np.isfinite(values)
        venue = "perp" if kind == "markPriceKlines" else "spot"
        extra[kind+"_matched_trade_close"] = np.where(exact, market.frame[venue+"_close"].to_numpy()[bounded], np.nan)
        extra[kind+"_observation_time_utc"] = pd.to_datetime(np.where(np.isfinite(values), obs, np.iinfo('int64').min), utc=True)
    selected = labels.loc[(labels.label_status == "observed") & (labels.entry_utc >= lo+pd.Timedelta(minutes=1))
                          & (labels.exit_utc < hi) & (labels.delay_minutes == cfg["entry_delay_minutes"])].copy()
    mt = market.times
    et = ns_index(events.settlement_time_utc)
    marked_funding, _ = reference_asof(mark, events.settlement_time_utc, feature=False)
    rows = []
    for r in selected.itertuples():
        en, ex = r.entry_utc.value, r.exit_utc.value
        left, right = np.searchsorted(mt, [en, ex])
        left = np.searchsorted(mt, en, side="right")
        times = np.r_[en, mt[left:right], ex]
        marks, age = reference_asof(mark, pd.to_datetime(times, utc=True), feature=False)
        a, b = np.searchsorted(et, [en, ex], side="right")
        cash = r.quantity*marked_funding[a:b]*events.funding_rate.to_numpy()[a:b]
        complete = bool(r.risk_valid and np.isfinite(marks).all() and np.isfinite(cash).all())
        result = {"symbol": symbol, "decision_time_utc": r.decision_time_utc, "mark_complete": complete,
                  "trade_collateral_ratio": r.collateral_ratio, "trade_funding_cash_bps": r.funding_cash_bps}
        if complete:
            funding = np.r_[0., np.cumsum(cash)][np.searchsorted(et[a:b], times, side="right")]
            costs = np.full(len(times), sum(fill_cost(r.quantity, r.spot_entry_price, r.perp_entry_price, cfg["costs"])))
            costs[-1] += sum(fill_cost(r.quantity, r.spot_exit_price, r.perp_exit_price, cfg["costs"]))
            reserve = np.max(cfg["primary_maintenance_fraction"]*r.quantity*marks+
                             r.quantity*(marks-r.perp_entry_price)-funding+costs)
            result.update(mark_collateral_ratio=max(0., reserve)/cfg["spot_notional"],
                          mark_funding_cash_bps=float(cash.sum()/cfg["spot_notional"]*1e4),
                          max_mark_age_minutes=float(age.max()))
        rows.append(result)
    return pd.DataFrame(rows), extra


def main():
    start = time.monotonic()
    verify_history()
    cfg = load_config(ROOT / "configs/target_audit.yaml")
    reference_manifest_path = ROOT / "manifests/phase2_7_supplementary.json"
    references = json.loads(reference_manifest_path.read_text())
    assert len(references["archives"]) == cfg["supplementary"]["maximum_download_archives"]
    for item in references["archives"]:
        assert item["month"] == cfg["supplementary"]["sample_month"] < "2023-01"
        verify_entry(item)
        assert file_sha256(ROOT / item["processed_path"]) == item["processed_sha256"]
    catalog = json.loads((ROOT / "manifests/phase2_dataset.json").read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    origins = pd.date_range(cfg["origin_start"], cfg["end"], freq="h", inclusive="left")
    artifacts, sources, timing = {}, {}, []
    sources.update({item["processed_path"]: item["processed_sha256"] for item in references["archives"]})
    all_labels, all_features, all_signals, opportunities, old_trades, mark_rows, ref_features = [], [], [], [], [], [], []
    for symbol in cfg["symbols"]:
        tick = time.monotonic()
        market, events, chosen, fund_path = load_training(symbol, cfg, catalog)
        sources.update({p["path"]: p["sha256"] for p in chosen})
        sources[fund_path] = file_sha256(ROOT / fund_path)
        engine = LabelEngine(market, events, cfg)
        labels = pd.DataFrame([engine.outcome(t, delay) for delay in (cfg["entry_delay_minutes"], cfg["delay_sensitivity_minutes"])
                               for t in origins])
        labels["symbol"] = symbol
        labels["primary_sparse"] = np.tile(np.arange(len(origins)) % cfg["independent_origin_spacing_hours"] == 0, 2)
        features = historical_features(market, events, origins, cfg)
        features["symbol"] = symbol
        history = matured_summaries(labels.loc[labels.delay_minutes == cfg["entry_delay_minutes"]], origins, cfg)
        features = features.merge(history, on="decision_time_utc", validate="one_to_one")
        signal_times = pd.to_datetime(market.times[market.eligible & (market.basis("close") >= cfg["signals"]["entry_arithmetic_basis"]-1e-12)], utc=True)
        signals = pd.DataFrame([engine.outcome(t) for t in signal_times])
        signals["symbol"] = symbol
        sf = historical_features(market, events, signal_times, cfg)
        sh = matured_summaries(labels.loc[labels.delay_minutes == cfg["entry_delay_minutes"]], signal_times, cfg)
        sf = sf.merge(sh, on="decision_time_utc", validate="one_to_one")
        signals = signals.merge(sf.drop(columns=["year"]), on="decision_time_utc", validate="one_to_one")
        counts, trades = signal_inventory(market, signals, symbol, cfg)
        signals["actual_original_entry_signal"] = signals.decision_time_utc.isin(trades.entry_signal_utc)
        marks, extras = mark_sample(market, events, labels, features, symbol, cfg)
        all_labels.append(labels); all_features.append(features); all_signals.append(signals)
        opportunities.append(counts); old_trades.append(trades); mark_rows.append(marks); ref_features.append(extras)
        timing.append({"symbol": symbol, "training_paired_rows": len(market.times), "hourly_origins": len(origins),
                       "two_delay_label_rows": len(labels), "construction_seconds": time.monotonic()-tick})
        print(timing[-1], flush=True)
        del engine, market, events
        gc.collect()
    for name, pieces in (("labels", all_labels), ("features", all_features), ("signal_labels", all_signals),
                         ("original_trade_opportunities", old_trades), ("mark_sensitivity", mark_rows),
                         ("reference_feature_sample", ref_features)):
        path = OUT / f"{name}.parquet"
        pd.concat(pieces, ignore_index=True).to_parquet(path, index=False)
        artifacts[str(path.relative_to(ROOT))] = file_sha256(path)
    pd.concat(opportunities, ignore_index=True).to_csv(OUT / "opportunities.csv", index=False)
    pd.DataFrame(timing).to_csv(OUT / "construction_timing.csv", index=False)
    verify_history()
    write_json(ROOT / "manifests/phase2_7_labels.json", {"created_at_utc": utc_now(), "config_sha256": file_sha256(ROOT / "configs/target_audit.yaml"),
        "protocol_sha256": file_sha256(ROOT / "docs/phase2_7_protocol.md"), "source_sha256": sources,
        "supplementary_manifest_sha256": file_sha256(reference_manifest_path),
        "source_months": sorted(set(p["month"] for p in training_partitions(catalog["partitions"], cfg["start"], cfg["end"]))),
        "outputs": artifacts, "seconds": time.monotonic()-start, "evaluation_outcomes_read": False})


if __name__ == "__main__":
    main()
