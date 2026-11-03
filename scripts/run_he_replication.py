"""Run the isolated He v7 mechanics and alternative-rate sensitivities offline."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from spot_perp_basis.artifacts import ROOT, file_sha256, utc_now, write_json, load_config
from spot_perp_basis.data.windows import (
    training_partitions,
    require_training_period,
    START,
    END,
    require_training_times,
)
from spot_perp_basis.data.market import Market
from spot_perp_basis.backtest.accounting import account_positions
from spot_perp_basis.backtest.execution import convergence_positions
from spot_perp_basis.data.features import hourly_theory_observations as hourly_observations
from spot_perp_basis.data.funding import prepare_effr
from spot_perp_basis.evaluation.metrics import summarize

BASE = ROOT / "experiments/experiment1_he2026"


def load_inputs(symbol, dataset, start, end, receipts):
    chosen = [p for p in training_partitions(dataset["partitions"], START, END) if p["symbol"] == symbol]
    if len(chosen) != 36 or any(not "2020-01" <= p["month"] <= "2022-12" for p in chosen):
        raise ValueError("Expected exactly 36 permitted monthly price partitions")
    columns = ["bar_open_time_utc"]+[f"{v}_{c}" for v in ("spot", "perp")
                                    for c in ("open", "close", "volume", "trades")]
    pieces = []
    for part in chosen:
        path = ROOT/part["path"]
        if file_sha256(path) != part["sha256"]:
            raise ValueError("Price input checksum changed")
        # Predicate excludes the last 2022 bar completing on the 2023 boundary
        # before materialization. No held-out partition is opened, even for schema.
        pieces.append(pd.read_parquet(path, columns=columns,
            filters=[("decision_time_utc", ">=", START), ("decision_time_utc", "<", END)]))
        receipts.append({"path": part["path"], "sha256": part["sha256"],
                         "role": "training_prices", "predicate": "2020 <= decision_time < 2023"})
    frame = pd.concat(pieces).sort_index()
    require_training_times(frame.index)
    frame = frame.loc[(frame.index >= start) & (frame.index < end)]
    funding_path = f"results/phase2_5/{symbol}_funding_notional_proxies.parquet"
    old = json.loads((ROOT/"manifests/phase2_7_history.json").read_text())
    expected = old["prior_results"][funding_path]
    if file_sha256(ROOT/funding_path) != expected:
        raise ValueError("Audited training-only funding proxy changed")
    events = pd.read_parquet(ROOT/funding_path)
    require_training_times(events.settlement_time_utc)
    receipts.append({"path": funding_path, "sha256": expected,
                     "role": "actual_training_settlements_with_past_trade_notional_proxy"})
    return Market(frame), events


def published_reference():
    rows = []
    data = {
        "BTCUSDT": [(7.18,49.62,6.91,-3.82,8.35,29.99,19.63,8616),
                    (12.36,64.81,5.24,-2.,9.17,34.76,30.05,8760),
                    (37.85,60.02,1.59,-.36,2.97,55.93,4.09,8760)],
        "ETHUSDT": [(10.43,67.53,6.48,-2.62,11.19,41.98,25.55,8616),
                    (12.04,75.56,6.27,-2.28,8.8,40.66,34.91,8760),
                    (25.92,70.73,2.73,-.67,3.02,65.52,5.21,8760)]}
    for symbol, years in data.items():
        for year, values in zip((2020,2021,2022), years):
            rows.append(dict(zip(["published_sharpe", "published_annual_return_pct", "published_volatility_pct",
                "published_max_drawdown_pct", "published_average_holding_hours", "published_price_return_pct",
                "published_funding_return_pct", "published_hour_observations"], values),
                symbol=symbol, year=year, cost_scenario="zero", source="He_v7_Tables_5_and_7",
                comparison_status="not_exactly_comparable_rates_prices_execution_hedge_denominator",
                original_aave_reproduced_annual_return_pct=np.nan))
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=BASE)
    args = parser.parse_args()
    out = args.output.resolve()
    if (out/"replication_summary.csv").exists() or (out/"run_receipt.json").exists():
        raise FileExistsError("Choose a new output directory; previous results are immutable")
    cfg = json.loads((BASE/"config.json").read_text())
    protocol = json.loads((BASE/"protocol_receipt.json").read_text())
    for name, key in (("config.json", "config_sha256"), ("implementation_spec.md", "spec_sha256")):
        if file_sha256(BASE/name) != protocol[key]:
            raise ValueError("Pre-implementation specification changed")
    for name, digest in protocol["preserved_files"].items():
        if file_sha256(ROOT/name) != digest:
            raise ValueError(f"Prior research changed: {name}")
    start, end = require_training_period(cfg["start"], cfg["end"], load_config()["splits"])
    if not (START <= start < end <= END):
        raise ValueError("Out-of-scope dates")
    source_receipt = json.loads((BASE/"rate_source_receipt.json").read_text())
    effr_source = source_receipt["sources"]["effr"]
    if file_sha256(BASE/effr_source["path"]) != effr_source["sha256"]:
        raise ValueError("Rate input changed")
    effr = prepare_effr(json.loads((BASE/effr_source["path"]).read_text()), cfg["effr_availability_lag_days"])
    zero = pd.DataFrame({"observation_time_utc": [START], "available_at_utc": [START],
                         "annual_rate": [0.], "source": ["assumed_constant_zero"]})
    rate_cases = {"zero_rate": zero, "nyfed_effr_lag7d": effr}
    dataset = json.loads((ROOT/"manifests/phase2_dataset.json").read_text())
    out.mkdir(parents=True, exist_ok=True)
    effr.to_csv(out/"alternative_rate_observations.csv", index=False)
    trades, opened, summaries, daily_rows, hourly_rows, funding_rows, audits = [], [], [], [], [], [], []
    inputs = [{"path": str((BASE/effr_source["path"]).relative_to(ROOT)), "sha256": effr_source["sha256"],
               "role": "alternative_interest_rate_only_2020_2022"}]
    for symbol in cfg["symbols"]:
        market, events = load_inputs(symbol, dataset, start, end, inputs)
        print(symbol, len(market.frame), "training minute observations", flush=True)
        for rate_case in cfg["rate_sensitivities"]:
            rates = rate_cases[rate_case]
            obs = hourly_observations(market, rates, start, end)
            saved = obs.copy()
            saved["symbol"], saved["rate_scenario"] = symbol, rate_case
            saved.to_parquet(out/f"{symbol}_{rate_case}_hourly_inputs.parquet")
            for year in (2020,2021,2022):
                part = obs.loc[obs.index.year == year]
                audits.append({"symbol": symbol, "year": year, "rate_scenario": rate_case,
                    "expected_hourly_samples": len(part), "observed_hourly_prices": int(part.spot_close.notna().sum()),
                    "inactive_or_missing_sample": int((~part.active_sample.eq(True)).sum()),
                    "missing_rate_samples": int(part.annual_rate.isna().sum()),
                    "eligible_signal_samples": int(part.signal_available.sum()),
                    "observed_funding_events": int(events.settlement_time_utc.dt.year.eq(year).sum()),
                    "funding_subsecond_events": int((events.loc[events.settlement_time_utc.dt.year.eq(year),
                        "settlement_time_utc"].dt.microsecond > 0).sum())})
            for cost_name, fee in cfg["cost_scenarios"].items():
                pos = convergence_positions(market, obs, **fee, delay_minutes=cfg["execution_delay_minutes"])
                rec, path, ledger = account_positions(market, pos, events, rates,
                    notional=cfg["perpetual_entry_notional_usdt"], capital=cfg["reporting_capital_usdt"], **fee)
                summary, daily, hourly = summarize(rec, path, market, start=start, end=end,
                    capital=cfg["reporting_capital_usdt"], notional=cfg["perpetual_entry_notional_usdt"])
                labels = {"symbol": symbol, "rate_scenario": rate_case, "cost_scenario": cost_name,
                    "replication_status": "exploratory_alternative_rate_sensitivity",
                    "funding_notional": "past_perpetual_trade_close_proxy",
                    "effective_spread_status": "excluded_unavailable", "return_denominator": "our_fixed_capital_3N"}
                for df in (rec, summary, daily, hourly, ledger):
                    for key, value in labels.items():
                        df[key] = value
                trades.append(rec.loc[rec.status == "closed"])
                opened.append(rec.loc[rec.status != "closed"])
                summaries.append(summary)
                daily_rows.append(daily.reset_index())
                hourly_rows.append(hourly.reset_index())
                funding_rows.append(ledger)
                # Accounting conservation includes unresolved positions at their last mark.
                for key in ("spot_pnl", "perp_pnl", "funding_pnl", "financing_cost", "transaction_costs", "net_pnl"):
                    if not np.isclose(rec[key].sum(), daily[key].sum(), atol=1e-6):
                        raise ValueError(f"Trade/calendar mismatch: {key}")
                print(symbol, rate_case, cost_name, len(rec), "positions; net", round(rec.net_pnl.sum(),2), flush=True)
    summary = pd.concat(summaries, ignore_index=True)
    blocked = [{"symbol": s, "year": y, "cost_scenario": c, "rate_scenario": "original_aave",
        "replication_status": "blocked_missing_aave", "effective_spread_status": "excluded_unavailable",
        "return_denominator": "unresolved_paper_empirical_denominator"}
        for s in cfg["symbols"] for y in (2020,2021,2022) for c in cfg["cost_scenarios"]]
    summary = pd.concat([summary, pd.DataFrame(blocked)], ignore_index=True)
    pd.concat(trades, ignore_index=True).to_csv(out/"replication_trades.csv", index=False)
    pd.concat(opened, ignore_index=True).to_csv(out/"open_positions.csv", index=False)
    summary.to_csv(out/"replication_summary.csv", index=False)
    pd.DataFrame(audits).to_csv(out/"data_availability.csv", index=False)
    pd.concat(daily_rows, ignore_index=True).to_csv(out/"daily_equity.csv", index=False)
    pd.concat(hourly_rows, ignore_index=True).to_parquet(out/"hourly_pnl.parquet", index=False)
    pd.concat(funding_rows, ignore_index=True).to_csv(out/"funding_ledger.csv", index=False)
    published_reference().to_csv(out/"published_reference_2020_2022.csv", index=False)
    receipt = {"completed_at_utc": utc_now(), "status": "partial_replication_alternative_rates_only",
        "input_receipts": inputs, "price_partitions_read": sum(x["role"] == "training_prices" for x in inputs),
        "post2022_market_value_files_opened": 0, "post2022_rate_observations_acquired": 0,
        "paper_primary_executed": False, "aave_rates_acquired": False,
        "protocol_sha256": file_sha256(BASE/"protocol_receipt.json"),
        "sources": {str(p.relative_to(ROOT)): file_sha256(p) for p in [
            ROOT/"src/spot_perp_basis/he_replication.py", ROOT/"scripts/run_he_replication.py",
            ROOT/"scripts/fetch_he_replication_inputs.py"]},
        "outputs": {str(p.relative_to(out)): file_sha256(p) for p in out.iterdir()
                    if p.is_file() and p.name not in {"run_receipt.json", "replication_audit.md", "verification.json", "unit_tests.txt"}}}
    write_json(out/"run_receipt.json", receipt)


if __name__ == "__main__":
    main()
