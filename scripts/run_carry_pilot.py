"""Build the frozen, training-only Phase 2.6 carry outcomes and simple forecasts."""

import json
import platform

import numpy as np
import pandas as pd

from spot_perp_basis.artifacts import ROOT, file_sha256, load_config, utc_now, write_json
from spot_perp_basis.data.windows import require_training_period, training_partitions, purged_fold
from spot_perp_basis.data.market import Market
from spot_perp_basis.backtest.carry import fixed_outcome
from spot_perp_basis.models.baselines import funding_forecasts
from spot_perp_basis.data.features import input_features, origin_grid


def check_frozen(cfg, frozen):
    assert file_sha256(ROOT / "configs/carry_pilot.yaml") == frozen["config_sha256"]
    assert file_sha256(ROOT / "docs/phase2_6_protocol.md") == frozen["protocol_note_sha256"]
    assert file_sha256(ROOT / "docs/research_plan.md") == frozen["plan_v2_sha256"]
    old = json.loads((ROOT / "manifests/research_plan_v2_amendment.json").read_text())
    for path, expected in old["preserved"].items():
        assert file_sha256(ROOT / path) == expected, path
    for item in old["relocated_historical_bytes"].values():
        assert file_sha256(ROOT / item["archive"]) == item["sha256"]


def policy_references(base, features, outcomes, cfg):
    rows, decisions = [], []
    for symbol in cfg["symbols"]:
        for horizon in cfg["horizons_hours"]:
            for fold in cfg["folds"]:
                origin = pd.to_datetime(base.decision_time_utc, utc=True)
                frame = base.loc[(base.symbol == symbol) & (base.horizon_hours == horizon)
                                 & (base.delay_minutes == cfg["primary_delay_minutes"])
                                 & base.primary_sparse & (origin >= pd.Timestamp(fold["eval_start"]))
                                 & (origin+pd.Timedelta(hours=horizon, minutes=2) < pd.Timestamp(fold["eval_end"]))].copy()
                frame = frame.merge(features, on=["symbol", "decision_time_utc"], suffixes=("", "_input"), validate="many_to_one")
                financial = outcomes.loc[(outcomes.scenario == cfg["primary_scenario"])
                                         & (outcomes.symbol == symbol) & (outcomes.horizon_hours == horizon)
                                         & (outcomes.delay_minutes == cfg["primary_delay_minutes"])].set_index("case_id")
                cost = cfg["cost_scenarios"][cfg["primary_scenario"]]
                for policy in cfg["policies"]:
                    total, completed, failed, attempts, blocked = 0., 0, 0, 0, False
                    counterfactual, last_exit = 0., None
                    for r in frame.sort_values("decision_time_utc").to_dict("records"):
                        ratio = r.get("perp_close", np.nan)/r.get("spot_close", np.nan)
                        predicted = r.get("funding_last_bps", np.nan)*horizon/8/1e4*ratio*cfg["spot_notional"]
                        expected_friction = cfg["spot_notional"]*(2*(cost["spot_fee_bps"]+ratio*cost["perp_fee_bps"])
                                            +2*(1+ratio)*cost["slippage_bps"])/1e4
                        expected_finance = cfg["spot_notional"]*cost["financing_apr"]*horizon/(365*24)
                        expected_net = predicted-expected_friction-expected_finance
                        trade = policy == "always_carry" or (policy == "persistence_break_even"
                                and r["funding_features_ready"] and np.isfinite(expected_net) and expected_net > 0)
                        if policy == "no_trade":
                            trade = False
                        log = {"symbol": symbol, "horizon_hours": horizon, "fold": fold["name"], "policy": policy,
                               "decision_time_utc": r["decision_time_utc"], "case_id": r["case_id"],
                               "attempt": trade, "persistence_implied_net_pnl": expected_net,
                               "status": "no_trade", "credited_net_pnl": 0.}
                        if trade:
                            attempts += 1
                            result = financial.loc[r["case_id"]] if r["case_id"] in financial.index else None
                            good = result is not None and bool(result.feasible_complete)
                            if result is not None:
                                counterfactual += result.net_pnl
                            if good and not blocked:
                                # Realized prior PnL remains cash; no external funding injections.
                                good = cfg["initial_capital"]+total >= result.required_initial_cash
                            if last_exit is not None and pd.Timestamp(r["decision_time_utc"]) <= last_exit:
                                raise ValueError("Prespecified policy grid overlaps a previous position")
                            if result is not None:
                                last_exit = pd.Timestamp(result.exit_utc)
                            if good and not blocked:
                                total += result.net_pnl
                                completed += 1
                                log.update(status="funded_complete", credited_net_pnl=float(result.net_pnl))
                            else:
                                failed += 1
                                blocked = True
                                log.update(status="infeasible_unknown_or_prior_failure", credited_net_pnl=None)
                        decisions.append(log)
                    days = (pd.Timestamp(fold["eval_end"])-pd.Timestamp(fold["eval_start"])).days
                    rows.append({"symbol": symbol, "horizon_hours": horizon, "fold": fold["name"], "policy": policy,
                                 "calendar_days": days, "opportunities": len(frame), "attempts": attempts,
                                 "funded_completed_before_any_failure": completed, "failed_or_unresolved_attempts": failed,
                                 "funded_net_pnl": None if blocked else total,
                                 "simple_annual_return": None if blocked else total/cfg["initial_capital"]*365/days,
                                 "counterfactual_observed_pnl_not_funded_performance": counterfactual,
                                 "performance_valid": not blocked})
    return pd.DataFrame(rows), pd.DataFrame(decisions)


def main():
    cfg = load_config(ROOT / "configs/carry_pilot.yaml")
    contract = load_config(ROOT / "configs/carry_experiment.yaml")
    frozen = json.loads((ROOT / "manifests/phase2_6_protocol.json").read_text())
    check_frozen(cfg, frozen)
    start, end = require_training_period(cfg["train_start"], cfg["train_end"], contract["splits"])
    if cfg["complex_models_allowed"] or cfg["full_modeling_authorized"]:
        raise ValueError("Full modeling is not authorized")
    out = ROOT / "results/phase2_6"
    out.mkdir(parents=True, exist_ok=True)
    catalog = json.loads((ROOT / "manifests/phase2_dataset.json").read_text())
    old_run = json.loads((ROOT / "manifests/phase2_5_run.json").read_text())
    selected = training_partitions(catalog["partitions"], start, end)
    if len(selected) != 72:
        raise ValueError("Expected exactly 72 training monthly price partitions")
    inputs, all_features, base_rows, scenario_rows, ledger_rows = [], [], [], [], []
    origins = origin_grid(cfg)
    columns = ["bar_open_time_utc"]+[f"{v}_{c}" for v in ("spot", "perp")
               for c in ("open", "high", "low", "close", "volume", "trades")]
    for symbol in cfg["symbols"]:
        print(f"Building fixed-origin pilot labels: {symbol}", flush=True)
        parts = []
        for row in selected:
            if row["symbol"] != symbol:
                continue
            assert file_sha256(ROOT / row["path"]) == row["sha256"]
            parts.append(pd.read_parquet(ROOT / row["path"], columns=columns))
            inputs.append({"path": row["path"], "sha256": row["sha256"], "role": "training_paired"})
        frame = pd.concat(parts).sort_index()
        frame = frame.loc[(frame.index >= start) & (frame.index < end)]
        market = Market(frame)
        source = f"results/phase2_5/{symbol}_funding_notional_proxies.parquet"
        assert file_sha256(ROOT / source) == old_run["outputs"][source]
        events = pd.read_parquet(ROOT / source)
        if not ((events.settlement_time_utc >= start) & (events.settlement_time_utc < end)).all():
            raise ValueError("Funding source contains values outside training")
        inputs.append({"path": source, "sha256": file_sha256(ROOT / source), "role": "training_funding_only"})
        features = input_features(market, events, origins, cfg)
        features.insert(0, "symbol", symbol)
        all_features.append(features)
        for n, origin in enumerate(origins):
            for horizon in cfg["horizons_hours"]:
                for delay in cfg["entry_delays_minutes"]:
                    case = f"{symbol}_{origin.strftime('%Y%m%dT%H%M')}_{horizon}h_d{delay}"
                    row, scenarios, ledger = fixed_outcome(market, events, origin, horizon, delay, cfg)
                    metadata = {"symbol": symbol, "case_id": case, "origin_number": n,
                                "primary_sparse": n % cfg["primary_nonoverlap_stride_days"] == 0}
                    base_rows.append({**metadata, **row})
                    scenario_rows.extend({**metadata, **value} for value in scenarios)
                    ledger_rows.extend({**metadata, **value} for value in ledger)
        del market, frame, parts
    features = pd.concat(all_features, ignore_index=True)
    base = pd.DataFrame(base_rows)
    outcomes = pd.DataFrame(scenario_rows)
    forecasts, coefficients = [], []
    # Funding predictability does not condition on future path completeness/profitability.
    primary = base.loc[(base.delay_minutes == cfg["primary_delay_minutes"])
                       & base.label_status.eq("observed") & base.funding_complete.fillna(False)]
    primary = primary.merge(features.drop(columns=["year", "origin_number", "primary_sparse"]),
                            on=["symbol", "decision_time_utc"], validate="many_to_one")
    primary = primary.loc[primary.funding_features_ready]
    for symbol in cfg["symbols"]:
        for horizon in cfg["horizons_hours"]:
            population = primary.loc[(primary.symbol == symbol) & (primary.horizon_hours == horizon)]
            for fold in cfg["folds"]:
                fit, evaluate = purged_fold(population, fold, cfg["embargo_hours"])
                predictions, coef = funding_forecasts(fit, evaluate, cfg["minimum_fit_rows"])
                for item in coef:
                    coefficients.append({"symbol": symbol, "horizon_hours": horizon, "fold": fold["name"], **item})
                for name, values in predictions.items():
                    for observed, prediction in zip(evaluate.to_dict("records"), values):
                        forecasts.append({"symbol": symbol, "horizon_hours": horizon, "fold": fold["name"],
                                          "decision_time_utc": observed["decision_time_utc"], "case_id": observed["case_id"],
                                          "year": observed["year"], "primary_sparse": observed["primary_sparse"],
                                          "label_available_at_utc": observed["label_available_at_utc"], "model": name,
                                          "actual_funding_sum_bps": observed["future_funding_rate_sum_bps"],
                                          "prediction_bps": float(prediction)})
    policy, policy_log = policy_references(base, features, outcomes, cfg)
    owned = {"features.parquet": features, "label_audit.parquet": base, "outcomes.parquet": outcomes,
             "funding_ledger.parquet": pd.DataFrame(ledger_rows), "funding_forecasts.parquet": pd.DataFrame(forecasts),
             "policy_decisions.parquet": policy_log}
    for filename, value in owned.items():
        value.to_parquet(out / filename, index=False)
    policy.to_csv(out / "policy_summary.csv", index=False)
    write_json(out / "coefficients.json", coefficients)
    write_json(out / "configuration.json", cfg)
    filenames = [*owned, "policy_summary.csv", "coefficients.json", "configuration.json"]
    sources = ["src/spot_perp_basis/carry_pilot.py", "scripts/run_carry_pilot.py"]
    check_frozen(cfg, frozen)
    write_json(ROOT / "manifests/phase2_6_run.json", {
        "created_at_utc": utc_now(), "protocol_sha256": file_sha256(ROOT / "manifests/phase2_6_protocol.json"),
        "config_sha256": frozen["config_sha256"], "source_sha256": {p: file_sha256(ROOT / p) for p in sources},
        "inputs": inputs, "outputs": {str((out / p).relative_to(ROOT)): file_sha256(out / p) for p in filenames},
        "features": len(features), "fixed_origin_cases": len(base), "scenario_outcomes": len(outcomes),
        "forecast_records": len(forecasts), "linear_fits": len(coefficients),
        "python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
        "complex_models_fitted": 0, "full_modeling_started": False, "validation_test_results_accessed": False})
    print(f"Built {len(base)} cases, {len(outcomes)} scenario outcomes, {len(forecasts)} forecast records", flush=True)


if __name__ == "__main__":
    main()
