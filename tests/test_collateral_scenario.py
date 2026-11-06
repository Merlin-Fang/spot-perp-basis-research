"""Economic identities, cash-only constraints and pre-transfer failures."""

import copy
import json
import unittest

import numpy as np
import pandas as pd

from spot_perp_basis.backtest.collateral import (
    CollateralTimeline,
    Wallet,
    historical_daily_labels,
    position_requirement,
    required_equity,
    reserve_forecast,
    planning_hours,
    simulate_episode,
)
from spot_perp_basis.artifacts import ROOT, file_sha256, load_config
from spot_perp_basis.data.windows import MINUTE_NS


def fixture():
    cfg = copy.deepcopy(load_config(ROOT / "configs/collateral_scenario.yaml"))
    cfg.update(start="2020-01-01T00:00Z", end="2020-01-05T00:00Z", spot_notional=100.)
    times = pd.date_range("2020-01-01T00:01Z", "2020-01-04T00:00Z", freq="min").as_unit("ns")
    perp = pd.DataFrame({"open": 100., "high": 100., "low": 100., "close": 100., "volume": 1.,
                         "trades": 1, "valid": True, "observed": True}, index=times)
    et = pd.date_range("2020-01-01T00:00Z", "2020-01-03T16:00Z", freq="8h").as_unit("ns")
    events = pd.DataFrame({"settlement_time_utc": et, "available_at_utc": et+pd.Timedelta(minutes=1),
                           "funding_rate": 0., "perp_price_proxy": 100.})
    mt = pd.DatetimeIndex(["2022-06-01T00:01Z"]).as_unit("ns")
    mark = pd.DataFrame({"observation_time_utc": mt, "feature_available_at_utc": mt+pd.Timedelta(minutes=1),
                         "close": 100., "high": 100.})
    ep = {"episode_id": "BTCUSDT_fixture", "symbol": "BTCUSDT", "year": 2020, "quarter": "fixture", "status": "observed",
        "calendar_start_utc": pd.Timestamp("2020-01-01T00:00Z"), "calendar_end_utc": pd.Timestamp("2020-01-04T00:00Z"),
        "entry_decision_utc": pd.Timestamp("2020-01-01T09:00Z"), "entry_utc": pd.Timestamp("2020-01-01T09:01Z"),
        "exit_utc": pd.Timestamp("2020-01-03T23:51Z"), "quantity": 1., "spot_entry": 100., "perp_entry": 100.,
        "spot_exit": 100., "perp_exit": 100., "price_pnl": 0., "entry_cost": .19, "exit_cost": .19,
        "seed_spot_decision": 100., "seed_perp_decision": 100.}
    history = pd.DataFrame({"planning_hours": pd.Series(dtype=int), "risk_ratio": pd.Series(dtype=float),
                            "label_available_at_utc": pd.Series([], dtype="datetime64[ns, UTC]")})
    return cfg, perp, events, mark, ep, history


class CollateralScenarioTests(unittest.TestCase):
    def test_wallet_equity_not_cash_or_combined_spot(self):
        w = Wallet(30., 170., 1., 100.)
        self.assertEqual(w.equity(120.), 10.)
        self.assertEqual(w.cash, 30.)
        self.assertEqual(w.treasury, 170.)

    def test_transfers_conserve_cash_and_fee_is_once(self):
        w = Wallet(30., 170., 1., 100.)
        ok, amount, fee = w.transfer(20., 1.)
        self.assertTrue(ok); self.assertEqual(amount, 20.)
        self.assertAlmostEqual(w.cash+w.treasury, 200.-.002)
        self.assertAlmostEqual(w.transfer_cost, fee)
        self.assertEqual(w.equity(100.), 50.)
        self.assertTrue(w.transfer(-10.)[0])
        self.assertAlmostEqual(w.cash+w.treasury, 199.998)

    def test_no_external_cash_rescue(self):
        w = Wallet(30., 2., 1., 100.)
        self.assertFalse(w.transfer(3.)[0])
        self.assertEqual((w.cash, w.treasury), (30., 2.))

    def test_positive_unrealized_profit_cannot_be_withdrawn_as_cash(self):
        w = Wallet(10., 90., 1., 150.)
        command = w.command(30., 100., cash_floor=5.)
        self.assertEqual(command, -5.)
        self.assertTrue(w.transfer(command)[0])
        self.assertEqual(w.cash, 5.)
        self.assertEqual(w.equity(100.), 55.)

    def test_hand_margin_equity_and_existing_entry_price_cash_dependence(self):
        self.assertAlmostEqual(required_equity(2., 100., [100., 120.], [0., 1.], .01), 41.4)
        a = position_requirement(1., 100., 100., [100., 110.], [0., 0.], .01)
        b = position_requirement(1., 150., 100., [100., 110.], [0., 0.], .01)
        self.assertAlmostEqual(a["margin_equity"], b["margin_equity"])
        self.assertAlmostEqual(a["required_cash"], 11.1)
        self.assertEqual(b["required_cash"], 0.)
        self.assertEqual(b["required_equity"], 50.)

    def test_funding_debit_needs_cash_despite_large_unrealized_profit(self):
        r = position_requirement(1., 150., 100., [100., 100.], [0., -10.], .01)
        self.assertEqual(r["required_cash"], 10.)
        self.assertEqual(r["required_equity"], 60.)
        self.assertEqual(r["margin_equity"], 11.)

    def test_funding_boundaries_and_closing_reconciliation(self):
        cfg, p, e, m, ep, h = fixture()
        e.loc[:, "funding_rate"] = .01
        # Entry-equal funding is excluded; exit-equal funding is included.
        et = pd.DatetimeIndex([ep["entry_utc"], ep["exit_utc"]]).as_unit("ns")
        e = pd.DataFrame({"settlement_time_utc": et, "funding_rate": [.5, -.01], "perp_price_proxy": 100.})
        timeline = CollateralTimeline(p, e, m, cfg)
        r, _, _, _ = simulate_episode(timeline, ep, h, cfg, cfg["scenarios"][0], "generous")
        self.assertEqual(r["status"], "completed")
        self.assertAlmostEqual(r["counterfactual_funding"], -1.)
        self.assertAlmostEqual(r["funded_terminal_pnl"], -1.38)
        self.assertLess(r["max_cash_conservation_error"], 1e-9)

    def test_same_hedge_pnl_for_successful_cash_allocation_policies(self):
        cfg, p, e, m, ep, h = fixture()
        e["funding_rate"] = .0001
        timeline = CollateralTimeline(p, e, m, cfg)
        results = [simulate_episode(timeline, ep, h, cfg, cfg["scenarios"][0], policy)[0] for policy in cfg["policies"]]
        self.assertTrue(all(r["status"] == "completed" for r in results))
        np.testing.assert_allclose([r["funded_terminal_pnl"] for r in results], results[0]["funded_terminal_pnl"])
        self.assertEqual({r["initial_total_commitment"] for r in results}, {300.})
        self.assertEqual({r["genuinely_released_cash"] for r in results}, {0.})

    def test_pre_rebalance_breach_is_absorbing(self):
        cfg, p, e, m, ep, h = fixture()
        at = pd.Timestamp("2020-01-02T12:00Z")
        p.loc[at, ["close", "high"]] = 150.
        timeline = CollateralTimeline(p, e, m, cfg)
        r, d, _, _ = simulate_episode(timeline, ep, h, cfg, cfg["scenarios"][0], "fixed30")
        self.assertEqual(r["status"], "margin_shortfall")
        self.assertEqual(r["failure_time_utc"], at)
        self.assertTrue(np.isnan(r["funded_terminal_pnl"]))
        self.assertTrue(all(x["decision_time_utc"] < at for x in d))
        self.assertGreater(r["max_shortfall_usdt"], 20.)
        self.assertGreaterEqual(r["shadow_shortfall_minutes"], 1.)

    def test_delayed_topup_cannot_retroactively_repair_failure(self):
        cfg, p, e, m, ep, h = fixture()
        p.loc["2020-01-02T08:50Z":"2020-01-02T09:29Z", ["close", "high"]] = 125.
        p.loc["2020-01-02T09:30Z", ["close", "high"]] = 145.
        timeline = CollateralTimeline(p, e, m, cfg)
        immediate = simulate_episode(timeline, ep, h, cfg, cfg["scenarios"][0], "fixed30")[0]
        delayed, _, transfers, _ = simulate_episode(timeline, ep, h, cfg, cfg["scenarios"][3], "fixed30")
        self.assertEqual(immediate["status"], "completed")
        self.assertEqual(delayed["status"], "margin_shortfall")
        self.assertEqual(delayed["failure_time_utc"], pd.Timestamp("2020-01-02T09:30Z"))
        self.assertFalse(any(x["execution_time_utc"] >= delayed["failure_time_utc"] for x in transfers))

    def test_unfundable_target_and_generous_inception_are_failures(self):
        cfg, p, e, m, ep, h = fixture()
        p.loc["2020-01-02T08:50Z":"2020-01-02T09:00Z", ["close", "high"]] = 125.
        timeline = CollateralTimeline(p, e, m, cfg)
        limited = cfg["scenarios"][5]
        r = simulate_episode(timeline, ep, h, cfg, limited, "fixed30")[0]
        self.assertEqual(r["status"], "liquidity_infeasible")
        self.assertTrue(np.isnan(r["funded_terminal_pnl"]))
        self.assertEqual(simulate_episode(timeline, ep, h, cfg, limited, "generous")[0]["status"], "infeasible_entry")

    def test_missing_path_unknown_not_safe(self):
        cfg, p, e, m, ep, h = fixture()
        p.loc["2020-01-02T12:00Z", "valid"] = False
        timeline = CollateralTimeline(p, e, m, cfg)
        r = simulate_episode(timeline, ep, h, cfg, cfg["scenarios"][0], "generous")[0]
        self.assertEqual(r["status"], "unknown_path")
        self.assertTrue(np.isnan(r["funded_terminal_pnl"]))

    def test_inactive_unknown_sensitivity(self):
        cfg, p, e, m, ep, h = fixture()
        p.loc["2020-01-02T12:00Z", "volume"] = 0.
        timeline = CollateralTimeline(p, e, m, cfg)
        self.assertEqual(simulate_episode(timeline, ep, h, cfg, cfg["scenarios"][0], "generous")[0]["status"], "completed")
        self.assertEqual(simulate_episode(timeline, ep, h, cfg, cfg["scenarios"][7], "generous")[0]["status"], "unknown_path")

    def test_known_features_and_matured_history_only(self):
        cfg, p, e, m, ep, h = fixture()
        t = pd.Timestamp("2020-01-02T09:00Z")
        h = pd.DataFrame({"planning_hours": [24]*21, "risk_ratio": [.1]*20+[999.],
            "label_available_at_utc": pd.date_range("2019-12-13T09:01Z", periods=20, freq="D").append(pd.DatetimeIndex(["2020-01-03T09:01Z"]))})
        a = reserve_forecast(t.value, 24, CollateralTimeline(p, e, m, cfg), h, cfg, .01, "rolling95")
        p.loc[p.index > t, "close"] = 9999.
        h.loc[h.label_available_at_utc > t, "risk_ratio"] = 0.
        b = reserve_forecast(t.value, 24, CollateralTimeline(p, e, m, cfg), h, cfg, .01, "rolling95")
        self.assertEqual(a, b)
        self.assertLessEqual(a["last_label_available_at_utc"], t)
        self.assertAlmostEqual(a["fraction"], .15)

    def test_training_and_label_boundaries(self):
        cfg, p, e, m, ep, h = fixture()
        timeline = CollateralTimeline(p, e, m, cfg)
        self.assertEqual(timeline.risk(pd.Timestamp("2020-01-04T09:00Z").value,
                                      pd.Timestamp(cfg["end"]).value)["status"], "boundary_censored")
        bad = p.copy(); bad.index = bad.index+pd.DateOffset(years=3)
        with self.assertRaises(ValueError):
            CollateralTimeline(bad, e, m, cfg)
        cfg["historical_label_horizons_hours"] = [24]
        labels = historical_daily_labels(timeline, cfg, .01)
        self.assertTrue((labels.loc[labels.status == "observed", "label_available_at_utc"] < pd.Timestamp(cfg["end"])).all())

    def test_strict_missing_sample_mark_and_publication_delay(self):
        cfg, p, e, m, ep, h = fixture()
        p.index = p.index+pd.DateOffset(years=2, months=5)
        e["settlement_time_utc"] = e.settlement_time_utc+pd.DateOffset(years=2, months=5)
        cfg.update(start="2022-06-01T00:00Z", end="2022-06-05T00:00Z")
        mt = p.index
        m = pd.DataFrame({"observation_time_utc": mt, "feature_available_at_utc": mt+pd.Timedelta(minutes=1),
                          "close": 101., "high": 101.}).drop(index=range(700, 705))
        timeline = CollateralTimeline(p, e, m, cfg)
        t = pd.Timestamp("2022-06-01T09:00Z")
        self.assertLessEqual(timeline.known(t.value)[2], t)
        self.assertEqual(timeline.known(t.value)[0], 101.)
        self.assertFalse(timeline.risk(t.value, (t+pd.Timedelta(hours=24)).value)["valid"])

    def test_maintenance_notional_and_funding_rmq_match_direct_formula(self):
        cfg, p, e, m, ep, h = fixture()
        p.loc["2020-01-02T12:00Z", ["close", "high"]] = 120.
        e["funding_rate"] = -.001
        timeline = CollateralTimeline(p, e, m, cfg)
        t, u = pd.Timestamp("2020-01-02T09:00Z").value, pd.Timestamp("2020-01-03T09:00Z").value
        a, b = timeline.bounds(t, u)
        direct = position_requirement(1., 130., 100., np.r_[100., timeline.close[a:b]],
            np.r_[0., timeline.funding[a:b]-timeline.funding_at(t)], .02)
        fast = timeline.risk(t, u, .02, reference=100., position_entry_price=130.)
        self.assertAlmostEqual(fast["required_equity"], direct["required_equity"])
        self.assertAlmostEqual(fast["required_wallet_cash"], direct["required_cash"])

    def test_old_history_and_both_pre_experiment_freezes_preserved(self):
        old = json.loads((ROOT / "manifests/phase2_7_5_history.json").read_text())
        for group in ("preserved_files", "prior_results"):
            for path, sha in old[group].items():
                self.assertEqual(file_sha256(ROOT / path), sha, path)
        initial = json.loads((ROOT / "manifests/phase2_7_5_protocol_initial.json").read_text())
        self.assertEqual(file_sha256(ROOT / "docs/history/phase2_7_5_protocol_initial.md"), initial["protocol_sha256"])
        self.assertEqual(file_sha256(ROOT / "docs/history/phase2_7_5_config_initial.yaml"), initial["config_sha256"])

    def test_high_excursion_does_not_credit_positive_funding_before_the_high(self):
        cfg, p, e, m, ep, h = fixture()
        p.loc["2020-01-02T16:01Z", "high"] = 110.
        stamp = pd.DatetimeIndex(["2020-01-02T16:00:00.005Z"])
        e = pd.DataFrame({"settlement_time_utc": stamp, "funding_rate": [.01], "perp_price_proxy": [100.]})
        t, u = pd.Timestamp("2020-01-02T15:00Z").value, pd.Timestamp("2020-01-02T17:00Z").value
        positive = CollateralTimeline(p, e, m, cfg, "hybrid_high").risk(t, u)
        self.assertAlmostEqual(positive["margin_required_equity"], 11.1)
        e["funding_rate"] = -.01
        negative = CollateralTimeline(p, e, m, cfg, "hybrid_high").risk(t, u)
        self.assertAlmostEqual(negative["margin_required_equity"], 12.1)

    def test_known_monday_outage_is_planned_without_an_extra_cash_injection(self):
        cfg, p, e, m, ep, h = fixture()
        case = cfg["scenarios"][4]
        self.assertEqual(planning_hours(pd.Timestamp("2020-01-05T09:00Z"), case), 48)
        self.assertEqual(planning_hours(pd.Timestamp("2020-01-06T09:00Z"), case), 24)
        times = pd.date_range("2020-01-01T00:01Z", "2020-01-09T23:59Z", freq="min").as_unit("ns")
        p = p.reindex(times).ffill()  # Synthetic constant fixture only, never production prices.
        cfg["end"] = "2020-01-10T00:00Z"
        ep.update(calendar_end_utc=pd.Timestamp(cfg["end"]), exit_utc=pd.Timestamp("2020-01-09T23:51Z"))
        timeline = CollateralTimeline(p, e, m, cfg)
        r, decisions, transfers, _ = simulate_episode(timeline, ep, h, cfg, case, "fixed30")
        self.assertEqual(r["status"], "completed")
        self.assertTrue(any(d["action"] == "outage_hold" for d in decisions))
        self.assertFalse(any(x["execution_time_utc"].dayofweek == 0 for x in transfers))
        self.assertLess(r["max_cash_conservation_error"], 1e-9)


if __name__ == "__main__":
    unittest.main()
