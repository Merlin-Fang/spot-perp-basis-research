"""Independent hand paths and chronology tests for fixed-horizon carry labels."""

import copy
import unittest

import numpy as np
import pandas as pd

from spot_perp_basis.evaluation.statistics import block_mean_interval
from spot_perp_basis.backtest.carry import fixed_outcome
from spot_perp_basis.models.baselines import funding_forecasts
from spot_perp_basis.data.features import input_features, origin_grid
from spot_perp_basis.data.windows import purged_fold
from spot_perp_basis.artifacts import ROOT, load_config
from spot_perp_basis.data.market import Market


def config():
    cfg = copy.deepcopy(load_config(ROOT / "configs/carry_pilot.yaml"))
    cfg.update(train_end="2020-01-05T00:00Z", spot_notional=100, initial_capital=300)
    cfg["cost_scenarios"] = {"hand": {"spot_fee_bps": 10, "perp_fee_bps": 20,
                                      "slippage_bps": 1, "financing_apr": 0}}
    return cfg


def path(n=490):
    opens = pd.date_range("2020-01-01", periods=n, freq="min", tz="UTC")
    f = pd.DataFrame({"bar_open_time_utc": opens}, index=opens+pd.Timedelta(minutes=1))
    for venue in ("spot", "perp"):
        for column in ("open", "high", "low", "close"):
            f[f"{venue}_{column}"] = 100. if venue == "spot" else 101.
        f[f"{venue}_volume"] = 1.
        f[f"{venue}_trades"] = 1
    for column in ("open", "high", "low", "close"):
        f.iloc[482:, f.columns.get_loc("perp_"+column)] = 99.
    return f


def events(times=(), rates=()):
    t = pd.to_datetime(list(times), utc=True, format="mixed")
    return pd.DataFrame({"settlement_time_utc": t, "available_at_utc": t+pd.Timedelta(minutes=1),
                         "funding_rate": list(rates), "perp_price_proxy": np.repeat(100., len(t))})


class CarryPilotTests(unittest.TestCase):
    def test_hand_price_funding_four_costs_and_delayed_times(self):
        m = Market(path())
        e = events(["2020-01-01T00:02Z", "2020-01-01T08:00:00.003Z", "2020-01-01T08:03Z"], [.9, .01, .9])
        base, scenarios, ledger = fixed_outcome(m, e, m.frame.index[0], 8, 1, config())
        self.assertEqual(base["entry_utc"], pd.Timestamp("2020-01-01T00:02Z"))
        self.assertEqual(base["exit_utc"], pd.Timestamp("2020-01-01T08:02Z"))
        self.assertEqual(base["label_available_at_utc"], pd.Timestamp("2020-01-01T08:03Z"))
        self.assertEqual(base["funding_event_count"], 1)
        self.assertEqual(len(ledger), 1)
        self.assertAlmostEqual(base["gross_price_pnl"], 2)
        self.assertAlmostEqual(base["funding_cash_proxy"], 1)
        self.assertAlmostEqual(scenarios[0]["fees"], .6)
        self.assertAlmostEqual(scenarios[0]["slippage"], .04)
        self.assertAlmostEqual(scenarios[0]["net_pnl"], 2.36)
        self.assertAlmostEqual(scenarios[0]["net_return_bps"], 2.36/300*1e4)
        self.assertTrue(scenarios[0]["feasible_complete"])

    def test_exit_equality_funding_included_entry_equality_excluded(self):
        m = Market(path())
        base, _, ledger = fixed_outcome(m, events(["2020-01-01T00:02Z", "2020-01-01T08:02Z"], [.99, -.01]),
                                        m.frame.index[0], 8, 1, config())
        self.assertEqual(len(ledger), 1)
        self.assertAlmostEqual(base["funding_cash_proxy"], -1)

    def test_gap_risk_is_unknown_and_cannot_be_success(self):
        f = path().drop(path().index[100])
        m = Market(f)
        base, scenarios, _ = fixed_outcome(m, events(["2020-01-01T08:00Z"], [.01]), m.frame.index[0], 8, 1, config())
        self.assertEqual(base["missing_paired_minutes"], 1)
        self.assertFalse(scenarios[0]["risk_complete"])
        self.assertEqual(scenarios[0]["risk_status"], "unknown")
        self.assertFalse(scenarios[0]["feasible_complete"])
        self.assertGreater(scenarios[0]["net_pnl"], 0)  # Positive counterfactual is still not feasible.

    def test_inactive_entry_waits_bounded_and_exit_uses_actual_entry_horizon(self):
        f = path()
        f.iloc[2:4, f.columns.get_loc("spot_volume")] = 0
        m = Market(f)
        base, _, _ = fixed_outcome(m, events(["2020-01-01T08:00Z"], [.01]), m.frame.index[0], 8, 1, config())
        self.assertEqual(base["entry_utc"].minute, 4)
        self.assertEqual(base["exit_utc"].minute, 4)
        self.assertEqual(base["duration_hours"], 8)
        f.iloc[2:10, f.columns.get_loc("spot_volume")] = 0
        base, scenarios, _ = fixed_outcome(Market(f), events(), m.frame.index[0], 8, 1, config())
        self.assertEqual(base["label_status"], "unfilled_entry")
        self.assertEqual(scenarios, [])

    def test_boundary_censor_requires_label_availability_strictly_inside_training(self):
        cfg = config()
        cfg["train_end"] = "2020-01-01T08:03Z"
        m = Market(path())
        base, scenarios, _ = fixed_outcome(m, events(), m.frame.index[0], 8, 1, cfg)
        self.assertEqual(base["label_status"], "censored_training_boundary")
        self.assertEqual(scenarios, [])
        with self.assertRaises(ValueError):
            fixed_outcome(m, events(), m.frame.index[0], 8, 0, cfg)

    def test_million_notional_finance_uses_calendar_time_without_overflow(self):
        cfg = config()
        cfg.update(spot_notional=1_000_000, initial_capital=3_000_000)
        cfg["cost_scenarios"]["hand"]["financing_apr"] = .05
        m = Market(path(1500))
        e = events(["2020-01-01T08:00Z", "2020-01-01T16:00Z", "2020-01-02T00:00Z"], [.0001]*3)
        base, scenarios, _ = fixed_outcome(m, e, m.frame.index[0], 24, 1, cfg)
        self.assertEqual(base["duration_hours"], 24)
        self.assertAlmostEqual(scenarios[0]["financing_hurdle"], 1_000_000*.05/365)

    def test_unrealized_spot_gains_are_not_futures_cash(self):
        f = path()
        for c in ("open", "high", "low", "close"):
            f.iloc[20:, f.columns.get_loc("spot_"+c)] = 500
            f.iloc[20:, f.columns.get_loc("perp_"+c)] = 501
        m = Market(f)
        _, scenarios, _ = fixed_outcome(m, events(["2020-01-01T08:00Z"], [0]), m.frame.index[0], 8, 1, config())
        self.assertTrue(scenarios[0]["observed_margin_breach"])
        self.assertGreater(scenarios[0]["required_initial_cash"], 500)
        self.assertFalse(scenarios[0]["feasible_complete"])

    def test_feature_funding_publication_and_future_mutation(self):
        f = path()
        m = Market(f)
        cfg = config()
        origin = pd.Timestamp("2020-01-01T01:01Z")
        e = events(["2020-01-01T00:00Z", "2020-01-01T01:00:00.005Z", "2020-01-01T02:00Z"], [.0001, .001, .1])
        a = input_features(m, e, [origin], cfg).iloc[0]
        self.assertEqual(a.funding_last_bps, 1)
        future = f.index > origin
        f.loc[future, "spot_close"] = 9999
        e.loc[e.available_at_utc > origin, "funding_rate"] = 9999
        b = input_features(Market(f), e, [origin], cfg).iloc[0]
        self.assertEqual(a.funding_last_bps, b.funding_last_bps)
        self.assertEqual(a.basis_bps, b.basis_bps)
        self.assertLessEqual(a.funding_last_available_at_utc, origin)

    def test_purging_embargo_and_eval_endpoint(self):
        frame = pd.DataFrame({"decision_time_utc": pd.to_datetime(["2020-12-26T00:00Z",
            "2020-12-28T00:00Z", "2021-01-01T09:00Z", "2021-12-31T09:00Z"], utc=True, format="mixed"),
            "label_available_at_utc": pd.to_datetime(["2020-12-27T00:00Z", "2020-12-29T00:00Z",
                "2021-01-02T09:02Z", "2022-01-01T09:02Z"], utc=True)})
        fold = {"fit_start": "2020-01-01T00:00Z", "eval_start": "2021-01-01T00:00Z", "eval_end": "2022-01-01T00:00Z"}
        fit, evaluation = purged_fold(frame, fold, 72)
        self.assertEqual(list(fit.index), [0])
        self.assertEqual(list(evaluation.index), [2])

    def test_simple_forecasts_never_fit_eval_targets(self):
        n = 120
        a = np.linspace(-2, 3, n)
        b = np.sin(np.arange(n))
        fit = pd.DataFrame({"funding_last_bps": a, "funding_mean_7d_bps": b,
                            "future_funding_rate_sum_bps": 2+3*a+4*b,
                            "label_available_at_utc": pd.date_range("2020-01-01", periods=n, tz="UTC")})
        evaluation = pd.DataFrame({"horizon_hours": [24, 24], "funding_last_bps": [1., 2.],
            "funding_mean_1d_bps": [.5, .5], "funding_mean_7d_bps": [.3, .4], "funding_mean_30d_bps": [.2, .2],
            "funding_expanding_mean_bps": [.1, .1], "future_funding_rate_sum_bps": [0, 0]})
        p, c = funding_forecasts(fit, evaluation)
        np.testing.assert_allclose(p["linear_last_and_7d"], [6.2, 9.6])
        np.testing.assert_allclose(p["persistence"], [3, 6])
        evaluation["future_funding_rate_sum_bps"] = 99999
        q, d = funding_forecasts(fit, evaluation)
        self.assertEqual(c, d)
        for name in p:
            np.testing.assert_array_equal(p[name], q[name])

    def test_fixed_sparse_grid_does_not_overlap_72hour_labels(self):
        cfg = load_config(ROOT / "configs/carry_pilot.yaml")
        g = origin_grid(cfg)[::cfg["primary_nonoverlap_stride_days"]]
        self.assertEqual(g[0], pd.Timestamp("2020-02-01T09:00Z"))
        self.assertTrue((g[1:]-g[:-1] > pd.Timedelta(hours=72, minutes=10)).all())
        self.assertLess(g[-1], pd.Timestamp("2023-01-01T00:00Z"))

    def test_block_interval_constant_loss_difference(self):
        self.assertEqual(block_mean_interval([np.repeat(-2., 30), np.repeat(-2., 30)], repetitions=100), (-2., -2.))

    def policy_fixture(self, feasible, pnl, cash):
        from scripts.run_carry_pilot import policy_references
        cfg = config()
        cfg.update(symbols=["BTCUSDT"], horizons_hours=[24], primary_scenario="hand")
        cfg["folds"] = [{"name": "eval_2021", "eval_start": "2021-01-01T00:00Z", "eval_end": "2022-01-01T00:00Z"}]
        dates = pd.to_datetime(["2021-01-01T09:00Z", "2021-01-05T09:00Z"], utc=True)
        base = pd.DataFrame({"symbol": ["BTCUSDT"]*2, "horizon_hours": [24]*2, "delay_minutes": [1]*2,
                             "primary_sparse": [True]*2, "decision_time_utc": dates, "case_id": ["a", "b"]})
        features = pd.DataFrame({"symbol": ["BTCUSDT"]*2, "decision_time_utc": dates,
                                 "spot_close": [100.]*2, "perp_close": [101.]*2,
                                 "funding_last_bps": [100.]*2, "funding_features_ready": [True]*2})
        outcomes = pd.DataFrame({"scenario": ["hand"]*2, "symbol": ["BTCUSDT"]*2, "horizon_hours": [24]*2,
                                 "delay_minutes": [1]*2, "case_id": ["a", "b"], "feasible_complete": feasible,
                                 "net_pnl": pnl, "required_initial_cash": cash,
                                 "exit_utc": dates+pd.Timedelta(hours=24, minutes=1)})
        return policy_references(base, features, outcomes, cfg)

    def test_policy_unknown_outcome_never_credits_a_counterfactual_winner(self):
        summary, log = self.policy_fixture([False, True], [10., 20.], [110., 110.])
        row = summary.loc[summary.policy == "always_carry"].iloc[0]
        self.assertFalse(row.performance_valid)
        self.assertTrue(pd.isna(row.funded_net_pnl))
        self.assertEqual(row.counterfactual_observed_pnl_not_funded_performance, 30)
        self.assertTrue(log.loc[log.policy == "always_carry", "credited_net_pnl"].isna().all())
        self.assertEqual(summary.loc[summary.policy == "no_trade", "funded_net_pnl"].iloc[0], 0)

    def test_policy_prior_losses_reduce_available_cash_without_injections(self):
        summary, log = self.policy_fixture([True, True], [-20., 100.], [110., 295.])
        row = summary.loc[summary.policy == "always_carry"].iloc[0]
        self.assertFalse(row.performance_valid)
        self.assertEqual(row.funded_completed_before_any_failure, 1)
        self.assertTrue(pd.isna(row.funded_net_pnl))


if __name__ == "__main__":
    unittest.main()
