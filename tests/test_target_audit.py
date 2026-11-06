"""Accounting identities, chronology, strict reference handling and frozen history."""

import copy
from io import BytesIO
import json
import unittest
from unittest.mock import patch
from zipfile import ZipFile

import numpy as np
import pandas as pd

from spot_perp_basis.backtest.carry import fixed_outcome
from spot_perp_basis.data.windows import purged_fold
from spot_perp_basis.artifacts import ROOT, file_sha256, load_config
from spot_perp_basis.data.market import Market, RangeMax
from spot_perp_basis.data.ingest import parse_reference
from spot_perp_basis.data.funding import reference_asof
from spot_perp_basis.data.labels import LabelEngine
from spot_perp_basis.data.features import historical_features, matured_summaries


def config():
    cfg = copy.deepcopy(load_config(ROOT / "configs/target_audit.yaml"))
    cfg.update(end="2020-01-05T00:00Z", primary_horizon_hours=8, spot_notional=100, committed_capital=300)
    return cfg


def path(n=1500):
    opens = pd.date_range("2020-01-01", periods=n, freq="min", tz="UTC")
    f = pd.DataFrame({"bar_open_time_utc": opens}, index=opens+pd.Timedelta(minutes=1))
    for venue in ("spot", "perp"):
        for column in ("open", "high", "low", "close"):
            f[f"{venue}_{column}"] = 100. if venue == "spot" else 101.
        f[f"{venue}_volume"] = 1.
        f[f"{venue}_trades"] = 1
        f[f"{venue}_quote_volume"] = 100.
        f[f"{venue}_taker_buy_quote_volume"] = 40.
    f.loc[f.index >= pd.Timestamp("2020-01-01T08:03Z"), "perp_open"] = 99.
    return f


def events(times=("2020-01-01T08:00Z",), rates=(.01,)):
    times = pd.to_datetime(list(times), utc=True, format="mixed")
    return pd.DataFrame({"settlement_time_utc": times, "available_at_utc": times+pd.Timedelta(minutes=1),
                         "funding_rate": rates, "perp_price_proxy": np.repeat(100., len(times))})


class TargetAuditTests(unittest.TestCase):
    def test_rmq_matches_exhaustive_random_intervals(self):
        rng = np.random.default_rng(0)
        values = rng.normal(size=3000)
        tree = RangeMax(values)
        for _ in range(200):
            a, b = sorted(rng.integers(0, len(values)+1, 2))
            self.assertEqual(tree.query(a, b), max(values[a:b], default=-np.inf))

    def test_hand_pnl_funding_and_delay(self):
        market = Market(path())
        result = LabelEngine(market, events(), config()).outcome(market.frame.index[0])
        self.assertEqual(result["entry_utc"], pd.Timestamp("2020-01-01T00:02Z"))
        self.assertEqual(result["exit_utc"], pd.Timestamp("2020-01-01T08:02Z"))
        self.assertAlmostEqual(result["basis_pnl_usdt"], 2.)
        self.assertAlmostEqual(result["funding_cash_usdt"], 1.)
        self.assertEqual(result["funding_events"], 1)
        self.assertEqual(result["label_available_at_utc"], pd.Timestamp("2020-01-01T08:03Z"))
        self.assertTrue(result["risk_valid"])

    def test_entry_excluded_exit_included_and_negative_funding(self):
        m = Market(path())
        e = events(["2020-01-01T00:02Z", "2020-01-01T08:02Z"], [.9, -.01])
        r = LabelEngine(m, e, config()).outcome(m.frame.index[0])
        self.assertAlmostEqual(r["funding_cash_usdt"], -1.)
        self.assertEqual(r["funding_events"], 1)

    def test_efficient_cash_equals_independent_slow_accountant(self):
        m, e, cfg = Market(path()), events(), config()
        old = load_config(ROOT / "configs/carry_pilot.yaml")
        old.update(train_end=cfg["end"], spot_notional=cfg["spot_notional"], initial_capital=cfg["committed_capital"],
                   cost_scenarios={"hand": {**cfg["costs"], "financing_apr": 0}})
        r = LabelEngine(m, e, cfg).outcome(m.frame.index[0])
        _, slow, _ = fixed_outcome(m, e, m.frame.index[0], 8, 1, old)
        self.assertAlmostEqual(r["collateral_ratio"], slow[0]["required_capital_multiple"]-1, places=12)
        self.assertAlmostEqual(r["collateral_ratio_2pct"], slow[0]["required_cash_maintenance_2pct"]/100-1, places=12)
        self.assertAlmostEqual(r["total_required_capital_usdt"], 100*(1+r["collateral_ratio"]))

    def test_spot_profit_does_not_collateralize_futures(self):
        f = path()
        f.loc[f.index[50]:, ["spot_close", "perp_close"]] = [500., 501.]
        m = Market(f)
        r = LabelEngine(m, events(), config()).outcome(m.frame.index[0])
        self.assertGreater(r["collateral_ratio"], 4.)
        self.assertFalse(r["feasible_at_committed_capital"])
        self.assertGreater(r["extra_cash_beyond_reserve_usdt"], 200)

    def test_missing_future_path_is_unknown(self):
        f = path().drop(path().index[50])
        m = Market(f)
        r = LabelEngine(m, events(), config()).outcome(m.frame.index[0])
        self.assertEqual(r["missing_minutes"], 1)
        self.assertFalse(r["risk_valid"])
        self.assertTrue(np.isnan(r["collateral_ratio"]))
        self.assertFalse(r["feasible_at_committed_capital"])

    def test_inactive_origin_entry_wait_and_future_inactivity(self):
        f = path()
        f.iloc[2:4, f.columns.get_loc("spot_volume")] = 0
        f.iloc[50, f.columns.get_loc("perp_trades")] = 0
        m = Market(f)
        r = LabelEngine(m, events(), config()).outcome(m.frame.index[0])
        self.assertEqual(r["entry_utc"].minute, 4)
        self.assertEqual(r["exit_utc"].minute, 4)
        self.assertEqual(r["inactive_minutes"], 1)
        self.assertEqual(LabelEngine(m, events(), config()).outcome(m.frame.index[2])["label_status"], "unavailable_origin")
        f.iloc[2:10, f.columns.get_loc("spot_volume")] = 0
        self.assertEqual(LabelEngine(Market(f), events(), config()).outcome(m.frame.index[0])["label_status"], "unfilled_entry")

    def test_training_guard_and_boundary_censor(self):
        cfg = config()
        cfg["end"] = "2020-01-01T08:03Z"
        f = path(481)
        r = LabelEngine(Market(f), events(), cfg).outcome(f.index[0])
        self.assertEqual(r["label_status"], "censored_training_boundary")
        with self.assertRaises(ValueError):
            LabelEngine(Market(path()), events(["2023-01-01"], [.1]), config())
        with self.assertRaises(ValueError):
            LabelEngine(Market(path()), events(), config()).outcome("2023-01-01T00:00Z")

    def test_features_unchanged_when_only_future_values_change(self):
        f, e = path(), events(["2020-01-01T00:00Z", "2020-01-01T01:00:00.005Z", "2020-01-01T08:00Z"], [.0001, .02, .3])
        origin = pd.Timestamp("2020-01-01T01:01Z")
        a = historical_features(Market(f), e, [origin], config())
        f.loc[f.index > origin, "perp_close"] *= 4
        e.loc[e.available_at_utc > origin, "funding_rate"] = 9999
        b = historical_features(Market(f), e, [origin], config())
        pd.testing.assert_frame_equal(a, b)
        self.assertEqual(a.funding_last_bps.iloc[0], 1.)
        self.assertLessEqual(a.funding_last_available_at_utc.iloc[0], origin)

    def test_matured_labels_only(self):
        cfg = config(); cfg["minimum_matured_label_history"] = 1
        t = pd.to_datetime(["2020-01-01T00:00Z", "2020-01-01T01:00Z"], utc=True)
        labels = pd.DataFrame({"label_status": ["observed"]*2, "label_available_at_utc": t,
                               "basis_pnl_bps": [1., 999.], "funding_cash_bps": [2., 999.], "collateral_ratio": [.1, 9.]})
        r = matured_summaries(labels, [pd.Timestamp("2020-01-01T00:30Z")], cfg)
        self.assertEqual(r.basis_pnl_bps_mean30.iloc[0], 1.)

    def test_split_purge_uses_availability_and_embargo(self):
        data = pd.DataFrame({"decision_time_utc": pd.to_datetime(["2020-12-20", "2020-12-28", "2021-12-30"], utc=True),
                             "label_available_at_utc": pd.to_datetime(["2020-12-24", "2020-12-31", "2022-01-02"], utc=True)})
        fit, evaluation = purged_fold(data, load_config(ROOT / "configs/target_audit.yaml")["folds"][0], 72)
        self.assertEqual(list(fit.index), [0]); self.assertEqual(len(evaluation), 0)

    def test_reference_asof_strict_no_fallback_and_extra_feature_lag(self):
        times = pd.to_datetime(["2020-01-01T00:01Z"], utc=True)
        f = pd.DataFrame({"observation_time_utc": times, "feature_available_at_utc": times+pd.Timedelta(minutes=1), "close": [101.]})
        query = pd.to_datetime(["2020-01-01T00:01Z", "2020-01-01T00:02Z", "2020-01-01T00:05Z"], utc=True)
        values, _ = reference_asof(f, query, feature=True)
        np.testing.assert_allclose(values, [np.nan, 101., np.nan], equal_nan=True)

    def test_premium_can_be_negative_but_mark_cannot(self):
        raw = "1577836800000,-0.1,0,-0.2,-0.1,0,1577836859999,0,60,0,0,0\n"
        stream = BytesIO()
        with ZipFile(stream, "w") as z:
            z.writestr("sample.csv", raw)
        frame, audit = parse_reference(stream.getvalue(), "premiumIndexKlines", "2020-01")
        self.assertLess(frame.close.iloc[0], 0)
        self.assertFalse(audit["activity_fields_are_observed_trades"])
        with self.assertRaises(ValueError):
            parse_reference(stream.getvalue(), "markPriceKlines", "2020-01")

    def test_original_history_unchanged(self):
        frozen = json.loads((ROOT / "manifests/phase2_7_history.json").read_text())
        for group in ("preserved_files", "prior_results"):
            for path, digest in frozen[group].items():
                self.assertEqual(file_sha256(ROOT / path), digest, path)

    def test_unavailable_funding_notional_is_not_zero_income(self):
        e = events()
        e["perp_price_proxy"] = np.nan
        m = Market(path())
        result = LabelEngine(m, e, config()).outcome(m.frame.index[0])
        self.assertFalse(result["funding_complete"])
        self.assertTrue(np.isnan(result["funding_cash_bps"]))
        self.assertTrue(np.isnan(result["economic_net_bps"]))
        self.assertTrue(np.isnan(result["collateral_ratio"]))

    def test_matured_merge_handles_mixed_datetime_resolution(self):
        cfg = config(); cfg["minimum_matured_label_history"] = 1
        labels = pd.DataFrame({"label_status": ["observed"],
            "label_available_at_utc": pd.DatetimeIndex(["2020-01-01T00:01Z"]).as_unit("ns"),
            "basis_pnl_bps": [1.], "funding_cash_bps": [2.], "collateral_ratio": [.1]})
        origins = pd.date_range("2020-01-01T00:00Z", periods=2, freq="h").as_unit("us")
        result = matured_summaries(labels, origins, cfg)
        self.assertTrue(np.isnan(result.basis_pnl_bps_last.iloc[0]))
        self.assertEqual(result.basis_pnl_bps_last.iloc[1], 1.)

    def test_missing_mark_target_does_not_use_complete_trade_path(self):
        from scripts.run_target_audit import mark_sample
        f, cfg = path(), config()
        cfg["supplementary"]["sample_month"] = "2020-01"
        market = Market(f)
        labels = pd.DataFrame([LabelEngine(market, events(), cfg).outcome(f.index[0])])
        features = historical_features(market, events(), [f.index[0]], cfg)
        features["symbol"] = "BTCUSDT"
        reference = pd.DataFrame({"bar_open_time_utc": f.bar_open_time_utc.to_numpy(),
            "observation_time_utc": f.index, "feature_available_at_utc": f.index+pd.Timedelta(minutes=1),
            "close": np.repeat(101., len(f))}).drop(index=range(200, 207))
        with patch("scripts.run_target_audit.pd.read_parquet", return_value=reference):
            marked, _ = mark_sample(market, events(), labels, features, "BTCUSDT", cfg)
        self.assertTrue(labels.risk_valid.iloc[0])
        self.assertFalse(marked.mark_complete.iloc[0])
        self.assertNotIn("mark_collateral_ratio", marked.columns)


if __name__ == "__main__":
    unittest.main()
