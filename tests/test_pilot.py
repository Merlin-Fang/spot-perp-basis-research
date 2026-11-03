"""Synthetic checks for market/file units, exact timing, gaps and pilot leakage."""

from pathlib import Path
import tempfile
import unittest
import zipfile

import numpy as np
import pandas as pd

from spot_perp_basis.artifacts import file_sha256
from spot_perp_basis.data.ingest import COLUMNS, parse_klines
from spot_perp_basis.data.market import MINUTE, align_markets
from spot_perp_basis.evaluation.metrics import benchmark
from spot_perp_basis.data.labels import horizon_targets


class PilotTests(unittest.TestCase):
    def archive(self, folder, unit="ms", rows=None, header=False):
        step = {"ms": 60_000, "us": 60_000_000}[unit]
        stamp = pd.Timestamp("2025-01-01T00:00:00Z").value // {"ms": 1_000_000, "us": 1_000}[unit]
        row = [stamp, 100, 101, 99, 100, 2, stamp + step - 1, 200, 10, 1, 100, 0]
        rows = [row] if rows is None else rows(row)
        body = ([",".join(COLUMNS)] if header else []) + [",".join(map(str, r)) for r in rows]
        path = Path(folder) / f"{unit}.zip"
        with zipfile.ZipFile(path, "w") as z:
            z.writestr("BTCUSDT-1m-2025-01.csv", "\n".join(body) + "\n")
        return path

    def test_each_market_file_detects_units_and_completed_bar_time(self):
        with tempfile.TemporaryDirectory() as folder:
            bars = {}
            for market, unit in (("spot", "us"), ("usdm_perpetual", "ms")):
                path = self.archive(folder, unit, header=market != "spot")
                bars[market], audit = parse_klines(path, "BTCUSDT", "2025-01", market, file_sha256(path))
                self.assertEqual(audit["timestamp_unit"], unit)
                self.assertEqual(audit["observed_minutes"], 1)
                self.assertEqual(audit["missing_minutes"], 44639)
            joined, grid = align_markets(bars["spot"], bars["usdm_perpetual"])
            self.assertEqual(joined.index[0], pd.Timestamp("2025-01-01T00:01:00Z"))
            self.assertEqual(joined.iloc[0].bar_open_time_utc + MINUTE, joined.index[0])
            self.assertAlmostEqual(joined.iloc[0].basis_bps, 0)
            self.assertTrue(grid.basis_bps.iloc[1:].isna().all())

    def test_duplicates_mixed_units_and_malformed_rows_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            builders = [lambda r: [r, r], lambda r: [r, [r[0] * 1000, *r[1:]]], lambda r: [r[:-1]]]
            for build in builders:
                path = self.archive(folder, rows=build)
                with self.subTest(build=build), self.assertRaises(ValueError):
                    parse_klines(path, "BTCUSDT", "2025-01", "spot", file_sha256(path))

    def test_bad_price_or_incomplete_bar_is_flagged_and_excluded(self):
        with tempfile.TemporaryDirectory() as folder:
            for column, value, flag in ((4, 200, "bad_price"), (6, 0, "bad_close_time"), (8, -1, "bad_activity")):
                def bad(row):
                    row[column] = value
                    return [row]
                path = self.archive(folder, rows=bad)
                bars, audit = parse_klines(path, "BTCUSDT", "2025-01", "spot", file_sha256(path))
                self.assertEqual(audit[flag], 1)
                self.assertFalse(bars.valid.iloc[0])

    def test_exact_basis_sign_and_horizon_from_decision_time(self):
        index = pd.date_range("2022-06-01", periods=40, freq="min", tz="UTC").as_unit("ns")
        spot = pd.DataFrame({c: 100.0 for c in ["open", "high", "low", "close"]}, index=index)
        for c in ["volume", "quote_volume", "trades", "taker_buy_volume", "taker_buy_quote_volume"]:
            spot[c] = 1.0
        spot["valid"] = True
        perp = spot.copy()
        perp["close"] = 100 * np.exp(np.arange(40) / 10_000)
        joined, grid = align_markets(spot, perp)
        target = horizon_targets(grid, 15)
        self.assertAlmostEqual(joined.basis_bps.iloc[1], 1.0)
        self.assertAlmostEqual(target.target_bps.iloc[0], 15.0)
        self.assertEqual(target.target_end_utc.iloc[0], index[0] + 16 * MINUTE)
        self.assertTrue(target.target_bps.iloc[-15:].isna().all())

    def test_missing_minute_breaks_all_crossing_windows_without_compressing_time(self):
        index = pd.date_range("2022-06-01", periods=40, freq="min", tz="UTC").as_unit("ns")
        frame = pd.DataFrame({"basis_bps": np.arange(40, dtype=float)}, index=index)
        frame.iloc[16] = np.nan
        target = horizon_targets(frame, 15)
        self.assertTrue(target.target_bps.iloc[1:17].isna().all())
        self.assertAlmostEqual(target.target_bps.iloc[0], 15)
        self.assertAlmostEqual(target.target_bps.iloc[17], 15)
        with self.assertRaises(ValueError):
            horizon_targets(frame.drop(index[16]), 15)

    def test_calibration_is_purged_and_future_evaluation_cannot_change_fit(self):
        index = pd.date_range("2022-06-01", periods=72, freq="min", tz="UTC").as_unit("ns")
        frame = pd.DataFrame({"basis_bps": np.sin(np.arange(72) / 5)}, index=index)
        cutoff, end = index[40], index[-1] + MINUTE
        targets = horizon_targets(frame, 15)
        model, _, evaluation = benchmark(targets, cutoff, end)
        changed = frame.copy()
        changed.loc[changed.index >= cutoff] *= 100
        other, _, _ = benchmark(horizon_targets(changed, 15), cutoff, end)
        self.assertEqual(model, other)
        self.assertEqual(model["fit_samples"], 25)
        self.assertTrue((evaluation.index >= cutoff).all())
        self.assertTrue(evaluation.target_end_utc.lt(end).all())
        self.assertEqual(len(evaluation), 17)


if __name__ == "__main__":
    unittest.main()
