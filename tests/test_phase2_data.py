"""Phase 2 funding timing, missing-price support and holdout access boundaries."""

from hashlib import sha256
import io
import json
from pathlib import Path
import tempfile
import unittest
from urllib.parse import parse_qs, urlparse
import zipfile

import numpy as np
import pandas as pd

from spot_perp_basis.artifacts import file_sha256
from spot_perp_basis.data.market import audit_market, intervals, pair_bars
from spot_perp_basis.data.windows import require_training_period, training_partitions
from spot_perp_basis.data.ingest import funding_api_pages, parse_funding_archive
from spot_perp_basis.data.funding import join_funding, prepare_funding


class Phase2FundingTests(unittest.TestCase):
    def events(self):
        return pd.DataFrame({"settlement_time_utc": pd.to_datetime(["2020-01-01T00:00:00Z", "2020-01-01T08:00:00.001Z"], format="mixed").as_unit("ns"),
                             "funding_rate": [.001, -.002], "funding_interval_hours": [8., 8.]})

    def test_funding_publication_lag_preserves_subminute_timestamp(self):
        events = prepare_funding(self.events(), 1)
        index = pd.to_datetime(["2020-01-01T00:00Z", "2020-01-01T00:01Z", "2020-01-01T08:00Z", "2020-01-01T08:01Z", "2020-01-01T08:02Z"]).as_unit("ns")
        paired = pd.DataFrame({"price": 100.}, index=index.rename("decision_time_utc"))
        joined = join_funding(paired, events)
        self.assertTrue(np.isnan(joined.last_settled_funding_rate.iloc[0]))
        self.assertEqual(joined.last_settled_funding_rate.iloc[1:4].tolist(), [.001, .001, .001])
        self.assertEqual(joined.last_settled_funding_rate.iloc[4], -.002)
        changed = self.events(); changed.loc[1, "funding_rate"] = 999.
        later = join_funding(paired, prepare_funding(changed, 1))
        pd.testing.assert_frame_equal(joined.iloc[:4], later.iloc[:4])

    def test_stale_rate_is_explicitly_masked_and_future_event_never_backfilled(self):
        events = prepare_funding(self.events(), 1)
        index = pd.to_datetime(["2019-12-31T23:00Z", "2020-01-02T00:00Z"]).as_unit("ns")
        joined = join_funding(pd.DataFrame(index=index.rename("decision_time_utc")), events, 12)
        self.assertFalse(joined.funding_known.any())
        self.assertFalse(joined.funding_stale.iloc[0])
        self.assertTrue(joined.funding_stale.iloc[1])
        self.assertTrue(joined.last_settled_funding_rate.isna().all())

    def test_archive_units_negative_rates_and_duplicate_rejection(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "funding.zip"
            for unit, stamp in (("ms", 1577836800000), ("us", 1577836800000000)):
                body = f"calc_time,funding_interval_hours,last_funding_rate\n{stamp},8,-0.0001\n"
                with zipfile.ZipFile(path, "w") as z:
                    z.writestr("BTCUSDT-fundingRate-2020-01.csv", body)
                events, audit = parse_funding_archive(path, "BTCUSDT", "2020-01", file_sha256(path))
                self.assertEqual(audit["timestamp_unit"], unit)
                self.assertEqual(events.funding_rate.iloc[0], -.0001)
            with zipfile.ZipFile(path, "w") as z:
                z.writestr("BTCUSDT-fundingRate-2020-01.csv", body + body.splitlines()[1] + "\n")
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                parse_funding_archive(path, "BTCUSDT", "2020-01", file_sha256(path))

    def test_api_pagination_advances_past_inclusive_endpoint_and_keeps_last_page(self):
        calls = []
        def request(url):
            args = parse_qs(urlparse(url).query); cursor = int(args["startTime"][0]); calls.append(cursor)
            page = [{"symbol": "BTCUSDT", "fundingTime": t, "fundingRate": "0.0001"} for t in [0, 100, 200] if t >= cursor][:2]
            return json.dumps(page).encode(), {}
        rows = funding_api_pages("BTCUSDT", 0, 200, request=request, limit=2)
        self.assertEqual([r["fundingTime"] for r in rows], [0, 100, 200])
        self.assertEqual(calls, [0, 101])
        def repeated(url):
            return b'[{"symbol":"BTCUSDT","fundingTime":0}]', {}
        with self.assertRaisesRegex(ValueError, "non-advancing"):
            funding_api_pages("BTCUSDT", 0, 200, request=repeated, limit=2)

    def test_cadence_gaps_are_reported_without_synthetic_settlements(self):
        events = self.events()
        events.loc[1, "settlement_time_utc"] = pd.Timestamp("2020-01-02T00:00:00Z")
        prepared = prepare_funding(events)
        self.assertEqual(len(prepared), 2)
        self.assertTrue(prepared.cadence_gap_flag.iloc[1])


class Phase2DatasetTests(unittest.TestCase):
    def test_missing_minute_remains_gap_after_pairing(self):
        index = pd.date_range("2020-01-01", periods=5, freq="min", tz="UTC").as_unit("ns")
        frame = pd.DataFrame({"open":100.,"high":101.,"low":99.,"close":100.,"volume":1.,"quote_volume":100.,
                              "trades":1,"taker_buy_volume":.5,"taker_buy_quote_volume":50.,"valid":True,
                              "observed":True,"bad_price":False,"bad_activity":False,"bad_close_time":False}, index=index)
        spot, _ = audit_market(frame, {})
        perp = spot.copy(); perp.loc[index[2], ["valid", "observed"]] = False
        pair = pair_bars(spot, perp)
        self.assertEqual(len(pair), 4)
        self.assertNotIn(index[2] + pd.Timedelta(minutes=1), pair.index)
        self.assertFalse(pair.loc[index[3] + pd.Timedelta(minutes=1), "contiguous_previous_minute"])
        self.assertNotIn("basis_bps", pair)
        gaps = intervals(index, ~perp.valid)
        self.assertEqual(gaps[0]["minutes"], 1)

    def test_training_path_selection_excludes_validation_and_holdout_before_io(self):
        partitions = [{"symbol":"BTCUSDT","month":m,"path":f"must_not_open/{m}.parquet"}
                      for m in ["2020-01", "2022-12", "2023-01", "2024-01", "2025-12"]]
        selected = training_partitions(partitions, "2020-01-01T00:00Z", "2023-01-01T00:00Z")
        self.assertEqual([r["month"] for r in selected], ["2020-01", "2022-12"])
        splits = {"train":["2020-01-01T00:00Z", "2023-01-01T00:00Z"]}
        with self.assertRaisesRegex(ValueError, "training interval"):
            require_training_period("2020-01-01T00:00Z", "2024-01-01T00:00Z", splits)


if __name__ == "__main__":
    unittest.main()
