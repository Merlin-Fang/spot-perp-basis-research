"""Hand-computed paths and independent timing/accounting assertions."""

import unittest

import numpy as np
import pandas as pd

from spot_perp_basis.data.market import Market, asof_prices
from spot_perp_basis.backtest.accounting import fill_cost, funding_payment
from spot_perp_basis.backtest.carry import build_accounting, carry_trades, evaluate, threshold_trades
from spot_perp_basis.evaluation.metrics import return_metrics
from spot_perp_basis.data.windows import require_training_period, training_partitions

ZERO = {"spot_fee_bps": 0, "perp_fee_bps": 0, "slippage_bps": 0}


def market(perp, spot=None, missing=None, zero=None, opens=None):
    n = len(perp)
    spot = np.repeat(100., n) if spot is None else np.array(spot, dtype=float)
    dates = pd.date_range("2020-01-01", periods=n, freq="min", tz="UTC")
    frame = pd.DataFrame({"bar_open_time_utc": dates}, index=dates+pd.Timedelta(minutes=1))
    for venue, values in (("spot", spot), ("perp", perp)):
        for column in ("open", "high", "low", "close"):
            frame[f"{venue}_{column}"] = np.array(values, dtype=float)
        frame[f"{venue}_volume"] = 1.
        frame[f"{venue}_trades"] = 10
    if opens:
        for row, (s, f) in opens.items():
            frame.iloc[row, frame.columns.get_loc("spot_open")] = s
            frame.iloc[row, frame.columns.get_loc("perp_open")] = f
    if zero:
        frame.iloc[zero, frame.columns.get_loc("spot_volume")] = 0
        frame.iloc[zero, frame.columns.get_loc("spot_trades")] = 0
    if missing:
        frame = frame.drop(frame.index[missing])
    return Market(frame)


def events(times=(), rates=(), prices=()):
    return pd.DataFrame({"settlement_time_utc": pd.to_datetime(list(times), utc=True, format="mixed"),
                         "funding_rate": list(rates), "perp_price_proxy": list(prices),
                         "spot_price_proxy": list(prices)})


def account(m, trades, e=None, cost=ZERO, **kwargs):
    start = "2020-01-01T00:00:00Z"
    end = "2020-01-02T00:00:00Z"
    p, r = build_accounting(m, trades, events() if e is None else e, start, end, notional=100)
    return evaluate(m, p, r, cost, start, end, capital=300, **kwargs)


class EconomicsTests(unittest.TestCase):
    def test_fixed_inclusive_thresholds_next_bar_and_no_overlap(self):
        m = market([100.5, 100.8, 100.7, 99.9, 99.8, 99.7, 100.5, 100.6, 100.7])
        trades, opp = threshold_trades(m)
        self.assertEqual(opp["all"], 6)
        self.assertEqual(len(trades), 2)
        self.assertEqual(trades[0]["entry_signal_row"], 0)
        self.assertEqual(trades[0]["entry_row"], 2)  # signal at 00:01, fill at 00:02
        self.assertEqual(trades[0]["exit_signal_row"], 3)
        self.assertEqual(trades[0]["exit_row"], 5)
        self.assertIsNone(trades[1]["exit_row"])
        self.assertGreater(m.times[trades[1]["entry_signal_row"]], m.opens[trades[0]["exit_row"]])
        with self.assertRaises(ValueError):
            threshold_trades(m, delay=0)

    def test_equal_quantity_signed_price_pnl_and_four_fee_charges(self):
        m = market([101, 101, 101, 99, 99, 99])
        trades, _ = threshold_trades(m)
        cost = {"spot_fee_bps": 10, "perp_fee_bps": 20, "slippage_bps": 1}
        t, daily, s, _ = account(m, trades, cost=cost)
        row = t.iloc[0]
        self.assertEqual(row.status, "closed")
        self.assertAlmostEqual(row.quantity, 1)
        self.assertAlmostEqual(row.spot_pnl, 0)
        self.assertAlmostEqual(row.perp_pnl, 2)
        self.assertAlmostEqual(row.gross_price_pnl, 2)
        self.assertEqual(row.fill_count, 4)
        # 100*.001 + 101*.002 + 100*.001 + 99*.002 = .6 USDT
        self.assertAlmostEqual(row.fees, .6)
        self.assertAlmostEqual(row.slippage, .04)
        self.assertAlmostEqual(row.net_pnl, 1.36)
        self.assertAlmostEqual(s["total_net_pnl"], 1.36)
        self.assertAlmostEqual(daily.equity.iloc[-1], 301.36)
        self.assertAlmostEqual(row.break_even_uniform_bps_per_fill, 50)

    def test_spot_and_perp_both_marked(self):
        m = market([101, 101, 101, 99, 99, 99], spot=[100, 100, 100, 102, 102, 102])
        trades, _ = threshold_trades(m)
        t, _, s, _ = account(m, trades)
        self.assertAlmostEqual(t.spot_pnl.iloc[0], 2)
        self.assertAlmostEqual(t.perp_pnl.iloc[0], 2)
        self.assertAlmostEqual(s["total_net_pnl"], 4)

    def test_funding_signs_boundaries_actual_events_no_repeat(self):
        self.assertEqual(funding_payment(2, .01, 100, -1), 2)
        self.assertEqual(funding_payment(2, .01, 100, 1), -2)
        self.assertEqual(funding_payment(2, -.01, 100, -1), -2)
        m = market([101, 101, 101, 99, 99, 99])
        trades, _ = threshold_trades(m)
        e = events(["2020-01-01 00:02Z", "2020-01-01 00:03:00.123Z", "2020-01-01 00:05Z",
                    "2020-01-01 00:06Z"], [.9, .01, -.005, .9], [100]*4)
        t, _, s, _ = account(m, trades, e)
        # Entry equality excluded; interior +1 and exit equality -.5; later ignored.
        self.assertEqual(t.funding_events.iloc[0], 2)
        self.assertAlmostEqual(t.funding_pnl.iloc[0], .5)
        self.assertAlmostEqual(s["total_net_pnl"], 2.5)
        no, _, sn, _ = account(m, trades, e, funding_on=False)
        self.assertAlmostEqual(sn["total_net_pnl"], 2)
        self.assertAlmostEqual(no.funding_pnl.iloc[0], 0)

    def test_no_future_funding_price(self):
        p, age = asof_prices(pd.to_datetime(["2020-01-01 00:01:30Z", "2019-12-31T00:00Z"], utc=True, format="mixed"),
                             pd.date_range("2020-01-01 00:01", periods=2, freq="min", tz="UTC"), [100, 999])
        self.assertEqual(p[0], 100)
        self.assertEqual(age[0], .5)
        self.assertTrue(np.isnan(p[1]))

    def test_missing_and_zero_activity_wait_without_retroactive_fill(self):
        m = market([101, 101, 101, 101, 101, 101, 99, 99, 99], missing=[2], zero=[3])
        trades, _ = threshold_trades(m)
        self.assertEqual(pd.Timestamp(m.opens[trades[0]["entry_row"]], tz="UTC").minute, 4)
        t, _, _, _ = account(m, trades)
        self.assertEqual(pd.Timestamp(t.entry_utc.iloc[0]).minute, 4)
        z = market([101]*6, zero=list(range(6)))
        self.assertEqual(threshold_trades(z)[0], [])
        _, daily, s, _ = account(z, [])
        self.assertEqual(s["total_net_pnl"], 0)
        self.assertEqual(s["active_time_fraction"], 0)
        self.assertIsNone(s["daily_sharpe"])
        self.assertEqual(daily.equity.iloc[0], 300)

    def test_open_mark_to_market_two_entry_fills_only(self):
        m = market([101, 101, 101, 103, 103, 103])
        trades, _ = threshold_trades(m)
        cost = {"spot_fee_bps": 10, "perp_fee_bps": 20, "slippage_bps": 0}
        t, _, s, _ = account(m, trades, cost=cost)
        row = t.iloc[0]
        self.assertEqual(row.status, "open")
        self.assertEqual(row.fill_count, 2)
        self.assertAlmostEqual(row.fees, .302)
        self.assertAlmostEqual(row.net_pnl, -2.302)
        self.assertAlmostEqual(s["total_net_pnl"], -2.302)
        self.assertAlmostEqual(row.terminal_liquidation_cost_estimate, .306)
        self.assertAlmostEqual(row.maximum_adverse_excursion_usdt, -2.302)

    def test_delay_and_changed_future_close_do_not_change_past_fill(self):
        m = market([101]*8+[99]*8)
        one, _ = threshold_trades(m, delay=1)
        six, _ = threshold_trades(m, delay=6)
        self.assertEqual(m.opens[six[0]["entry_row"]]-m.opens[one[0]["entry_row"]], 5*60*10**9)
        original = one[0]["entry_row"]
        m.frame.iloc[original, m.frame.columns.get_loc("perp_close")] = 500
        again, _ = threshold_trades(m)
        self.assertEqual(again[0]["entry_row"], original)

    def test_margin_does_not_count_unrealized_spot_as_collateral(self):
        m = market([101, 101, 101, 501, 501, 501], spot=[100, 100, 100, 500, 500, 500])
        trades, _ = threshold_trades(m)
        _, _, s, _ = account(m, trades)
        self.assertAlmostEqual(s["total_net_pnl"], 0)
        self.assertTrue(s["margin_proxy_breach"])
        self.assertGreater(s["additional_cash_required_proxy"], 200)

    def test_financing_and_full_calendar_returns(self):
        m = market([101]*8)
        trades, _ = threshold_trades(m)
        t, _, _, _ = account(m, trades, financing_apr=.1)
        self.assertAlmostEqual(t.financing_hurdle.iloc[0], 100*.1*6/(365*24*60))
        s = return_metrics([300, 303, 303, 300], 300, 4)
        self.assertEqual(s["total_net_pnl"], 0)
        self.assertEqual(s["simple_annual_return"], 0)
        expected = np.array([0, .01, 0, -3/303])
        self.assertAlmostEqual(s["daily_sharpe"], expected.mean()/expected.std(ddof=1)*np.sqrt(365))

    def test_reject_overlaps_and_out_of_training_inputs(self):
        m = market([101]*8)
        trades, _ = threshold_trades(m)
        with self.assertRaises(ValueError):
            account(m, trades*2)
        with self.assertRaises(ValueError):
            account(m, trades, events(["2024-01-01T00:00Z"], [.1], [100]))
        splits = {"train": ["2020-01-01T00:00Z", "2023-01-01T00:00Z"]}
        with self.assertRaises(ValueError):
            require_training_period("2020-01-01T00:00Z", "2024-01-01T00:00Z", splits)
        selected = training_partitions([{"month": "2022-12", "symbol": "BTCUSDT"},
                                        {"month": "2023-01", "symbol": "BTCUSDT"}],
                                       "2020-01-01T00:00Z", "2023-01-01T00:00Z")
        self.assertEqual(len(selected), 1)

    def test_funding_during_missing_paired_minute_is_still_booked(self):
        m = market([101]*6+[99]*4, missing=[3, 4])
        trades, _ = threshold_trades(m)
        e = events(["2020-01-01 00:04:00.004Z"], [.01], [100])
        t, _, s, _ = account(m, trades, e)
        self.assertEqual(t.funding_events.iloc[0], 1)
        self.assertAlmostEqual(s["funding_pnl"], 1)
        self.assertEqual(t.missing_paired_minutes_held.iloc[0], 2)

    def test_measured_signal_never_replaces_open_execution_price(self):
        m = market([101]*5+[99]*5, opens={2: (100, 102)})
        trades, _ = threshold_trades(m, measurement="OHLC4")
        t, _, _, _ = account(m, trades)
        self.assertAlmostEqual(t.entry_fill_basis.iloc[0], .02)
        self.assertAlmostEqual(t.perp_entry_price.iloc[0], 102)
        hourly, _ = threshold_trades(m, spacing=60)
        self.assertEqual(hourly, [])  # No UTC hourly decision in this short path.

    def test_carry_is_calendar_driven_and_can_remain_unresolved(self):
        m = market([99]*8)
        trades, opp = carry_trades(m, "continuous")
        self.assertEqual(len(trades), 1)
        self.assertEqual(opp["all"], 1)
        self.assertIsNone(trades[0]["exit_row"])
        self.assertEqual(trades[0]["entry_row"], 2)
        t, _, _, _ = account(m, trades)
        self.assertEqual(t.status.iloc[0], "open")

    def test_exit_signal_without_later_fill_remains_open(self):
        m = market([101, 101, 101, 101, 99])
        trades, _ = threshold_trades(m)
        self.assertIsNotNone(trades[0]["exit_signal_row"])
        self.assertIsNone(trades[0]["exit_row"])
        t, _, s, _ = account(m, trades)
        self.assertEqual(s["completed_trades"], 0)
        self.assertEqual(s["open_trades"], 1)
        self.assertAlmostEqual(t.net_pnl.iloc[0], 2)

    def test_cross_year_open_exposure_is_calendar_marked(self):
        m = market([101]*10+[99]*5)
        shifted = m.frame.copy()
        offset = pd.Timestamp("2020-12-31 23:55Z")-pd.Timestamp("2020-01-01T00:00Z")
        shifted.index += offset
        shifted["bar_open_time_utc"] += offset
        m = Market(shifted)
        trades, _ = threshold_trades(m)
        e = events(["2021-01-01T00:00:00.005Z"], [.01], [100])
        p, records = build_accounting(m, trades, e, "2020-12-31T00:00Z", "2021-01-02T00:00Z", notional=100)
        t, daily, s, years = evaluate(m, p, records, ZERO, "2020-12-31T00:00Z", "2021-01-02T00:00Z", capital=300)
        self.assertEqual(list(daily.index.year), [2020, 2021])
        self.assertEqual(t.status.iloc[0], "closed")
        self.assertAlmostEqual(s["total_net_pnl"], 3)
        self.assertAlmostEqual(sum(y["total_net_pnl"] for y in years), 3)
        self.assertAlmostEqual(years[0]["total_net_pnl"], 0)
        self.assertAlmostEqual(years[1]["funding_pnl"], 1)

    def test_million_dollar_financing_does_not_overflow_integer_nanoseconds(self):
        m = market([101]*1500)
        trades, _ = threshold_trades(m)
        p, records = build_accounting(m, trades, events(), "2020-01-01T00:00Z", "2020-01-03T00:00Z", notional=1_000_000)
        expected = 1_000_000 * 1498/(365*24*60)
        self.assertAlmostEqual(records[0]["financing_dollar_years"], expected)
        t, _, _, _ = evaluate(m, p, records, ZERO, "2020-01-01T00:00Z", "2020-01-03T00:00Z", financing_apr=.1)
        self.assertAlmostEqual(t.financing_hurdle.iloc[0], expected*.1)


if __name__ == "__main__":
    unittest.main()
