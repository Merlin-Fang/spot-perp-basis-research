"""Manually calculable regression cases for He v7 replication components."""

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from spot_perp_basis.artifacts import ROOT
from spot_perp_basis.data.market import Market
from spot_perp_basis.theory import ANNUAL_PERIODS, GAMMA, IOTA, KAPPA, cost_bound, entry_side, exit_crossed, theory_signal
from spot_perp_basis.data.windows import END, START, require_training_times
from spot_perp_basis.backtest.accounting import account_positions, selected_funding
from spot_perp_basis.backtest.execution import convergence_positions
from spot_perp_basis.evaluation.metrics import active_hour_metrics, summarize
from spot_perp_basis.data.features import hourly_theory_observations as hourly_observations
from spot_perp_basis.data.funding import prepare_effr, rates_asof


def market(n=360, start="2020-01-01", spot=100., perp=100.):
    opens = pd.date_range(start, periods=n, freq="min", tz="UTC")
    frame = pd.DataFrame({"bar_open_time_utc": opens}, index=opens+pd.Timedelta(minutes=1))
    for venue, value in (("spot", spot), ("perp", perp)):
        frame[f"{venue}_open"] = value
        frame[f"{venue}_close"] = value
        frame[f"{venue}_volume"] = 1.
        frame[f"{venue}_trades"] = 1
    return Market(frame)


def rate(value=0.):
    return pd.DataFrame({"observation_time_utc": [START], "available_at_utc": [START], "annual_rate": [value]})


def events(times=(), rates=(), prices=()):
    return pd.DataFrame({"settlement_time_utc": pd.to_datetime(list(times), utc=True, format="mixed"),
        "funding_rate": list(rates), "perp_price_proxy": list(prices), "perp_proxy_age_minutes": 0.})


def position(en=1, ex=121, side=-1, hedge=2., start="2020-01-01", number=1):
    time = pd.Timestamp(start, tz="UTC")
    return {"entry_row": en, "exit_row": ex, "perpetual_side": side,
        "hedge_lambda": hedge, "trade_number": number,
        "entry_signal_utc": str(time+pd.Timedelta(minutes=en-1)),
        "exit_signal_utc": None if ex is None else str(time+pd.Timedelta(minutes=ex-1))}


def account(m, positions, e=None, r=None, **kwargs):
    return account_positions(m, positions, events() if e is None else e, rate() if r is None else r,
        notional=100., capital=300., spot_fee_bps=kwargs.get("spot_fee_bps",0.),
        perp_fee_bps=kwargs.get("perp_fee_bps",0.))


def observations(rhos, fee=0.):
    dates = pd.date_range("2020-01-01 01:00", periods=len(rhos), freq="h", tz="UTC")
    return pd.DataFrame({"signal_available": True, "rho": rhos, "annual_rate": 0.,
                         "rate_available_at_utc": START, "lambda": 1.}, index=dates)


class SignalTests(unittest.TestCase):
    def test_equation17_not_raw_basis_or_phi_minus_r(self):
        rho, hedge = theory_signal(100., 101., .05)
        self.assertAlmostEqual(float(rho), 1095/101+.5475-.05)
        self.assertAlmostEqual(float(hedge), 1095/(1095+.5475-.05))
        rho_above, _ = theory_signal(100., 101., .2)
        self.assertAlmostEqual(float(rho_above), 1095/101-.5475-.2)
        at_parity, _ = theory_signal(100.,100.,0.)
        self.assertAlmostEqual(float(at_parity),.5475)

    def test_annualization_units_and_published_fee_bounds(self):
        self.assertEqual(ANNUAL_PERIODS,1095)
        self.assertAlmostEqual(GAMMA,.5475)
        self.assertAlmostEqual(IOTA,.1095)
        for s,f,expected in [(0,0,0),(2.25,.18,.53217),(4.5,.72,1.14318),(6.75,1.44,1.79361)]:
            self.assertAlmostEqual(cost_bound(s,f),expected)

    def test_strict_entry_and_exit_crossing_both_directions(self):
        for value in [-.5,.5,0.]:
            self.assertEqual(entry_side(value,.5),0)
        self.assertEqual(entry_side(.50001,.5),-1)
        self.assertEqual(entry_side(-.50001,.5),1)
        self.assertTrue(exit_crossed(-1,0.))
        self.assertTrue(exit_crossed(-1,-.1))
        self.assertFalse(exit_crossed(-1,.1))
        self.assertTrue(exit_crossed(1,0.))
        self.assertTrue(exit_crossed(1,.1))
        self.assertFalse(exit_crossed(1,-.1))

    def test_interval_equilibrium_rejected_and_missing_rate_not_zero(self):
        with self.assertRaisesRegex(ValueError,"interval"):
            theory_signal(100.,100.,IOTA)
        rho, hedge = theory_signal(100.,100.,np.nan)
        self.assertTrue(np.isnan(rho) and np.isnan(hedge))
        with self.assertRaises(ValueError):
            theory_signal(0.,100.,0.)

    def test_observed_hour_endpoint_and_no_full_hour_lookahead(self):
        m=market(180)
        m.frame.loc[m.frame.index == pd.Timestamp("2020-01-01 01:01",tz="UTC"),"perp_close"]=200.
        obs=hourly_observations(m,rate(),START,START+pd.Timedelta(hours=3))
        self.assertFalse(obs.signal_available.iloc[0])  # no preceding 2019 bar
        self.assertEqual(obs.perp_close.iloc[1],100.)
        self.assertAlmostEqual(obs.rho.iloc[1],.5475)

    def test_missing_and_inactive_hour_samples_not_filled(self):
        m=market(240)
        m.frame=m.frame.drop(pd.Timestamp("2020-01-01 01:00",tz="UTC"))
        m.frame.loc[pd.Timestamp("2020-01-01 02:00",tz="UTC"),"spot_trades"]=0
        m=Market(m.frame)
        obs=hourly_observations(m,rate(),START,START+pd.Timedelta(hours=4))
        self.assertTrue(obs.rho.iloc[:3].isna().all())
        self.assertTrue(obs.signal_available.iloc[3])

    def test_crossing_close_and_reverse_no_overlapping_exposure(self):
        m=market(360)
        positions=convergence_positions(m,observations([1.,2.,-.2,-.1,1.]),spot_fee_bps=0.,perp_fee_bps=0.)
        self.assertEqual(positions[0]["perpetual_side"],-1)
        self.assertEqual(m.opens[positions[0]["entry_row"]],pd.Timestamp("2020-01-01 01:01",tz="UTC").value)
        self.assertEqual(positions[0]["exit_row"],positions[1]["entry_row"])
        self.assertEqual(positions[1]["perpetual_side"],1)
        for a,b in zip(positions,positions[1:]):
            self.assertLessEqual(a["exit_row"],b["entry_row"])

    def test_no_pyramiding_and_fees_change_only_entry_bound(self):
        m=market(480)
        obs=observations([1.,2.,1.,.1,-.01,-1.])
        pos=convergence_positions(m,obs,spot_fee_bps=2.25,perp_fee_bps=.18)
        self.assertEqual(len(pos),2)
        self.assertEqual(pd.Timestamp(pos[0]["exit_signal_utc"]).hour,5)
        self.assertEqual(pd.Timestamp(pos[1]["entry_signal_utc"]).hour,6)

    def test_future_price_rate_mutation_cannot_change_past_entry(self):
        m=market(360,perp=101.)
        obs=hourly_observations(m,rate(),START,START+pd.Timedelta(hours=5))
        a=convergence_positions(m,obs,spot_fee_bps=0.,perp_fee_bps=0.)
        future=m.frame.index > pd.Timestamp("2020-01-01 02:00",tz="UTC")
        m.frame.loc[future,"perp_close"]=90.
        r=rate(); r.loc[1]=[pd.Timestamp("2020-01-01 02:30",tz="UTC")]*2+[.2]
        changed=hourly_observations(Market(m.frame),r,START,START+pd.Timedelta(hours=5))
        b=convergence_positions(Market(m.frame),changed,spot_fee_bps=0.,perp_fee_bps=0.)
        for key in ("entry_signal_utc","entry_signal_rho","entry_rate_annual","entry_row","perpetual_side"):
            self.assertEqual(a[0][key],b[0][key])


class FundingAndAccountingTests(unittest.TestCase):
    def test_unequal_hedge_leg_units_and_four_fee_charges(self):
        m=market()
        m.frame.iloc[121,m.frame.columns.get_loc("spot_open")]=102.
        m.frame.iloc[121,m.frame.columns.get_loc("perp_open")]=99.
        rec,path,_=account(m,[position()],spot_fee_bps=10.,perp_fee_bps=20.)
        x=rec.iloc[0]
        self.assertEqual(x.spot_quantity,2.)
        self.assertEqual(x.perp_quantity,1.)
        self.assertAlmostEqual(x.spot_pnl,4.)
        self.assertAlmostEqual(x.perp_pnl,1.)
        self.assertAlmostEqual(x.transaction_costs,.802)
        self.assertAlmostEqual(x.net_pnl,4.198)
        self.assertAlmostEqual(path.net_pnl.sum(),x.net_pnl)
        self.assertEqual(x.fill_count,4)

    def test_reverse_direction_leg_pnl_and_financing_credit(self):
        m=market()
        m.frame.iloc[121,m.frame.columns.get_loc("spot_open")]=102.
        m.frame.iloc[121,m.frame.columns.get_loc("perp_open")]=99.
        rec,_,_=account(m,[position(side=1)],r=rate(.1))
        self.assertAlmostEqual(rec.spot_pnl.iloc[0],-4.)
        self.assertAlmostEqual(rec.perp_pnl.iloc[0],-1.)
        self.assertAlmostEqual(rec.financing_cost.iloc[0],-200*.1*2/8760)

    def test_financing_dollars_elapsed_time_not_return_bps(self):
        m=market(); rec,_,_=account(m,[position()],r=rate(.1))
        self.assertAlmostEqual(rec.financing_cost.iloc[0],200*.1*2/8760)
        self.assertAlmostEqual(rec.net_pnl.iloc[0],-200*.1*2/8760)

    def test_funding_sign_and_exact_settlement_ownership(self):
        m=market()
        e=events(["2020-01-01 00:01:00","2020-01-01 00:01:00.001",
                  "2020-01-01 02:01:00","2020-01-01 02:01:00.001"],
                 [.9,.01,-.02,.03],[100.]*4)
        pos=[position(),position(en=121,ex=241,side=1,number=2)]
        rec,path,ledger=account(m,pos,e)
        self.assertAlmostEqual(rec.funding_pnl.iloc[0],-1.)
        self.assertAlmostEqual(rec.funding_pnl.iloc[1],-3.)
        self.assertEqual(len(ledger),3)
        self.assertFalse(ledger.settlement_time_utc.duplicated().any())
        self.assertEqual(ledger.trade_number.tolist(),[1,1,2])
        self.assertAlmostEqual(path.funding_pnl.sum(),-4.)
        self.assertEqual(ledger.cash_booked_at_utc.iloc[0],pd.Timestamp("2020-01-01 00:02",tz="UTC"))

    def test_no_future_funding_books_and_unknown_notional_fails(self):
        m=market()
        e=events(["2020-01-01 03:00"],[.1],[100.])
        rec,_,ledger=account(m,[position()],e)
        self.assertEqual(rec.funding_pnl.iloc[0],0.)
        self.assertTrue(ledger.empty)
        e=events(["2020-01-01 01:00"],[.1],[np.nan])
        with self.assertRaisesRegex(ValueError,"unknown"):
            account(m,[position()],e)

    def test_open_position_has_mark_and_two_fills_no_fake_exit(self):
        m=market()
        m.frame.iloc[-1,m.frame.columns.get_loc("perp_close")]=99.
        rec,path,_=account(m,[position(ex=None)],spot_fee_bps=10.,perp_fee_bps=20.)
        self.assertEqual(rec.status.iloc[0],"open_censored")
        self.assertIsNone(rec.exit_utc.iloc[0])
        self.assertEqual(rec.fill_count.iloc[0],2)
        self.assertEqual(rec.exit_cost.iloc[0],0.)
        self.assertAlmostEqual(path.net_pnl.sum(),.6)

    def test_duplicate_or_overlapping_or_second_open_rejected(self):
        m=market()
        for pos in ([position(),position(en=60,ex=180,number=2)],
                    [position(ex=None),position(en=121,ex=180,number=2)]):
            with self.assertRaisesRegex(ValueError,"Overlapping"):
                account(m,pos)

    def test_missing_held_minutes_reported_and_not_filled(self):
        m=market(); m=Market(m.frame.drop(m.frame.index[60:63]))
        # Address entries/exits by actual observed opens, not compressed minute numbers.
        p=position(ex=118)
        rec,_,_=account(m,[p])
        self.assertEqual(rec.missing_held_minutes.iloc[0],3)

    def test_entry_must_follow_completed_signal(self):
        p=position(); p["entry_signal_utc"]="2020-01-01 00:01:00+00:00"
        with self.assertRaisesRegex(ValueError,"later"):
            account(market(),[p])


class RatesMetricsAndBoundaryTests(unittest.TestCase):
    def test_effr_percent_units_lag_and_no_backward_imputation(self):
        data={"refRates":[{"effectiveDate":"2020-01-02","type":"EFFR","percentRate":1.55}]}
        rates=prepare_effr(data)
        self.assertAlmostEqual(rates.annual_rate.iloc[0],.0155)
        idx=pd.to_datetime(["2020-01-08 23:59","2020-01-09 00:00"],utc=True)
        values,available=rates_asof(idx,rates)
        self.assertTrue(np.isnan(values[0]))
        self.assertAlmostEqual(values[1],.0155)
        self.assertEqual(pd.Timestamp(available[1]),idx[1])

    def test_future_rate_observations_do_not_enter_current_signal(self):
        rates=rate(.01)
        rates.loc[1]=[pd.Timestamp("2020-01-02",tz="UTC"),pd.Timestamp("2020-01-09",tz="UTC"),.9]
        values,_=rates_asof(pd.to_datetime(["2020-01-08"],utc=True),rates)
        self.assertAlmostEqual(values[0],.01)

    def test_active_moments_distinct_from_calendar_and_missingness(self):
        x=active_hour_metrics([.01,-.005,0.,0.],[True,True,False,False],1.)
        self.assertAlmostEqual(x["paper_style_notional_annual_return"],.005)
        self.assertAlmostEqual(x["paper_style_notional_annual_volatility"],.015)
        self.assertAlmostEqual(x["paper_style_notional_sharpe"],1/3)
        bad=active_hour_metrics([.01,-.005],[True,True],1.,complete=False)
        self.assertTrue(np.isnan(bad["paper_style_notional_sharpe"]))

    def test_cross_year_calendar_conservation_and_initial_equity_drawdown(self):
        m=market(n=2880,start="2020-12-31")
        p=position(en=1,ex=1501,start="2020-12-31")
        e=events(["2020-12-31 12:00","2021-01-01 00:00"],[.01,-.02],[100.,100.])
        rec,path,_=account(m,[p],e,spot_fee_bps=10.,perp_fee_bps=20.)
        summary,daily,hourly=summarize(rec,path,m,start="2020-12-31T00:00:00Z",
                                     end="2021-01-02T00:00:00Z",capital=300.,notional=100.)
        self.assertAlmostEqual(daily.net_pnl.sum(),rec.net_pnl.sum())
        self.assertAlmostEqual(summary.net_pnl.sum(),rec.net_pnl.sum())
        self.assertEqual(summary.completed_trade_count.tolist(),[0,1])
        self.assertAlmostEqual(summary.funding_pnl.iloc[0],1.)
        self.assertAlmostEqual(summary.funding_pnl.iloc[1],-2.)
        self.assertLess(summary.observed_minute_max_drawdown.iloc[0],0.)
        self.assertAlmostEqual(daily.capital_return.iloc[0],daily.net_pnl.iloc[0]/300.)

    def test_heldout_observations_and_funding_rejected(self):
        for year in (2023,2024,2025):
            with self.assertRaisesRegex(ValueError,"2020"):
                require_training_times(pd.date_range(f"{year}-01-01",periods=2,tz="UTC"))
            with self.assertRaises(ValueError):
                prepare_effr({"refRates":[{"effectiveDate":f"{year}-01-01","type":"EFFR","percentRate":1.}]})

    def test_whole_missing_hour_keeps_exposure_and_invalidates_active_metrics(self):
        m=market(n=1440)
        m=Market(m.frame.drop(m.frame.index[60:120]))
        rec,path,_=account(m,[position(ex=1351)])
        summary,_,hourly=summarize(rec,path,m,start="2020-01-01T00:00:00Z",
                                  end="2020-01-02T00:00:00Z",capital=300.,notional=100.)
        self.assertTrue(hourly.loc[pd.Timestamp("2020-01-01 02:00",tz="UTC"),"active"])
        self.assertGreater(summary.missing_active_hour_endpoints.iloc[0],0)
        self.assertTrue(np.isnan(summary.paper_style_notional_sharpe.iloc[0]))

    def test_loader_selects_before_reads_and_predicates_boundary(self):
        from scripts.run_he_replication import load_inputs
        # Synthetic catalog includes forbidden months and another symbol; no
        # archived manifests or market files are needed for this IO-bound test.
        catalog={"partitions":[{"symbol":symbol,"month":str(month),
            "path":f"synthetic/{symbol}/{month}.parquet","sha256":"a"*64}
            for symbol in ("BTCUSDT","ETHUSDT")
            for month in pd.period_range("2020-01","2025-12",freq="M")]}
        reads=[]
        def read(path,**kwargs):
            reads.append((str(path),kwargs))
            if "funding_notional" in str(path):
                return events(["2020-01-08 08:00"],[.0001],[100.])
            month=Path(path).stem
            self.assertLess(month,"2023-01")
            self.assertIn(("decision_time_utc","<",END),kwargs["filters"])
            return market(n=1,start=month+"-15").frame
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);(root/"manifests").mkdir()
            (root/"manifests/phase2_7_history.json").write_text(json.dumps({"prior_results":{
                "results/phase2_5/BTCUSDT_funding_notional_proxies.parquet":"a"*64}}))
            with patch("scripts.run_he_replication.ROOT",root), \
                 patch("scripts.run_he_replication.file_sha256",return_value="a"*64), \
                 patch("scripts.run_he_replication.pd.read_parquet",side_effect=read):
                m,e=load_inputs("BTCUSDT",catalog,pd.Timestamp("2020-01-08",tz="UTC"),END,[])
        self.assertEqual(len(reads),37)
        self.assertEqual(len(m.frame),36)
        self.assertTrue((m.frame.index<END).all())


if __name__ == "__main__":
    unittest.main()
