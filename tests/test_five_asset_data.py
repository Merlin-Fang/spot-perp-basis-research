"""Synthetic data-integration regressions; no targets, strategy runs or model fits."""

import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from spot_perp_basis.data.market import PRICE_COLUMNS, hourly_panel, market_quality, minute_panel
from spot_perp_basis.data.windows import START, END, bounded
from spot_perp_basis.data.dataset import allowed_month, preserve, save_frame, study
from spot_perp_basis.data.funding import funding_history


def bars(start='2020-01-01T00:00:00Z',count=121,price=10.):
    index=pd.date_range(start,periods=count,freq='min',name='open_time_utc')
    frame=pd.DataFrame(index=index)
    for col in PRICE_COLUMNS: frame[col]=price if col in ['open','high','low','close'] else 1.
    frame['quote_volume']=price; frame['taker_buy_volume']=0.; frame['taker_buy_quote_volume']=0.
    frame['close_time']=(index.as_unit('ms').asi8+59999)
    for col in ['observed','valid']: frame[col]=True
    for col in ['bad_price','bad_activity','bad_close_time']: frame[col]=False
    return frame


def events(symbol='BTCUSDT',stamps=('2020-01-01T00:00:00Z',),rates=None):
    times=pd.to_datetime(stamps,utc=True,format='ISO8601')
    return pd.DataFrame({'symbol':symbol,'settlement_time_utc':times,
        'funding_rate':rates if rates is not None else [0.0001]*len(times),
        'funding_interval_hours':8.,'source_month':times.strftime('%Y-%m'),'source_sha256':'a'*64})


class FiveAssetTests(unittest.TestCase):
    def setUp(self): self.cfg=study()

    def funding(self,symbol='BTCUSDT',stamps=('2020-01-01T00:00:00Z',),rates=None):
        return funding_history(events(symbol,stamps,rates),symbol,self.cfg['assets'][symbol])

    def test_launch_structural_missing_inactive_and_active_are_distinct(self):
        s=bars('2020-02-10T07:59:00Z',4); p=s.copy()
        p.loc[p.index[1],['observed','valid']]=False
        s.loc[s.index[2],['volume','quote_volume','trades']]=0
        e=self.funding('BNBUSDT',('2020-02-10T08:00:00Z',))
        f=minute_panel(market_quality(s,None,self.cfg),market_quality(p,None,self.cfg),e,'BNBUSDT',self.cfg)
        self.assertEqual(f.pair_status.tolist(),['structural_prelisting','missing_or_invalid','inactive','active'])
        self.assertFalse(f.perp_missing.iloc[0]); self.assertTrue(f.perp_missing.iloc[1])
        self.assertTrue(f.spot_observed.iloc[2]); self.assertFalse(f.active_sample.iloc[2])

    def test_low_price_quote_validation_is_relative_and_non_destructive(self):
        s=bars(price=.001); s['quote_volume']=.00105
        f=market_quality(s,None,self.cfg)
        self.assertTrue(f.quote_price_flag.all()); self.assertTrue(f.valid.all())
        self.assertTrue(f.close.eq(.001).all())

    def test_funding_milliseconds_signs_and_no_rounding(self):
        e=self.funding(stamps=('2020-01-01T00:00:00.043Z','2020-01-01T08:00:00.001Z'),rates=[-.0002,0.])
        self.assertEqual(e.funding_rate.tolist(),[-.0002,0.])
        self.assertEqual(e.settlement_time_utc.iloc[0].microsecond,43000)
        self.assertEqual(e.available_at_utc.iloc[0],pd.Timestamp('2020-01-01T00:01:00.043Z'))

    def test_prelaunch_records_quarantined_for_features_and_accounting(self):
        e=self.funding('ADAUSDT',('2020-01-19T16:00:00Z','2020-01-31T08:00:00Z','2020-01-31T16:00:00Z'))
        self.assertEqual(e.quarantined.tolist(),[True,True,False])
        self.assertEqual(e.eligible_for_accounting.tolist(),[False,False,True])
        self.assertEqual(e.quarantine_reason.tolist(),['before_public_open','before_first_observed_bar',''])
        s=market_quality(bars('2020-01-31T08:01:00Z',3),None,self.cfg)
        f=minute_panel(s,s,e,'ADAUSDT',self.cfg)
        self.assertFalse(f.funding_known.any())

    def test_spot_perpetual_exact_minute_pairing_keeps_missing_grid(self):
        s=bars(count=3); p=s.copy(); p.loc[p.index[1],['valid','observed']]=False
        f=minute_panel(market_quality(s,None,self.cfg),market_quality(p,None,self.cfg),self.funding(),'BTCUSDT',self.cfg)
        self.assertEqual(len(f),3); self.assertEqual(f.paired_valid.tolist(),[True,False,True])
        self.assertTrue(pd.isna(f.basis_bps.iloc[1]))

    def test_completed_bar_and_execution_clock(self):
        s=market_quality(bars(count=3),None,self.cfg)
        f=minute_panel(s,s,self.funding(),'BTCUSDT',self.cfg)
        self.assertEqual(f.index[0],pd.Timestamp('2020-01-01T00:01:00Z'))
        self.assertEqual(f.earliest_execution_time_utc.iloc[0],pd.Timestamp('2020-01-01T00:02:00Z'))
        self.assertTrue((f.bar_available_at_utc==f.index).all())

    def test_causal_funding_join_respects_millisecond_plus_lag(self):
        e=self.funding(stamps=('2020-01-01T00:00:00.043Z',))
        s=market_quality(bars(count=3),None,self.cfg); f=minute_panel(s,s,e,'BTCUSDT',self.cfg)
        self.assertFalse(f.funding_known.iloc[0]); self.assertTrue(f.funding_known.iloc[1])
        self.assertLessEqual(f.last_funding_available_at_utc.iloc[1],f.index[1])

    def test_hourly_uses_completed_hour_only(self):
        s=bars(); s['quote_volume']=np.arange(len(s))+10.
        q=market_quality(s,None,self.cfg); f=minute_panel(q,q,self.funding(),'BTCUSDT',self.cfg)
        h=hourly_panel(f,'BTCUSDT',self.cfg)
        self.assertEqual(h.index[0],pd.Timestamp('2020-01-01T01:00:00Z'))
        self.assertEqual(h.spot_hour_quote_volume.iloc[0],s.quote_volume.iloc[:60].sum())
        changed=s.copy(); changed.loc[changed.index>=pd.Timestamp('2020-01-01T01:00:00Z'),'quote_volume']=1e9
        c=market_quality(changed,None,self.cfg)
        other=hourly_panel(minute_panel(c,c,self.funding(),'BTCUSDT',self.cfg),'BTCUSDT',self.cfg)
        self.assertEqual(h.spot_hour_quote_volume.iloc[0],other.spot_hour_quote_volume.iloc[0])

    def test_incomplete_hour_is_not_filled(self):
        s=bars(); s.loc[s.index[10],['observed','valid']]=False
        q=market_quality(s,None,self.cfg)
        h=hourly_panel(minute_panel(q,q,self.funding(),'BTCUSDT',self.cfg),'BTCUSDT',self.cfg)
        self.assertFalse(h.spot_hour_complete.iloc[0]); self.assertTrue(pd.isna(h.spot_hour_volume.iloc[0]))
        self.assertEqual(h.spot_hour_observed_minutes.iloc[0],59)

    def test_cross_month_minute_continuity_and_return_flag(self):
        a=market_quality(bars('2020-02-29T23:59:00Z',1),None,self.cfg)
        left=minute_panel(a,a,self.funding(),'BTCUSDT',self.cfg)
        b=market_quality(bars('2020-03-01T00:00:00Z',2,price=20),a.iloc[-1],self.cfg)
        right=minute_panel(b,b,self.funding(),'BTCUSDT',self.cfg,left.iloc[-1])
        self.assertTrue(right.contiguous_previous_minute.iloc[0]); self.assertTrue(b.large_return_flag.iloc[0])

    def test_cross_month_funding_cadence_and_interval_change(self):
        e=events(stamps=('2020-01-31T16:00:00Z','2020-02-01T00:00:00.001Z','2020-02-01T04:00:00Z'))
        e.loc[2,'funding_interval_hours']=4.
        f=funding_history(e,'BTCUSDT',self.cfg['assets']['BTCUSDT'])
        self.assertTrue(f.cross_month_event.iloc[1]); self.assertTrue(f.interval_change_flag.iloc[2])
        self.assertFalse(f.interval_mismatch_flag.any())

    def test_funding_long_gap_flag_does_not_create_events(self):
        e=self.funding(stamps=('2020-01-01T00:00:00Z','2020-01-02T00:00:00Z'))
        self.assertEqual(len(e),2); self.assertTrue(e.cadence_gap_flag.iloc[1])

    def test_duplicate_funding_rejected(self):
        with self.assertRaisesRegex(ValueError,'Duplicate'):
            self.funding(stamps=('2020-01-01T00:00:00Z','2020-01-01T00:00:00Z'))

    def test_strict_completion_boundary_and_clock_mask(self):
        q=market_quality(bars('2022-12-31T23:58:00Z',2),None,self.cfg)
        f=minute_panel(q,q,self.funding(),'BTCUSDT',self.cfg)
        self.assertEqual(len(f),1); self.assertTrue((f.index<END).all())
        self.assertTrue(pd.isna(f.earliest_execution_time_utc.iloc[0]))
        with self.assertRaises(ValueError): bounded(pd.DataFrame(index=pd.date_range(END,periods=1)))

    def test_forbidden_source_months_are_rejected_before_io(self):
        for month in ['2019-12','2023-01','2025-12']:
            with self.assertRaises(ValueError): allowed_month('BTCUSDT',month,self.cfg)
        with self.assertRaises(ValueError): allowed_month('DOGEUSDT','2020-06',self.cfg)

    def test_multi_asset_isolation(self):
        with self.assertRaisesRegex(ValueError,'Cross-symbol'):
            funding_history(events('ETHUSDT'),'BTCUSDT',self.cfg['assets']['BTCUSDT'])
        s=market_quality(bars('2020-07-10T09:00:00Z'),None,self.cfg)
        for symbol in self.cfg['assets']:
            e=self.funding(symbol,('2020-07-10T08:00:00Z',))
            self.assertTrue(minute_panel(s,s,e,symbol,self.cfg).symbol.eq(symbol).all())

    def test_derived_outputs_are_idempotent_and_fail_on_corruption_or_changed_provenance(self):
        with tempfile.TemporaryDirectory() as folder,patch('spot_perp_basis.data.dataset.ROOT',Path(folder)):
            path=Path(folder)/'out.parquet'; frame=pd.DataFrame({'a':[1,2]})
            first=save_frame(frame,path,{'source':'frozen'})
            self.assertEqual(save_frame(frame,path,{'source':'frozen'}),first)
            with self.assertRaisesRegex(ValueError,'provenance'): save_frame(frame,path,{'source':'changed'})
            path.write_bytes(b'bad')
            with self.assertRaises(ValueError): save_frame(frame,path,{'source':'frozen'})

    def test_existing_sources_data_and_experiments_preserved(self):
        p=preserve(); self.assertGreaterEqual(p['content_hashes_unchanged'],657)
        self.assertGreaterEqual(p['data_result_file_states_unchanged'],1691)


if __name__=='__main__': unittest.main()
