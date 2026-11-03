"""Reader guards are tested without opening any market file."""

import unittest
from unittest.mock import patch

import pandas as pd

from spot_perp_basis.data.windows import START, END
from spot_perp_basis.data.dataset import funding_events, hourly_observations, minute_partitions


class InterfaceTests(unittest.TestCase):
    def test_holdout_requests_rejected_before_manifest_or_parquet_io(self):
        with patch('pandas.read_parquet') as read:
            for loader in [funding_events,hourly_observations]:
                with self.assertRaises(ValueError): loader('BTCUSDT',START,pd.Timestamp('2024-01-01',tz='UTC'))
            with self.assertRaises(ValueError): list(minute_partitions('BTCUSDT',END,END+pd.Timedelta(days=1)))
            read.assert_not_called()

    def test_hourly_predicates_include_symbol_and_exclusive_end(self):
        f=pd.DataFrame({'symbol':['BNBUSDT']},index=pd.date_range('2020-02-10T09:00Z',periods=1))
        m={'hourly_panel':{'path':'fake'}}
        with patch('spot_perp_basis.data.dataset._window',return_value=({},m,START,END)),patch('spot_perp_basis.data.dataset._verified',return_value='fake'),patch('pandas.read_parquet',return_value=f) as read:
            hourly_observations('BNBUSDT')
            self.assertIn(('symbol','==','BNBUSDT'),read.call_args.kwargs['filters'])
            self.assertIn(('decision_time_utc','<',END),read.call_args.kwargs['filters'])

    def test_funding_default_filters_quarantine_before_materialization(self):
        f=pd.DataFrame({'symbol':['ADAUSDT']})
        m={'artifacts':[{'path':'data/processed/five_asset_v1/funding/ADAUSDT.parquet'}]}
        with patch('spot_perp_basis.data.dataset._window',return_value=({},m,START,END)),patch('spot_perp_basis.data.dataset._verified',return_value='fake'),patch('pandas.read_parquet',return_value=f) as read:
            funding_events('ADAUSDT')
            self.assertIn(('eligible_for_causal_use','==',True),read.call_args.kwargs['filters'])
            funding_events('ADAUSDT',purpose='accounting')
            self.assertIn(('eligible_for_accounting','==',True),read.call_args.kwargs['filters'])


if __name__=='__main__': unittest.main()
