"""Synthetic structural integration, with pre-2023-only numerical exercises."""

from hashlib import sha256
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from spot_perp_basis.data.final_sources import Source, ASSETS, MONTHS, digest, json_bytes
from spot_perp_basis.data.final_inputs import structural_only, causal_effr
from spot_perp_basis.data.forecast_inputs import *
from spot_perp_basis.data.input_preflight import inspect_inputs, contract, HISTORY_MONTHS, FORMAT, _digest_set
from spot_perp_basis.data.funding import prepare_funding, join_funding
from spot_perp_basis.data.labels import labels
from spot_perp_basis.data.features import FEATURES
from spot_perp_basis.data.windows import origin_mask
from spot_perp_basis.training.annual import causal_features, fit_development_year, raw_probabilities
from spot_perp_basis.training.fit import hourly_risk_set
from spot_perp_basis.training.checkpoints import BASE_INPUT, OPTIONAL_INPUT
from test_final_data import put, archive


ROOT = Path(__file__).resolve().parents[1]


def write(root, name, value):
    p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(json_bytes(value))
    return {"path":name,"sha256":digest(p)}


def save(root, name, frame, **metadata):
    p=root/name;p.parent.mkdir(parents=True,exist_ok=True);frame.to_parquet(p,index=True)
    return {"path":name,"sha256":digest(p),**metadata}


def raw_frame(clock, symbol="BTCUSDT", interruption=False):
    clock=pd.DatetimeIndex(clock).as_unit("ns").rename("decision_time_utc")
    f=pd.DataFrame(index=clock);f["symbol"]=symbol;f["bar_open_time_utc"]=clock-pd.Timedelta(minutes=1)
    for v in ("spot","perp"):
        f[v+"_close"]=100. if v=="spot" else 100.1
        f[v+"_volume"]=1.;f[v+"_trades"]=3;f[v+"_observed"]=True;f[v+"_valid"]=True
    if interruption:
        partial=clock==pd.Timestamp("2023-03-24 12:40",tz="UTC")
        gap=(clock>=pd.Timestamp("2023-03-24 12:41",tz="UTC"))&(clock<=pd.Timestamp("2023-03-24 14:00",tz="UTC"))
        f.loc[partial,"spot_valid"]=False;f.loc[gap,["spot_valid","spot_observed"]]=False
        f.loc[gap,["spot_close","spot_volume","spot_trades"]]=np.nan
    masks=observation_masks(f,symbol);f["pair_valid"]=masks.pair_valid;f["pair_active"]=masks.pair_active
    return f


def fixture(root):
    """Full hourly clocks, small synthetic archives and explicit input pins."""
    root=Path(root);data=root/"final";history=root/"history";rows=[];hist_sources=[];by_source={}
    for symbol in ASSETS:
        for month in MONTHS:
            for source in ("spot","usdm_perpetual","funding"):
                s=Source(symbol,month,source);r=put(data,s,unit="us" if source=="spot" and month>="2025-01" else "ms")
                rows.append({"key":s.key,"receipt":r});by_source[(symbol,month,source)]=r
        for month in HISTORY_MONTHS:
            for source in ("spot","usdm_perpetual","funding"):
                kind="fundingRate" if source=="funding" else "1m";name=f"{symbol}-{kind}-{month}.zip"
                if month=="2022-12":body=(data/by_source[(symbol,month,source)]["path"]).read_bytes()
                else:body=archive(types.SimpleNamespace(symbol=symbol,month=month,source=source,filename=name))
                path=f"raw/{source}/{symbol}/{month}/{name}";p=history/path;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(body)
                check=(sha256(body).hexdigest()+"  "+name+"\n").encode();p.with_name(name+".CHECKSUM").write_bytes(check)
                r={"symbol":symbol,"month":month,"kind":"funding" if source=="funding" else "klines","market":"spot" if source=="spot" else "usdm_perpetual","path":path,"sha256":digest(p),"checksum_path":path+".CHECKSUM","checksum_sha256":sha256(check).hexdigest(),"url":Source(symbol,"2022-12",source).url.replace("2022-12",month)}
                hist_sources.append(r);by_source[(symbol,month,source)]=r
    acquisition=write(history,"sources.json",{"schema_version":"five-asset-source-v1","complete":True,"identity":{"config_sha256":"0"*64},"files":hist_sources})
    days=pd.date_range("2020-01-02","2022-12-30",freq="B",tz="UTC")
    effr={"refRates":[{"type":"EFFR","effectiveDate":str(t.date()),"percentRate":4.} for t in days]}
    er=write(history,"effr.json",effr)
    erp=write(history,"effr_receipt.json",{"sources":{"effr":{"sha256":er["sha256"],"url":"https://markets.newyorkfed.org/api/rates/unsecured/effr/search.json?startDate=2020-01-01&endDate=2022-12-31"}}})
    effective=pd.date_range("2022-12-01","2025-12-31",freq="B",tz="UTC").as_unit("ns")
    rates=pd.DataFrame({"observation_time_utc":effective,"available_at_utc":effective+pd.Timedelta(days=7),"annual_rate":.04,"source":"NYFed_EFFR_current_vintage_percent_ACT365_proxy"})
    fa=[save(data,"canonical/effr.parquet",rates)];ha=[]
    for symbol in ASSETS:
        times=pd.date_range("2020-12-01","2026-01-01",freq="8h",inclusive="left",tz="UTC").as_unit("ns")
        events=prepare_funding(pd.DataFrame({"symbol":symbol,"settlement_time_utc":times,"funding_rate":np.where(np.arange(len(times))%2,.0001,-.0001),"funding_interval_hours":8.,"source_month":times.strftime("%Y-%m")}))
        events["source_sha256"]=[by_source[(symbol,m,"funding")]["sha256"] for m in events.source_month]
        events["quarantined"]=False
        old=events.loc[events.settlement_time_utc<START].reset_index(drop=True)
        new=events.loc[events.settlement_time_utc>=pd.Timestamp("2022-12-01",tz="UTC")].reset_index(drop=True)
        ha.append(save(history,f"data/processed/five_asset_v1/funding/{symbol}.parquet",old))
        fa.append(save(data,f"canonical/funding/{symbol}.parquet",new))
        clock=pd.date_range(HISTORY_START,END,freq="h",inclusive="left").as_unit("ns")
        full=join_funding(raw_frame(clock,symbol,interruption=True),events)
        for month in HISTORY_MONTHS:
            begin=pd.Timestamp(month+"-01",tz="UTC");end=begin+pd.offsets.MonthBegin()
            f=full.loc[(full.bar_open_time_utc>=begin)&(full.bar_open_time_utc<end)&(full.index<START)].copy()
            f=f.rename(columns={"pair_valid":"paired_valid","pair_active":"active_sample"})
            ha.append(save(history,f"data/processed/five_asset_v1/hourly/{symbol}/{month}.parquet",f,
                provenance={"symbol":symbol,"source_month":month,"spot_sha256":by_source[(symbol,month,"spot")]["sha256"],"perp_sha256":by_source[(symbol,month,"usdm_perpetual")]["sha256"]}))
        for year in (2023,2024,2025):
            f=full.loc[full.index.year==year].copy();f=f.join(causal_effr(f.index,rates))
            fa.append(save(data,f"canonical/hourly/{symbol}/{year}.parquet",f))
        march=raw_frame(pd.date_range("2023-03-24 12:40","2023-03-24 14:00",freq="min",tz="UTC"),symbol,interruption=True)
        fa.append(save(data,f"canonical/minute/{symbol}/2023-03.parquet",march))
    hd=write(history,"dataset.json",{"schema_version":"five-asset-observations-v1","identity":{"config_sha256":"0"*64,"acquisition_sha256":acquisition["sha256"]},"artifacts":ha})
    inventory={"complete":True,"files":rows,"effr":{"receipt":{"sha256":"a"*64}}}
    ir=write(data,"inventory.json",inventory)
    br=write(data,"build_receipt.json",{"schema_version":"final-inputs-v1","inventory_sha256":ir["sha256"],"artifacts":fa})
    qr=write(data,"qa_receipt.json",{"status":"FINAL_DATA_STRUCTURAL_QA_READY","build_sha256":br["sha256"]})
    public=write(root,"public.json",{"data_schema":"final-inputs-v1","status":"FINAL_DATA_STRUCTURAL_QA_READY","data_receipts_sha256":{"inventory.json":ir["sha256"],"build_receipt.json":br["sha256"],"qa_receipt.json":qr["sha256"]},"implementation_sha256":{},"canonical_source_set_sha256":_digest_set([[r['key'],r['receipt']['sha256']] for r in rows]+[['NYFED/EFFR/2022-12_2025-12','a'*64]]),"canonical_table_set_sha256":_digest_set([[a['path'],a['sha256']] for a in fa]),"timestamp_conventions":{"spot_2025":"us","internal":"UTC_ns"}})
    catalog={"schema_version":FORMAT,"final_verification":public,"history_dataset":hd,"history_sources":acquisition,"historical_effr":er,"historical_effr_receipt":erp}
    write(root,"catalog.json",catalog)
    return data,history,catalog


class FinalInputIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory();cls.root=Path(cls.temp.name)
        cls.data,cls.history,cls.catalog=fixture(cls.root)

    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()

    def development(self, frequency="h"):
        clock=pd.date_range("2021-03-23","2021-03-27",freq=frequency,inclusive="left",tz="UTC").as_unit("ns")
        f=raw_frame(clock)
        ev=prepare_funding(pd.DataFrame({"settlement_time_utc":pd.date_range("2021-03-22","2021-03-27",freq="8h",tz="UTC").as_unit("ns"),"funding_rate":-.001,"funding_interval_hours":8.}))
        return join_funding(f,ev)

    def test_all_annual_schedules_and_purge_counts(self):
        c=contract();self.assertEqual(c['feature_order'],list(FEATURES['minimal']))
        for p in c['plans']:
            self.assertEqual(p['schedule_counts']['fit']['clock']-p['schedule_counts']['fit']['origin_bound_permitted'],16)
            self.assertEqual(p['schedule_counts']['calibration']['clock']-p['schedule_counts']['calibration']['origin_bound_permitted'],16)
            self.assertEqual(p['schedule_counts']['evaluation']['clock'],8784 if p['year']==2024 else 8760)
            self.assertEqual(p['schedule_counts']['evaluation']['clock'],p['schedule_counts']['evaluation']['origin_bound_permitted'])
        self.assertEqual(c['plans'][0]['warmup'][0],'2020-12-31T00:00:00+00:00')

    def test_input_schema_matches_real_annual_interface(self):
        c,m=adapt_hourly(self.development(),'BTCUSDT','zero_rate')
        self.assertTrue(BASE_INPUT<=set(c)<=BASE_INPUT|OPTIONAL_INPUT)
        self.assertEqual(c.index.unit,'ns');self.assertNotIn('rho',c)
        f=causal_features(c,'BTCUSDT','zero_rate');self.assertEqual(f[list(FEATURES['minimal'])].shape,(96,6))

    def test_all_four_assets_two_controls_and_three_years(self):
        report=inspect_inputs(self.data,self.history,self.catalog,self.root)
        self.assertEqual(report['available_cells'],24);self.assertEqual(len(report['cells']),24)
        self.assertEqual(report['historical_raw_archives_verified'],300)
        self.assertTrue(all(c['statistical_support']=='not_evaluated' for c in report['cells']))

    def test_missing_invalid_inactive_and_active_are_distinct(self):
        f=self.development().iloc[:4].copy()
        f.iloc[0,f.columns.get_loc('spot_observed')]=False;f.iloc[0,f.columns.get_loc('spot_valid')]=False
        f.iloc[1,f.columns.get_loc('spot_valid')]=False;f.iloc[2,f.columns.get_loc('spot_volume')]=0
        f=f.drop(columns=['pair_valid','pair_active'])
        c,m=adapt_hourly(f,'BTCUSDT','zero_rate')
        self.assertEqual(m.spot_state.tolist(),['missing','invalid','inactive','active'])
        self.assertEqual(c.theory_available.tolist(),[False,False,False,True])
        self.assertTrue(c.spot_close.iloc[:2].isna().all());self.assertEqual(c.spot_close.iloc[2],100.)
        self.assertEqual(len(c),4)

    def test_exact_completed_minute_and_symbol_identity(self):
        f=self.development();bad=f.copy();bad['bar_open_time_utc']=bad.index
        with self.assertRaises(ValueError):adapt_hourly(bad,'BTCUSDT','zero_rate')
        with self.assertRaises(ValueError):adapt_hourly(f,'ADAUSDT','zero_rate')

    def test_duplicate_and_wrong_timestamp_units_rejected(self):
        f=self.development()
        with self.assertRaises(ValueError):adapt_hourly(pd.concat([f.iloc[:1],f]),'BTCUSDT','zero_rate')
        f.index=f.index.as_unit('us')
        with self.assertRaises(ValueError):adapt_hourly(f,'BTCUSDT','zero_rate')

    def test_month_midnight_and_terminal_handoffs(self):
        report=inspect_inputs(self.data,self.history,self.catalog,self.root)
        self.assertEqual(len(report['boundaries']),12)
        self.assertTrue(all(pd.Timestamp(x['bar_open'])+pd.Timedelta(minutes=1)==pd.Timestamp(x['origin']) for x in report['boundaries']))
        self.assertEqual(report['annual_plans'][-1]['evaluation'][-1],'2026-01-01T00:00:00+00:00')
        self.assertFalse(list(self.data.rglob('*2026*')))

    def test_strict_origin_lower_bounds_and_publication_calendar(self):
        for p in contract()['plans']:
            for name in ('fit','calibration'):
                begin,end=map(pd.Timestamp,p[name])
                clock=pd.DatetimeIndex([begin-pd.Timedelta(hours=1),begin,end-pd.Timedelta(hours=17),end-pd.Timedelta(hours=16),end])
                self.assertEqual(origin_mask(clock,begin,end,purge=True).tolist(),[False,True,True,False,False])
                self.assertTrue((clock[origin_mask(clock,begin,end,purge=True)]+pd.Timedelta(hours=16)<end).all())
        self.assertEqual(annual_plan(2025)['evaluation_origin_purge_hours'],0)

    def test_2025_microsecond_archive_normalizes_before_adapter(self):
        from spot_perp_basis.data.ingest import parse_klines
        with tempfile.TemporaryDirectory() as d:
            parsed=[]
            for month,unit in [('2024-12','ms'),('2025-01','us')]:
                spec=Source('BTCUSDT',month,'spot');r=put(d,spec,unit=unit)
                f,a=parse_klines(Path(d)/spec.relative,spec.symbol,spec.month,spec.source,r['sha256'])
                self.assertEqual(a['timestamp_unit'],unit);self.assertEqual(f.index.unit,'ns');parsed.append(f)
            self.assertEqual(parsed[0].index[-1]+pd.Timedelta(minutes=1),parsed[1].index[0])
            origin=parsed[0].index[-1]+pd.Timedelta(minutes=1)
            self.assertEqual(origin,START+pd.DateOffset(years=2))
            self.assertEqual(origin-pd.Timedelta(minutes=1),pd.Timestamp('2024-12-31 23:59',tz='UTC'))

    def test_conflicting_development_final_event_handoff_fails(self):
        from spot_perp_basis.data.input_preflight import _merge_events
        old=pd.read_parquet(self.history/'data/processed/five_asset_v1/funding/BTCUSDT.parquet')
        new=pd.read_parquet(self.data/'canonical/funding/BTCUSDT.parquet')
        new.loc[0,'funding_rate']+=.01
        with self.assertRaises(AssertionError):_merge_events(old,new,'BTCUSDT')

    def test_cached_archive_alias_uses_official_checksum_filename(self):
        from spot_perp_basis.data.input_preflight import _history_source
        source=json.loads((self.history/'sources.json').read_text())['files'][0]
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);p=root/'raw/cached_alias.zip';p.parent.mkdir();p.write_bytes((self.history/source['path']).read_bytes())
            c=root/'raw/check.CHECKSUM';c.write_bytes((self.history/source['checksum_path']).read_bytes())
            row={**source,'path':'raw/cached_alias.zip','checksum_path':'raw/check.CHECKSUM'}
            _history_source(root,row)

    def test_missing_final_annual_input_keeps_all_cells(self):
        path=self.data/'canonical/hourly/ADAUSDT/2024.parquet';body=path.read_bytes();path.unlink()
        try:r=inspect_inputs(self.data,self.history,self.catalog,self.root)
        finally:path.write_bytes(body)
        self.assertEqual(len(r['cells']),24)
        own=[c for c in r['cells'] if c['asset']=='ADAUSDT']
        self.assertTrue(all(c['input_status']=='unavailable' for c in own))

    def test_missing_history_manifest_cli_preserves_unavailable_matrix(self):
        from scripts.preflight_final_integration import main
        with tempfile.TemporaryDirectory() as td:
            output=Path(td)/'metadata.json'
            status=main(['--data-root',str(self.data),'--history-root',td,'--input-catalog',str(self.root/'catalog.json'),'--public-root',str(self.root),'--output',str(output)])
            r=json.loads(output.read_text());self.assertEqual(status,2);self.assertEqual(len(r['cells']),24)
            self.assertTrue(all(c['input_status']=='unavailable' for c in r['cells']))

    def test_empty_asset_input_has_explicit_absent_clock(self):
        from spot_perp_basis.data.input_preflight import _stage
        empty=pd.DataFrame(columns=RAW+FUNDING+['pair_valid','pair_active'],index=pd.DatetimeIndex([],tz='UTC').as_unit('ns'))
        r=_stage(empty,pd.Timestamp('2024-01-01',tz='UTC'),pd.Timestamp('2025-01-01',tz='UTC'))
        self.assertFalse(r['available']);self.assertEqual(r['present_clock'],0);self.assertEqual(r['absent_clock_rows'],8784)

    def test_wrong_final_schema_and_wrong_effr_instrument(self):
        self.assertRaises(ValueError,reference_history,{'refRates':[{'type':'SOFR','effectiveDate':'2021-01-01','percentRate':1.}]})
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);public=json.loads((self.root/'public.json').read_text());public['data_schema']='unverified'
            ref=write(root,'public.json',public);catalog={**self.catalog,'final_verification':ref}
            with self.assertRaisesRegex(ValueError,'verified final data'):inspect_inputs(self.data,self.history,catalog,root)

    def test_real_feature_entrypoint_still_rejects_final_numerical_access(self):
        f=pd.read_parquet(self.data/'canonical/hourly/BTCUSDT/2023.parquet').iloc[:2]
        context,_=adapt_hourly(f,'BTCUSDT','zero_rate')
        with self.assertRaisesRegex(ValueError,'2020'):causal_features(context,'BTCUSDT','zero_rate')
        with structural_only(),self.assertRaises(PermissionError):causal_features(context,'BTCUSDT','zero_rate')

    def test_subsecond_publication_age_and_signed_rate(self):
        f=self.development().iloc[:1].copy();t=f.index[0];settle=t-pd.Timedelta(minutes=1)+pd.Timedelta(milliseconds=15)
        f['last_funding_settlement_utc']=settle;f['last_funding_available_at_utc']=settle+pd.Timedelta(minutes=1)
        with self.assertRaisesRegex(ValueError,'[Ff]unding'):adapt_hourly(f,'BTCUSDT','zero_rate')
        f['last_funding_settlement_utc']=t-pd.Timedelta(hours=12);f['last_funding_available_at_utc']=f.last_funding_settlement_utc+pd.Timedelta(minutes=1)
        c,_=adapt_hourly(f,'BTCUSDT','zero_rate');self.assertTrue(c.funding_known.iloc[0])
        f['last_funding_settlement_utc']-=pd.Timedelta(milliseconds=1)
        with self.assertRaises(ValueError):adapt_hourly(f,'BTCUSDT','zero_rate')
        self.assertLess(f.last_settled_funding_rate.iloc[0],0)

    def test_historical_effr_exact_lag(self):
        f=self.development();raw={'refRates':[{'type':'EFFR','effectiveDate':'2021-03-15','percentRate':2.},{'type':'EFFR','effectiveDate':'2021-03-17','percentRate':4.}]}
        c,_=adapt_hourly(f,'BTCUSDT','nyfed_effr_lag7d',historical_rates=reference_history(raw))
        self.assertEqual(c.annual_rate.iloc[0],.02)
        self.assertEqual(c.loc[pd.Timestamp('2021-03-24',tz='UTC'),'annual_rate'],.04)
        self.assertTrue((c.rate_available_at_utc<=c.index).all())

    def test_final_reference_clock_rejects_future_rate(self):
        f=pd.read_parquet(self.data/'canonical/hourly/BTCUSDT/2025.parquet').iloc[:2].copy()
        f['effr_available_at_utc']=f.index+pd.Timedelta(hours=1)
        with self.assertRaises(ValueError):adapt_hourly(f,'BTCUSDT','nyfed_effr_lag7d')

    def test_unknown_flag_cannot_hide_future_funding_publication(self):
        f=self.development().iloc[:1].copy();f['funding_known']=False
        f['last_funding_available_at_utc']=f.index+pd.Timedelta(milliseconds=15)
        with self.assertRaisesRegex(ValueError,'publication'):adapt_hourly(f,'BTCUSDT','zero_rate')

    def test_synthetic_march_interruption_preserved_and_censored(self):
        f=self.development();gap=f.index.isin(pd.DatetimeIndex(['2021-03-24 13:00Z','2021-03-24 14:00Z']))
        f.loc[gap,['spot_valid','spot_observed']]=False;f.loc[gap,['spot_close','spot_volume','spot_trades']]=np.nan
        f=f.drop(columns=['pair_valid','pair_active']);f.loc[f.index>=pd.Timestamp('2021-03-24 15:00',tz='UTC'),'perp_close']=99.9
        c,_=adapt_hourly(f,'BTCUSDT','zero_rate');context=causal_features(c,'BTCUSDT','zero_rate')
        self.assertEqual(context.loc[pd.Timestamp('2021-03-24 15:00',tz='UTC'),'minimal_eligible36'],False)
        label=labels(context,pd.Timestamp('2021-03-24 11:00',tz='UTC'),pd.Timestamp('2021-03-27',tz='UTC'))
        r=label.iloc[0];self.assertFalse(r.event_observed);self.assertEqual(r.followup_hours,1)
        self.assertEqual(r.record_available_at_utc,pd.Timestamp('2021-03-24 13:00',tz='UTC'))
        minute=raw_frame(pd.date_range('2023-03-24 12:40','2023-03-24 14:00',freq='min',tz='UTC'),interruption=True)
        states=observation_masks(minute,'BTCUSDT').spot_state
        self.assertEqual(states.eq('missing').sum(),80);self.assertEqual(states.eq('invalid').sum(),1)

    def test_causal_feature_history_ignores_future_mutation(self):
        f=self.development();a,_=adapt_hourly(f,'BTCUSDT','zero_rate');cut=f.index[55]
        altered=f.copy();altered.loc[altered.index>cut,'perp_close']=90.
        b,_=adapt_hourly(altered,'BTCUSDT','zero_rate')
        x=causal_features(a,'BTCUSDT','zero_rate');y=causal_features(b,'BTCUSDT','zero_rate')
        pd.testing.assert_frame_equal(x.loc[:cut,list(FEATURES['minimal'])],y.loc[:cut,list(FEATURES['minimal'])])

    def test_issuance_inputs_do_not_depend_on_future_labels(self):
        f=self.development();f['event_by_8h']=0.;a,_=adapt_hourly(f,'BTCUSDT','zero_rate')
        f['event_by_8h']=np.nan;b,_=adapt_hourly(f,'BTCUSDT','zero_rate');pd.testing.assert_frame_equal(a,b)
        self.assertNotIn('event_by_8h',a)

    def test_matched_raw_feature_and_hourly_trial_contract_without_fitting(self):
        c,_=adapt_hourly(self.development(),'BTCUSDT','zero_rate');f=causal_features(c,'BTCUSDT','zero_rate')
        end=pd.Timestamp('2021-03-27',tz='UTC');label=labels(f,f.index[0],end)
        r=hourly_risk_set(f,label,'minimal',end)
        self.assertTrue(r.available_at_utc.lt(end).all());self.assertFalse(r[['origin_time_utc','hazard_hour']].duplicated().any())
        X=f.reindex(pd.DatetimeIndex(r.origin_time_utc))[list(FEATURES['minimal'])].to_numpy()
        self.assertEqual(X.shape[1],6);self.assertEqual(len(X),len(r.response))

    def test_missing_history_remains_explicit_unavailable_cells(self):
        path=self.history/'data/processed/five_asset_v1/hourly/DOGEUSDT/2021-01.parquet';body=path.read_bytes();path.unlink()
        try:r=inspect_inputs(self.data,self.history,self.catalog,self.root)
        finally:path.write_bytes(body)
        own=[c for c in r['cells'] if c['asset']=='DOGEUSDT'];self.assertEqual(len(own),6)
        self.assertTrue(all(c['input_status']=='unavailable' for c in own));self.assertEqual(r['available_cells'],18)

    def test_wrong_manifest_schema_and_financing_fail(self):
        bad=dict(self.catalog);bad['schema_version']='wrong'
        with self.assertRaises(ValueError):inspect_inputs(self.data,self.history,bad,self.root)
        bad={**self.catalog,'history_dataset':{**self.catalog['history_dataset'],'sha256':'f'*64}}
        with self.assertRaisesRegex(ValueError,'provenance'):inspect_inputs(self.data,self.history,bad,self.root)
        with self.assertRaises(ValueError):adapt_hourly(self.development(),'BTCUSDT','SOFR')

    def test_source_only_cli_export_with_external_synthetic_inputs(self):
        with tempfile.TemporaryDirectory() as td:
            checkout=Path(td)/'checkout';checkout.mkdir()
            with tarfile.open(fileobj=io.BytesIO(subprocess.check_output(['git','archive','HEAD'],cwd=ROOT))) as tar:tar.extractall(checkout,filter='data')
            for name in ['src/spot_perp_basis/data/forecast_inputs.py','src/spot_perp_basis/data/input_preflight.py','scripts/preflight_final_integration.py','configs/forecast_input_integration_v1.json']:
                p=checkout/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/name).read_bytes())
            env={**os.environ,'PYTHONPATH':str(checkout/'src'),'PYTHONDONTWRITEBYTECODE':'1'}
            code="""import socket,sys
socket.socket.connect=lambda *a,**k: (_ for _ in ()).throw(AssertionError('Network forbidden'))
from scripts.preflight_final_integration import main
raise SystemExit(main(sys.argv[1:]))
"""
            run=subprocess.run([sys.executable,'-B','-c',code,'--data-root',str(self.data),'--history-root',str(self.history),'--input-catalog',str(self.root/'catalog.json'),'--public-root',str(self.root)],cwd=checkout,env=env,capture_output=True,text=True)
            self.assertEqual(run.returncode,0,run.stdout+run.stderr);self.assertIn('"available_cells": 24',run.stdout)

    def test_scope_rejects_final_fitting(self):
        with structural_only(),self.assertRaises(PermissionError):fit_development_year({},2023,'zero_rate')
        with self.assertRaises(PermissionError):fit_development_year({},2023,'zero_rate')

    def test_scope_rejects_final_label_generation(self):
        f=raw_frame(pd.date_range('2023-01-01',periods=2,freq='h',tz='UTC'))
        with structural_only(),self.assertRaises(PermissionError):labels(f,START,END)

    def test_scope_rejects_forecasting_and_scoring(self):
        from spot_perp_basis.evaluation.survival_audit import matched_ipcw_scores
        with structural_only(),self.assertRaises(PermissionError):raw_probabilities({},pd.DataFrame())
        with structural_only(),self.assertRaises(PermissionError):matched_ipcw_scores(None,None,None)

    def test_frozen_files_unchanged_and_no_final_feature_builder_call(self):
        self.assertEqual(contract()['protocol'],'final-survival-evaluation-v1.0.0')
        with patch('spot_perp_basis.training.annual.causal_features',side_effect=AssertionError('Forbidden')):
            report=inspect_inputs(self.data,self.history,self.catalog,self.root)
        self.assertTrue(report['no_final_features_labels_fits_forecasts_scores_or_economics'])


if __name__=='__main__':unittest.main()
