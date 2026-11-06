"""Prefix/resume and storage parity, including real chunk-boundary hazards."""

import copy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from scripts.preflight_annual_forecasting import synthetic_contexts
from spot_perp_basis.training.annual import (
    ASSETS, MODELS, AnnualForecastSnapshot, fit_development_year, issue_forecasts,
    attach_outcomes, evaluate_forecasts, identity, portable,
)
from spot_perp_basis.training.checkpoints import CheckpointContractError, CheckpointUnsupportedError, save_checkpoint
from spot_perp_basis.training.operational import (
    ForecastCursor, operational_binding, annual_availability, start_forecasting,
    forecast_chunk, save_resume, load_resume, save_forecasts, load_forecasts,
    assemble_forecasts, score_forecasts,
    save_progress, load_progress,
)


ROOT=Path(__file__).resolve().parents[1]
BEGIN=pd.Timestamp('2022-01-01',tz='UTC')
STOP=BEGIN+pd.Timedelta(hours=72)


def select(contexts,begin,end):
    return {s:f.loc[(f.index>=begin)&(f.index<end)].copy() for s,f in contexts.items()}


def unsupported_snapshot(snapshot):
    p=snapshot.read()
    p['assets']['ADAUSDT']['fitted']['matched_logistic']['status']='unsupported_numerical_nonconvergence'
    p['assets']['DOGEUSDT']['calibration_maps']['calibrated_tree']['status']='unsupported_calibration_numerical'
    for a in p['assets'].values():
        a['fitting_identity']=identity(a['fitted']);a['calibration_identity']=identity(a['calibration_maps'])
    return AnnualForecastSnapshot(json.dumps(p,sort_keys=True,separators=(',',':'),allow_nan=False))


class AnnualOperationalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.contexts=synthetic_contexts();cls.snapshot=fit_development_year(cls.contexts,2022,'zero_rate')
        cls.binding=operational_binding(ROOT)
        cls.clock=pd.date_range(BEGIN,STOP,freq='h',inclusive='left',name='origin_time_utc')
        cls.original=issue_forecasts(cls.snapshot,cls.contexts,origins=cls.clock)

    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.directory=Path(self.temp.name)

    def start(self,snapshot=None,contexts=None):
        contexts=self.contexts if contexts is None else contexts
        return start_forecasting(snapshot or self.snapshot,{s:f.loc[f.index<BEGIN] for s,f in contexts.items()},binding=self.binding)

    def resume(self,state):
        pin=save_resume(state,self.directory)
        return load_resume(self.directory,pin,year=2022,financing='zero_rate',binding=self.binding),pin

    def test_chunked_resume_population_order_and_probabilities_exact(self):
        state=self.start();parts=[]
        for a,b in [(BEGIN,BEGIN+pd.Timedelta(hours=24)),(BEGIN+pd.Timedelta(hours=24),STOP)]:
            state,f=forecast_chunk(state,select(self.contexts,a,b));parts.append(f);state,_=self.resume(state)
        merged=assemble_forecasts(self.snapshot,parts,begin=BEGIN,end=STOP)
        pd.testing.assert_frame_equal(merged,self.original,check_exact=True)
        self.assertEqual(state.next_origin,STOP)

    def test_forecast_storage_exact_keys_metadata_and_idempotence(self):
        pin=save_forecasts(self.snapshot,self.original,self.directory,begin=BEGIN,end=STOP,binding=self.binding)
        before={p.name:p.stat().st_mtime_ns for p in self.directory.iterdir()}
        f=load_forecasts(self.snapshot,self.directory,pin,begin=BEGIN,end=STOP,binding=self.binding)
        pd.testing.assert_frame_equal(f,self.original,check_exact=True)
        self.assertEqual(save_forecasts(self.snapshot,f,self.directory,begin=BEGIN,end=STOP,binding=self.binding),pin)
        self.assertEqual(before,{p.name:p.stat().st_mtime_ns for p in self.directory.iterdir()})

    def test_pending_outcome_storage_without_issuance_filtering(self):
        early=attach_outcomes(self.snapshot,self.original,self.contexts,BEGIN)
        pin=save_forecasts(self.snapshot,early,self.directory,begin=BEGIN,end=STOP,binding=self.binding,as_of=BEGIN)
        result=load_forecasts(self.snapshot,self.directory,pin,begin=BEGIN,end=STOP,binding=self.binding,as_of=BEGIN)
        pd.testing.assert_frame_equal(early,result,check_exact=True)
        self.assertEqual(len(result),4*9*72)

    def test_resume_history_exact_and_repeated_save_unchanged(self):
        state,_=forecast_chunk(self.start(),select(self.contexts,BEGIN,STOP))
        loaded,pin=self.resume(state)
        for s in ASSETS:pd.testing.assert_frame_equal(loaded.contexts[s],state.contexts[s],check_exact=True)
        self.assertEqual(loaded.snapshot.content,state.snapshot.content)
        self.assertEqual(save_resume(loaded,self.directory),pin)

    def test_fresh_process_resume_reproduces_next_chunk(self):
        state,_=forecast_chunk(self.start(),select(self.contexts,BEGIN,BEGIN+pd.Timedelta(hours=24)))
        pin=save_resume(state,self.directory)
        program='''
import json,sys,pandas as pd
from pathlib import Path
from scripts.test_portable import deny_archive_contents
sys.addaudithook(deny_archive_contents)
from scripts.preflight_annual_forecasting import synthetic_contexts
from spot_perp_basis.training.operational import *
binding=operational_binding(Path.cwd())
state=load_resume(sys.argv[1],sys.argv[2],year=2022,financing='zero_rate',binding=binding)
end=state.next_origin+pd.Timedelta(hours=24)
c={s:f.loc[(f.index>=state.next_origin)&(f.index<end)] for s,f in synthetic_contexts().items()}
new,f=forecast_chunk(state,c)
pin=save_forecasts(state.snapshot,f,sys.argv[1],begin=state.next_origin,end=end,binding=binding)
print(pin)
'''
        child=subprocess.run([sys.executable,'-B','-c',program,str(self.directory),pin],cwd=ROOT,text=True,capture_output=True,check=True)
        loaded=load_forecasts(self.snapshot,self.directory,child.stdout.strip(),begin=BEGIN+pd.Timedelta(hours=24),end=BEGIN+pd.Timedelta(hours=48),binding=self.binding)
        _,expected=forecast_chunk(state,select(self.contexts,BEGIN+pd.Timedelta(hours=24),BEGIN+pd.Timedelta(hours=48)))
        pd.testing.assert_frame_equal(loaded,expected,check_exact=True)

    def test_open_paths_gap_and_zero_at_chunk_boundaries(self):
        c={s:f.copy() for s,f in self.contexts.items()}
        boundary=BEGIN+pd.Timedelta(hours=24)
        for f in c.values():
            h=np.arange(len(f));mask=(f.index>=BEGIN-pd.Timedelta(days=100))&(f.index<boundary)
            # One long open positive-rho path must not become a published
            # terminal record merely because a chunk stopped.
            f.loc[mask,'perp_close']=f.loc[mask,'spot_close']/(1-(1.-.5475)/1095)
            f.loc[mask,'theory_available']=True
            f.loc[boundary,'theory_available']=False
            f.loc[boundary+pd.Timedelta(hours=1),'perp_close']=f.loc[boundary+pd.Timedelta(hours=1),'spot_close']/(1-(-1.-.5475)/1095)
        state=self.start(contexts=c);parts=[]
        for a,b in [(BEGIN,boundary),(boundary,boundary+pd.Timedelta(hours=1)),(boundary+pd.Timedelta(hours=1),STOP)]:
            state,f=forecast_chunk(state,select(c,a,b));parts.append(f);state,_=self.resume(state)
        expected=issue_forecasts(self.snapshot,c,origins=self.clock)
        actual=assemble_forecasts(self.snapshot,parts,begin=BEGIN,end=STOP)
        pd.testing.assert_frame_equal(actual,expected,check_exact=True)
        rolling=actual.loc[actual.model.eq('rolling_magnitude_KM')]
        pub=pd.to_datetime(rolling.rolling_latest_record_publication_utc,utc=True)
        self.assertTrue((pub.isna() | pub.lt(rolling.origin_time_utc)).all())

    def test_exact_zero_crossing_boundary_uses_existing_target(self):
        from spot_perp_basis.data.labels import labels
        clock=pd.date_range(BEGIN,periods=5,freq='h')
        f=pd.DataFrame({'rho':[1.,1.,0.,-1.,-1.],'theory_available':True},index=clock)
        a=labels(f,BEGIN,clock[-1]+pd.Timedelta(hours=1));b=labels(f.iloc[:3],BEGIN,clock[2]+pd.Timedelta(hours=1))
        self.assertEqual(a.iloc[0].event_time_utc,clock[2]);self.assertEqual(b.iloc[0].event_time_utc,clock[2])
        self.assertEqual(a.iloc[0].record_available_at_utc,b.iloc[0].record_available_at_utc)

    def test_current_and_future_mutations_do_not_change_prior_issuance(self):
        state=self.start();next_state,f=forecast_chunk(state,select(self.contexts,BEGIN,STOP))
        later={s:g.copy() for s,g in self.contexts.items()}
        for g in later.values():g.loc[g.index>=STOP,'perp_close']*=1.1
        again,other=forecast_chunk(state,select(later,BEGIN,STOP))
        pd.testing.assert_frame_equal(f,other,check_exact=True)
        oracle=issue_forecasts(self.snapshot,later,origins=self.clock)
        pd.testing.assert_frame_equal(f,oracle,check_exact=True)
        self.assertEqual(next_state.history_identity,again.history_identity)

    def test_skip_overlap_duplicates_and_missing_asset_fail(self):
        state=self.start()
        cases=[select(self.contexts,BEGIN+pd.Timedelta(hours=1),STOP),
            select(self.contexts,BEGIN-pd.Timedelta(hours=1),STOP)]
        bad=select(self.contexts,BEGIN,STOP);bad['BTCUSDT']=pd.concat([bad['BTCUSDT'],bad['BTCUSDT'].iloc[-1:]])
        cases.append(bad);bad=select(self.contexts,BEGIN,STOP);bad.pop('ADAUSDT');cases.append(bad)
        for c in cases:
            with self.assertRaises(ValueError):forecast_chunk(state,c)

    def test_history_mutation_or_cursor_tampering_fails(self):
        state=self.start();s=ASSETS[0]
        state.contexts[s].iloc[-1,state.contexts[s].columns.get_loc('spot_close')]*=1.01
        with self.assertRaises(CheckpointContractError):forecast_chunk(state,select(self.contexts,BEGIN,STOP))
        with self.assertRaises(CheckpointContractError):save_resume(replace(self.start(),next_origin=STOP),self.directory)

    def test_manifest_frame_corruption_and_provenance_fail(self):
        pin=save_forecasts(self.snapshot,self.original,self.directory,begin=BEGIN,end=STOP,binding=self.binding)
        manifest=json.loads((self.directory/(pin+'.json')).read_text());parquet=self.directory/(manifest['frame']['file_sha256']+'.parquet')
        content=parquet.read_bytes();parquet.write_bytes(content[:-1])
        with self.assertRaises(CheckpointContractError):load_forecasts(self.snapshot,self.directory,pin,begin=BEGIN,end=STOP,binding=self.binding)
        parquet.write_bytes(content);wrong=copy.deepcopy(self.binding);wrong['operations_sha256']='0'*64
        with self.assertRaises(CheckpointContractError):load_forecasts(self.snapshot,self.directory,pin,begin=BEGIN,end=STOP,binding=wrong)

    def test_dropped_duplicate_and_extra_forecast_origins_fail(self):
        for f in [self.original.iloc[:-1],pd.concat([self.original,self.original.iloc[:1]]),self.original.loc[self.original.symbol.ne('ADAUSDT')]]:
            with self.assertRaises(CheckpointContractError):save_forecasts(self.snapshot,f,self.directory,begin=BEGIN,end=STOP,binding=self.binding)
        with self.assertRaises(CheckpointContractError):assemble_forecasts(self.snapshot,[self.original,self.original],begin=BEGIN,end=STOP)

    def test_partial_commits_recover_and_conflicting_content_fails(self):
        state=self.start()
        with patch('spot_perp_basis.training.operational.os.link',side_effect=OSError('injected crash')):
            with self.assertRaises(OSError):save_resume(state,self.directory)
        self.assertFalse(list(self.directory.glob('.forecast-*.tmp')))
        loaded,pin=self.resume(state);self.assertEqual(loaded.history_identity,state.history_identity)
        manifest=self.directory/(pin+'.json');manifest.write_bytes(b'conflict')
        with self.assertRaises(CheckpointContractError):save_resume(state,self.directory)
        self.assertEqual(manifest.read_bytes(),b'conflict')

    def test_future_year_requests_rejected_before_file_read(self):
        with patch('pathlib.Path.read_bytes',side_effect=AssertionError('forbidden read')):
            for year in (2023,2024,2025):
                with self.assertRaises(CheckpointContractError):load_resume(self.directory,'0'*64,year=year,financing='zero_rate',binding=self.binding)

    def test_unsupported_cases_persist_every_cell_and_null_prediction(self):
        s=unsupported_snapshot(self.snapshot);cells=annual_availability(s)
        unavailable=[c for c in cells if c['availability']=='unavailable'];self.assertEqual(len(cells),36);self.assertEqual(len(unavailable),3)
        with self.assertRaises(CheckpointUnsupportedError):save_checkpoint(s,self.directory/'annual_2022_zero_rate.checkpoint.json',provenance={})
        state=self.start(snapshot=s);state,f=forecast_chunk(state,select(self.contexts,BEGIN,STOP));state,pin=self.resume(state)
        self.assertEqual(annual_availability(state.snapshot),cells)
        sha=save_forecasts(s,f,self.directory,begin=BEGIN,end=STOP,binding=self.binding)
        loaded=load_forecasts(s,self.directory,sha,begin=BEGIN,end=STOP,binding=self.binding)
        self.assertEqual(len(loaded),4*9*72)
        attached=attach_outcomes(s,loaded,self.contexts,STOP)
        score=score_forecasts(s,attached,begin=BEGIN,end=STOP,as_of=STOP)
        self.assertEqual(score['annual_equal_asset']['status'],'unavailable_required_cell')
        self.assertEqual(sum(r['cell_status']=='unavailable_annual_model' for r in score['model_scores']),3)
        for cell in unavailable:
            g=loaded.loc[loaded.symbol.eq(cell['symbol'])&loaded.model.eq(cell['model'])]
            self.assertEqual(len(g),72);self.assertTrue(g.event_probability.isna().all())

    def test_canonical_scoring_unchanged_after_forecast_reload(self):
        attached=attach_outcomes(self.snapshot,self.original,self.contexts,STOP)
        pin=save_forecasts(self.snapshot,attached,self.directory,begin=BEGIN,end=STOP,binding=self.binding,as_of=STOP)
        reloaded=load_forecasts(self.snapshot,self.directory,pin,begin=BEGIN,end=STOP,binding=self.binding,as_of=STOP)
        pd.testing.assert_frame_equal(attached,reloaded,check_exact=True)
        score=score_forecasts(self.snapshot,reloaded,begin=BEGIN,end=STOP,as_of=STOP)
        baseline=evaluate_forecasts(self.snapshot,attached)
        score.pop('annual_availability')
        for row in score['model_scores']:
            for k in ('annual_availability','annual_diagnostic','cell_status'):row.pop(k)
        self.assertEqual(portable(score),portable(baseline))

    def test_progress_restores_completed_outputs_without_memory_ledger(self):
        state,f=forecast_chunk(self.start(),select(self.contexts,BEGIN,STOP))
        pin=save_forecasts(self.snapshot,f,self.directory,begin=BEGIN,end=STOP,binding=self.binding)
        completed=[{'begin':BEGIN.isoformat(),'end':STOP.isoformat(),'sha256':pin}]
        progress=save_progress(state,self.directory,completed)
        del state,completed
        resumed,completed=load_progress(self.directory,progress,year=2022,financing='zero_rate',binding=self.binding)
        self.assertEqual(resumed.next_origin,STOP);self.assertEqual(len(completed),1)
        self.assertEqual(save_progress(resumed,self.directory,completed),progress)
        f=load_forecasts(resumed.snapshot,self.directory,completed[0]['sha256'],begin=BEGIN,end=STOP,binding=self.binding)
        pd.testing.assert_frame_equal(f,self.original,check_exact=True)

    def test_progress_rejects_missing_duplicate_or_uncommitted_output(self):
        state,f=forecast_chunk(self.start(),select(self.contexts,BEGIN,STOP))
        pin=save_forecasts(self.snapshot,f,self.directory,begin=BEGIN,end=STOP,binding=self.binding)
        entry={'begin':BEGIN.isoformat(),'end':STOP.isoformat(),'sha256':pin}
        for completed in ([],[entry,entry],[{**entry,'begin':(BEGIN+pd.Timedelta(hours=1)).isoformat()}]):
            with self.assertRaises(CheckpointContractError):save_progress(state,self.directory,completed)

    def test_normalized_nanosecond_observations_match_annual_decision_clock(self):
        c={s:f.copy() for s,f in self.contexts.items()}
        for f in c.values():f.index=f.index.as_unit('ns')
        state=self.start(contexts=c);state,f=forecast_chunk(state,select(c,BEGIN,STOP))
        expected=issue_forecasts(self.snapshot,c,origins=self.clock)
        pd.testing.assert_frame_equal(f,expected,check_exact=True)
        loaded,pin=self.resume(state)
        for s in ASSETS:pd.testing.assert_frame_equal(loaded.contexts[s],state.contexts[s],check_exact=True)


if __name__=='__main__':unittest.main()
