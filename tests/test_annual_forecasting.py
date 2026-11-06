"""Adversarial annual integration checks, using dated synthetic frames only."""

import copy
from dataclasses import FrozenInstanceError
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from scripts.preflight_annual_forecasting import synthetic_contexts, frozen_contract
from scripts.checkpoint_survival_baseline import Sources
from spot_perp_basis.data.features import FEATURES
from spot_perp_basis.data.labels import labels
from spot_perp_basis.data.windows import START, END, AnnualWindows, annual_windows, origin_mask, require_training_times
from spot_perp_basis.evaluation.statistics import complete_matrix_mean, joint_calendar_interval
from spot_perp_basis.evaluation.survival_audit import matched_ipcw_scores
from spot_perp_basis.models import trees, logistic
from spot_perp_basis.models.calibration import platt_fit, calibrate
from spot_perp_basis.models.survival import km_fit, rolling_curves
from spot_perp_basis.training.annual import (
    ASSETS, MODELS, AnnualForecastSnapshot, causal_features, fit_development_year,
    issue_forecasts, attach_outcomes, evaluate_forecasts, portable, identity,
)
from spot_perp_basis.training.fit import fit_window, calibration_sample, fit_calibrators, array_hash


def stamp(text):return pd.Timestamp(text,tz='UTC')


class AnnualCalendarTests(unittest.TestCase):
    def test_registered_windows_equal_frozen_yaml(self):
        self.assertEqual(len(frozen_contract()),3)
        for year in (2023,2024,2025):
            w=annual_windows(year)
            self.assertEqual(w.fit[1]-pd.DateOffset(months=20),w.fit[0])
            self.assertEqual(w.calibration[1]-pd.DateOffset(months=4),w.calibration[0])
            self.assertEqual(w.fit[1],w.calibration[0]);self.assertEqual(w.calibration[1],w.evaluation[0])
            self.assertEqual(w.as_dict()['role'],'registered_calendar_only')
        self.assertEqual(annual_windows(2023).fit,(stamp('2021-01-01'),stamp('2022-09-01')))
        with self.assertRaises(ValueError):annual_windows(True)
        w=annual_windows(2024)
        with self.assertRaises(ValueError):AnnualWindows(2024,(w.fit[0]-pd.Timedelta(hours=1),w.fit[1]),w.calibration,w.evaluation)

    def test_half_open_lower_bound_and_strict_purge(self):
        b,e=stamp('2021-01-01'),stamp('2021-02-01')
        ix=pd.DatetimeIndex([b-pd.Timedelta(hours=1),b,e-pd.Timedelta(hours=17),e-pd.Timedelta(hours=16),e-pd.Timedelta(hours=1),e])
        np.testing.assert_array_equal(origin_mask(ix,b,e),[0,1,1,1,1,0])
        np.testing.assert_array_equal(origin_mask(ix,b,e,purge=True),[0,1,1,0,0,0])

    def test_leap_year_metadata_and_year_end_synthetic_censor(self):
        w=annual_windows(2024);ix=pd.date_range(*w.evaluation,freq='h',inclusive='left')
        self.assertEqual(len(ix),8784);self.assertIn(stamp('2024-02-29'),ix)
        # Pure simulated labels, no future-period observations or model execution.
        f=pd.DataFrame({'rho':1.,'theory_available':True},index=ix[-20:])
        p=labels(f,ix[-20],w.evaluation[1]);self.assertEqual(len(p),20)
        self.assertEqual(p.iloc[-1].followup_hours,0)
        self.assertFalse(p.iloc[-1].event_observed)
        self.assertEqual(p.iloc[-1].record_available_at_utc,stamp('2024-12-31T23:00'))
        self.assertTrue(pd.isna(p.iloc[-1].event_by_8h))
        self.assertTrue(p.boundary_window_purged.iloc[-1]) # diagnostics only, not an evaluation screen

    def test_final_numerical_execution_fails_before_using_contexts(self):
        class NoRead(dict):
            def __iter__(self):raise AssertionError('Contexts accessed before permission guard')
            def __getitem__(self,key):raise AssertionError('Final content accessed')
        for year in (2023,2024,2025):
            with self.assertRaises(PermissionError):fit_development_year(NoRead(),year,'zero_rate')
        with self.assertRaises(ValueError):require_training_times([stamp('2023-01-01')])
        self.assertEqual(END,stamp('2023-01-01'))

    def test_development_allowlist_rejects_future_before_io(self):
        src=Sources(('zero_rate',))
        with self.assertRaises(ValueError):src.path('data/raw/BTCUSDT/2024-01.zip')
        with patch('pandas.read_parquet',side_effect=AssertionError('IO occurred')):
            with self.assertRaises(ValueError):src.frame('not_read',begin=stamp('2022-01-01'),end=stamp('2024-01-01'))
        with self.assertRaises(PermissionError):src.guard('socket.connect',())
        with self.assertRaises(PermissionError):src.guard('open',(str(Path.cwd()/'data/raw/2025.zip'),'r',0))


class AnnualAdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Unit fixture is deliberately bounded. The separate runner uses every
        # hour in three years and all frozen rounds; no E2E truncation.
        cls.small=synthetic_contexts(2000)
        cls.contexts=synthetic_contexts()
        cls.snapshot=fit_development_year(cls.small,2022,'zero_rate')
        cls.f=causal_features(cls.small['BTCUSDT'],'BTCUSDT','zero_rate')
        cls.fitted=cls.snapshot.read()['assets']['BTCUSDT']['fitted']

    def test_lower_bound_warmup_matrix_scaler_cuts_and_KM(self):
        begin=self.f.index[100];end=self.f.index[1000]
        fit,risk=fit_window(self.f,begin,end)
        self.assertGreaterEqual(risk.origin_time_utc.min(),begin)
        self.assertTrue((risk.origin_time_utc+pd.Timedelta(hours=16)<end).all())
        self.assertTrue(risk.available_at_utc.lt(end).all())
        x=self.f.reindex(pd.DatetimeIndex(risk.origin_time_utc))[list(FEATURES['minimal'])].to_numpy()
        self.assertEqual(array_hash(np.c_[x,risk.hazard_hour]),fit['manifest']['raw_X_and_hour_sha256'])
        self.assertEqual(array_hash(risk.response.to_numpy(float)),fit['manifest']['response_sha256'])
        np.testing.assert_allclose(fit['matched_logistic']['mean'],x.mean(axis=0))
        np.testing.assert_allclose(fit['matched_logistic']['scale'],np.where(x.std(axis=0)>1e-12,x.std(axis=0),1))
        raw=np.c_[x,risk.hazard_hour]
        for j,cuts in enumerate(fit['tree']['cuts']):
            np.testing.assert_array_equal(cuts,np.unique(np.quantile(raw[:,j],np.arange(1,32)/32)))
        lab=labels(self.f,begin,end)
        empirical=(~lab.boundary_window_purged)&self.f.minimal_eligible.reindex(lab.index)&lab.survival_origin_eligible
        primary=(~lab.boundary_window_purged)&self.f.minimal_eligible36.reindex(lab.index)&lab.survival_origin_eligible
        self.assertEqual(fit['empirical']['asset_km'],km_fit(lab.loc[empirical,'followup_hours'],lab.loc[empirical,'event_observed']))
        self.assertEqual(fit['scoring_censor_km'],km_fit(lab.loc[primary,'followup_hours'],lab.loc[primary,'event_observed'],reverse=True))
        np.testing.assert_array_equal(fit['empirical']['magnitude_cutpoints'],np.unique(np.quantile(self.f.abs_rho.reindex(lab.index)[empirical],[.2,.4,.6,.8])))
        self.assertTrue(self.f.loc[begin,'minimal_eligible36']) # history before lower bound is warmup

    def test_features_and_training_invariant_to_future_mutation(self):
        b=self.small['BTCUSDT'].copy();cut=b.index[1000]
        b.loc[b.index>cut,'perp_close']*=1.2
        changed=causal_features(b,'BTCUSDT','zero_rate')
        pd.testing.assert_frame_equal(self.f.loc[:cut,list(FEATURES['minimal'])],changed.loc[:cut,list(FEATURES['minimal'])])
        begin,end=self.f.index[100],self.f.index[900]
        a,_=fit_window(self.f,begin,end);b,_=fit_window(changed,begin,end)
        self.assertEqual(identity(a),identity(b))

    def test_minute_and_funding_publication_boundaries(self):
        b=self.small['BTCUSDT'].iloc[:25].copy();t=b.index[-1]
        b['bar_available_at_utc']=b.bar_available_at_utc.dt.as_unit('ns')
        b.loc[t,'bar_available_at_utc']=t+pd.Timedelta(nanoseconds=1)
        with self.assertRaises(ValueError):causal_features(b,'BTCUSDT','zero_rate')
        b=self.small['BTCUSDT'].iloc[:25].copy();b.loc[t,'bar_open_time_utc']=t
        with self.assertRaises(ValueError):causal_features(b,'BTCUSDT','zero_rate')
        b=self.small['BTCUSDT'].iloc[:25].copy()
        b['funding_known']=True;b['last_funding_settlement_utc']=b.index-pd.Timedelta(minutes=1)
        b['last_funding_available_at_utc']=b.index
        causal_features(b,'BTCUSDT','zero_rate') # publication exactly at origin is allowed
        b.loc[t,'last_funding_available_at_utc']=t-pd.Timedelta(seconds=1)
        with self.assertRaises(ValueError):causal_features(b,'BTCUSDT','zero_rate')
        b['last_funding_available_at_utc']=b.index
        b.loc[t,'last_funding_settlement_utc']=t-pd.Timedelta(hours=13)
        with self.assertRaises(ValueError):causal_features(b,'BTCUSDT','zero_rate')

    def test_missing_hours_and_mixed_assets_are_rejected(self):
        b=self.small['BTCUSDT'].drop(self.small['BTCUSDT'].index[20])
        with self.assertRaises(ValueError):causal_features(b,'BTCUSDT','zero_rate')
        b=self.small['BTCUSDT'].copy();b['symbol']='DOGEUSDT'
        with self.assertRaises(ValueError):causal_features(b,'BTCUSDT','zero_rate')

    def test_snapshot_is_immutable_and_nested_reads_are_copies(self):
        before=self.snapshot.sha256;p=self.snapshot.read();p['assets'].clear()
        self.assertEqual(self.snapshot.sha256,before);self.assertEqual(set(self.snapshot.read()['assets']),set(ASSETS))
        with self.assertRaises(FrozenInstanceError):self.snapshot.content='changed'

    def test_snapshot_identity_and_cutoff_tampering_rejected(self):
        p=self.snapshot.read();p['assets']['BTCUSDT']['fitted']['tree']['depth']=5
        bad=AnnualForecastSnapshot(json.dumps(p))
        with self.assertRaises(ValueError):issue_forecasts(bad,self.contexts,origins=pd.DatetimeIndex([stamp('2022-01-01')]))

    def test_eval_values_never_enter_fit_snapshot(self):
        # Use the already fitted numerical state to inspect adapter boundaries,
        # avoiding a redundant four-model optimizer run in this boundary test.
        def existing(context,begin,end):
            self.assertLess(context.index.max(),stamp('2022-01-01'))
            symbol=context.symbol.iloc[0]
            return copy.deepcopy(self.snapshot.read()['assets'][symbol]['fitted']),None
        original=self.contexts;mutated={s:f.copy() for s,f in original.items()}
        for f in mutated.values():f.loc[f.index>=stamp('2022-01-01'),'perp_close']=-1e50
        with patch('spot_perp_basis.training.annual.fit_window',side_effect=existing):
            a=fit_development_year(original,2022,'zero_rate')
            b=fit_development_year(mutated,2022,'zero_rate')
        self.assertEqual(a.sha256,b.sha256)

    def test_issuance_and_later_label_attachment_are_separate(self):
        origin=stamp('2022-01-01');clock=pd.DatetimeIndex([origin,stamp('2022-12-31T23:00')])
        issued=issue_forecasts(self.snapshot,self.contexts,origins=clock)
        self.assertEqual(len(issued),4*2*len(MODELS))
        self.assertNotIn('followup_hours',issued)
        early=attach_outcomes(self.snapshot,issued,self.contexts,origin)
        pd.testing.assert_frame_equal(early[issued.columns],issued)
        self.assertTrue(early.label_available_at_utc.isna().all())
        self.assertTrue(early.loc[early.origin_time_utc.gt(origin),'outcome_status'].eq('pending_observation').all())
        late=attach_outcomes(self.snapshot,issued,self.contexts,stamp('2022-12-31T23:00'))
        pd.testing.assert_frame_equal(late[issued.columns],issued)
        last=late.loc[late.origin_time_utc.eq(clock[-1])]
        self.assertTrue(last.followup_hours.eq(0).all());self.assertTrue(last.event_by_8h.isna().all())
        self.assertTrue(issued.loc[issued.model.eq('calibrated_tree'),'forecast_status'].eq('unsupported_model_or_map').all())

    def test_origin_predictions_and_rolling_references_ignore_future(self):
        origin=stamp('2022-04-01');clock=pd.DatetimeIndex([origin])
        a=issue_forecasts(self.snapshot,self.contexts,origins=clock)
        mutated={s:f.copy() for s,f in self.contexts.items()}
        for f in mutated.values():f.loc[f.index>origin,'perp_close']*=1.1
        b=issue_forecasts(self.snapshot,mutated,origins=clock)
        pd.testing.assert_frame_equal(a,b)
        rolling=a.loc[a.model.eq('rolling_magnitude_KM')]
        self.assertTrue(pd.to_datetime(rolling.rolling_latest_record_publication_utc,utc=True).lt(origin).all())

    def test_unsupported_primary_fit_not_replaced_in_scoring(self):
        p=self.snapshot.read();a=p['assets']['ADAUSDT'];a['fitted']['tree']={'status':'unsupported_training'}
        a['fitting_identity']=identity(a['fitted'])
        snapshot=AnnualForecastSnapshot(json.dumps(p))
        clock=pd.date_range('2022-01-01',periods=24,freq='h',tz='UTC')
        issued=issue_forecasts(snapshot,self.contexts,origins=clock)
        bad=issued.loc[issued.symbol.eq('ADAUSDT')&issued.model.eq('raw_tree')]
        self.assertEqual(len(bad),24);self.assertTrue(bad.event_probability.isna().all())
        ev=evaluate_forecasts(snapshot,attach_outcomes(snapshot,issued,self.contexts,clock[-1]))
        self.assertEqual(ev['annual_equal_asset']['status'],'unavailable_required_cell')
        self.assertEqual(ev['annual_uncertainty']['interval_status'],'unavailable_required_asset_or_score')

    def test_financing_controls_and_forecast_identity_cannot_mix(self):
        f=self.small['BTCUSDT'].copy();f['annual_rate']=.04
        zero=causal_features(self.small['BTCUSDT'],'BTCUSDT','zero_rate')
        effr=causal_features(f,'BTCUSDT','nyfed_effr_lag7d')
        good=zero.theory_available
        np.testing.assert_allclose(zero.rho[good]-effr.rho[good],.04)
        issued=issue_forecasts(self.snapshot,self.contexts,origins=pd.DatetimeIndex([stamp('2022-01-01')]))
        issued.loc[0,'financing']='nyfed_effr_lag7d'
        with self.assertRaises(ValueError):attach_outcomes(self.snapshot,issued,self.contexts,stamp('2022-01-01'))

    def test_unsupported_censor_blocks_scores_and_interval_together(self):
        p=self.snapshot.read();a=p['assets']['BTCUSDT']
        a['fitted']['scoring_censor_km']={'status':'fit','n':100,'maximum_followup':7,'survival':[1.]*8}
        a['fitting_identity']=identity(a['fitted']);snapshot=AnnualForecastSnapshot(json.dumps(p))
        clock=pd.date_range('2022-01-01',periods=48,freq='h',tz='UTC')
        issued=issue_forecasts(snapshot,self.contexts,origins=clock)
        ev=evaluate_forecasts(snapshot,attach_outcomes(snapshot,issued,self.contexts,clock[-1]))
        primary=next(x for x in ev['matched_panels'] if x['symbol']=='BTCUSDT' and x['panel']=='primary')
        self.assertTrue(all(np.isnan(s['ipcw_brier']) for s in primary['scores'].values()))
        self.assertTrue(np.isnan(ev['primary_differences'][0]['gain']))
        self.assertEqual(ev['annual_equal_asset']['status'],'unavailable_required_cell')


class AnnualCalibrationAndAggregationTests(unittest.TestCase):
    @staticmethod
    def samples(counts):
        result={}
        for s,n in zip(ASSETS,counts):
            ix=pd.date_range('2021-09-01',periods=n,freq='h',tz='UTC')
            result[s]=pd.DataFrame({'symbol':s,'origin_time_utc':ix,
                'prediction':np.resize([.3,.7],n),'actual':np.resize([0.,1.],n),
                'label_available_utc':ix+pd.Timedelta(hours=8)})
        return result

    def test_individual_support_does_not_fit_pool(self):
        with patch('spot_perp_basis.training.fit.platt_fit',wraps=platt_fit) as fit:
            maps=fit_calibrators(self.samples([200]*4),stamp('2021-09-01'),stamp('2022-01-01'))
        self.assertEqual(fit.call_count,4)
        self.assertTrue(all(m['calibration_source']=='asset' and m['status']=='fit' for m in maps.values()))

    def test_registered_pool_is_single_lazy_fallback_only(self):
        with patch('spot_perp_basis.training.fit.platt_fit',wraps=platt_fit) as fit:
            maps=fit_calibrators(self.samples([10,200,200,10]),stamp('2021-09-01'),stamp('2022-01-01'))
        self.assertEqual(fit.call_count,5)
        self.assertEqual(sum(c.args[2:]==(400,40) for c in fit.call_args_list),1)
        self.assertEqual(maps['BTCUSDT']['calibration_source'],'pooled_primary_assets')
        self.assertEqual(maps['ETHUSDT']['calibration_source'],'asset')
        self.assertEqual(maps['ADAUSDT']['unique_asset_origins'],420)
        self.assertGreaterEqual(maps['BTCUSDT']['slope'],0)

    def test_failed_pool_remains_unavailable(self):
        maps=fit_calibrators(self.samples([10]*4),stamp('2021-09-01'),stamp('2022-01-01'))
        self.assertTrue(all(m['status']=='unsupported_calibration_support' for m in maps.values()))
        self.assertTrue(np.isnan(calibrate(maps['BTCUSDT'],[.2,.5])).all())
        with patch('spot_perp_basis.training.fit.platt_fit',return_value={'status':'unsupported_calibration_numerical','n':800}):
            failed=fit_calibrators(self.samples([200]*4),stamp('2021-09-01'),stamp('2022-01-01'))
        self.assertTrue(all(m['status']=='unsupported_calibration_numerical' for m in failed.values()))

    def test_calibration_cutoff_purge_and_publication(self):
        f=causal_features(synthetic_contexts(200)['BTCUSDT'],'BTCUSDT','zero_rate')
        begin,end=f.index[50],f.index[150];lab=labels(f,begin,end);pred=pd.Series(.5,index=lab.index)
        # Event publication exactly at cutoff is forbidden independently of purge.
        i=lab.index[0];lab.loc[i,'available_8h_utc']=end
        sample=calibration_sample(pred,f,lab,'BTCUSDT',begin,end)
        self.assertNotIn(i,sample.origin_time_utc.to_list())
        self.assertTrue((sample.origin_time_utc+pd.Timedelta(hours=16)<end).all())
        self.assertTrue(sample.label_available_utc.lt(end).all())
        samples=self.samples([200]*4);samples['BTCUSDT'].loc[0,'label_available_utc']=stamp('2022-01-01')
        with self.assertRaises(ValueError):fit_calibrators(samples,stamp('2021-09-01'),stamp('2022-01-01'))
        samples=self.samples([200]*4);samples['BTCUSDT']=pd.concat([samples['BTCUSDT']]*2)
        with self.assertRaises(ValueError):fit_calibrators(samples,stamp('2021-09-01'),stamp('2022-01-01'))

    def test_complete_twelve_cell_mean_separate_controls_no_reweighting(self):
        rows=pd.DataFrame([{'symbol':a,'year':y,'financing':r,'gain':float(j+i)}
            for r in ('zero_rate','nyfed_effr_lag7d') for i,y in enumerate((2023,2024,2025)) for j,a in enumerate(ASSETS)])
        out=complete_matrix_mean(rows,ASSETS,[2023,2024,2025],'zero_rate')
        self.assertEqual(out['gain'],2.5);self.assertEqual(out['weight_per_cell'],1/12)
        self.assertNotIn('ci_low',out)
        rows.loc[0,'gain']=np.nan
        self.assertIsNone(complete_matrix_mean(rows,ASSETS,[2023,2024,2025],'zero_rate')['gain'])
        self.assertEqual(complete_matrix_mean(rows,ASSETS,[2023,2024,2025],'nyfed_effr_lag7d')['gain'],2.5)
        with self.assertRaises(ValueError):complete_matrix_mean(pd.concat([rows,rows]),ASSETS,[2023,2024,2025],'zero_rate')

    def test_joint_four_asset_draws_are_shared_and_support_is_strict(self):
        clock=pd.date_range('2024-01-01','2025-01-01',freq='D',inclusive='left',tz='UTC')
        records=pd.concat([pd.DataFrame({'symbol':s,'origin_time_utc':clock,'gain':np.arange(366)*(.001+j*.0001)}) for j,s in enumerate(ASSETS)],ignore_index=True)
        real=np.random.default_rng;calls=[]
        class Generator:
            def __init__(self,seed):self.rng=real(seed)
            def multinomial(self,n,pvals,size):
                calls.append((n,len(pvals),size));return self.rng.multinomial(n,pvals,size=size)
        with patch('spot_perp_basis.evaluation.statistics.np.random.default_rng',side_effect=Generator):
            out=joint_calendar_interval(records,2024,ASSETS)
        self.assertEqual(calls,[(27,27,1000)]) # one draw matrix for all four
        self.assertEqual(out['paired_origins'],1464);self.assertEqual(out['assets'],4)
        sparse=records.loc[~records.symbol.eq('ADAUSDT')]
        self.assertIsNone(joint_calendar_interval(sparse,2024,ASSETS)['macro_gain'])
        with self.assertRaises(ValueError):joint_calendar_interval(pd.concat([records,records]),2024,ASSETS)

    def test_rolling_terminal_publication_is_strict_and_cutpoints_frozen(self):
        f=causal_features(synthetic_contexts(2000)['BTCUSDT'],'BTCUSDT','zero_rate')
        terminal=labels(f,f.index.min(),f.index[-1]+pd.Timedelta(hours=1))
        origin=f.index[1500];fit={'magnitude_cutpoints':[.5,1.],
            'magnitude_curves':[km_fit([10]*20,[False]*20)]*3}
        a,detail=rolling_curves(f,pd.DatetimeIndex([origin]),fit,terminal,horizons=(8,))
        self.assertLess(detail.iloc[0].maximum_used_record_publication_utc,origin)
        changed=terminal.copy();future=changed.record_available_at_utc.ge(origin)
        changed.loc[future,'followup_hours']=1;changed.loc[future,'event_observed']=True
        b,_=rolling_curves(f,pd.DatetimeIndex([origin]),fit,changed,horizons=(8,))
        np.testing.assert_array_equal(a,b);self.assertEqual(fit['magnitude_cutpoints'],[.5,1.])


if __name__=='__main__':unittest.main()
