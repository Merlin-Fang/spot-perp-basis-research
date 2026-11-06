"""Hand-derived event/censor examples and portable development leakage audits."""

import copy
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from scripts.checkpoint_survival_baseline import (
    origin_segment, audit_eight_hour_labels, joint_calendar_summary, Sources, ASSETS, BTC_ETH,
)
from spot_perp_basis.data.features import features, FEATURES
from spot_perp_basis.data.funding import prepare_funding, join_funding
from spot_perp_basis.data.labels import first_passage, labels
from spot_perp_basis.data.market import align_markets
from spot_perp_basis.evaluation.statistics import block_interval
from spot_perp_basis.evaluation.survival_audit import probability_audit, matched_ipcw_scores
from spot_perp_basis.models import logistic, trees
from spot_perp_basis.models.survival import km_fit, km_predict, rolling_curves
from spot_perp_basis.training.fit import hourly_risk_set, calibration_sample


def context(rho):
    rho = np.asarray(rho,float)
    clock = pd.date_range('2020-01-01',periods=len(rho),freq='h',tz='UTC',name='origin_time_utc')
    f = pd.DataFrame({'rho':rho,'abs_rho':abs(rho),'theory_available':np.isfinite(rho),
        'spot_close':100.,'perp_close':100.1,'annual_rate':0.,'rich_eligible':False},index=clock)
    f['rho_change1'] = f.rho.diff()
    f['minimal_eligible'] = f.theory_available & f.rho.ne(0)
    return features(f,cohorts=('minimal',))


class SurvivalCheckpointTests(unittest.TestCase):
    def audit_origin(self, values, duration, event, reason, available_hour, valid=None):
        f = context(values)
        if valid is not None:f['theory_available']=valid
        end=f.index[-1]+pd.Timedelta(hours=1)
        lab=labels(f,f.index[0],end)
        row=lab.iloc[0]
        self.assertEqual(row.followup_hours,duration)
        self.assertEqual(bool(row.event_observed),event)
        self.assertEqual(row.censor_reason,reason)
        self.assertEqual(row.record_available_at_utc,f.index[available_hour])
        if event:self.assertEqual(row.event_time_utc,f.index[duration])
        else:self.assertTrue(pd.isna(row.event_time_utc))
        # Isolate origin zero with a manually ready feature flag to audit likelihood trials.
        f['minimal_eligible36']=False;f.iloc[0,f.columns.get_loc('minimal_eligible36')]=True
        risk=hourly_risk_set(f,lab,'minimal',end)
        k=list(range(1,min(duration,16)+1))
        self.assertEqual(risk.hazard_hour.tolist(),k)
        self.assertEqual(risk.response.tolist(),[int(event and j==duration) for j in k])
        self.assertEqual(pd.DatetimeIndex(risk.available_at_utc).tolist(),[f.index[j] for j in k])
        if event and duration<=8:
            self.assertEqual(row.event_by_8h,1.)
            self.assertEqual(row.available_8h_utc,f.index[duration])
        elif duration>=8:
            self.assertEqual(row.event_by_8h,0.)
            self.assertEqual(row.available_8h_utc,f.index[8])
        else:
            self.assertTrue(pd.isna(row.event_by_8h))
            self.assertTrue(pd.isna(row.available_8h_utc))

    def test_first_crossing_hour_1(self):
        self.audit_origin([2,-1]+[1]*30,1,True,'observed_crossing',1)

    def test_first_crossing_hour_8(self):
        self.audit_origin([2]*8+[-1]+[2]*23,8,True,'observed_crossing',8)

    def test_first_crossing_after_8_is_8h_survivor(self):
        self.audit_origin([2]*9+[-1]+[2]*22,9,True,'observed_crossing',9)

    def test_exact_zero_touch_and_opposite_direction(self):
        self.audit_origin([2,0]+[2]*30,1,True,'observed_crossing',1)
        self.audit_origin([-2,-1,0]+[-2]*29,2,True,'observed_crossing',2)

    def test_missing_future_hour_censors_before_gap_discovered_later(self):
        self.audit_origin([2,2,np.nan,-1]+[2]*28,1,False,'missing_or_inactive_hour',2)

    def test_inactive_hour_has_same_gap_censoring_and_no_postgap_negatives(self):
        self.audit_origin([2,2,2,-1]+[2]*28,1,False,'missing_or_inactive_hour',2,
                          valid=[True,True,False]+[True]*29)

    def test_administrative_no_event_survives_beyond_horizon(self):
        self.audit_origin([2]*32,31,False,'administrative_prefix_end',31)

    def test_boundary_touch_is_excluded_and_fit_calibration_purge_is_strict(self):
        f=context([2]*24+[0]+[2]*15)
        cutoff=f.index[24]
        ff,lab=origin_segment(f,f.index[16],cutoff)
        self.assertEqual(lab.iloc[0].followup_hours,7)
        self.assertFalse(lab.iloc[0].event_observed)
        self.assertTrue(lab.boundary_window_purged.all())
        self.assertTrue(hourly_risk_set(ff,lab,'minimal',cutoff).empty)
        sample=calibration_sample(pd.Series(.5,index=lab.index),ff,lab,'BTCUSDT',f.index[16],cutoff)
        self.assertTrue(sample.empty)
        # Equality origin+16h == cutoff is purged even if an early crossing was observed.
        _,all_lab=origin_segment(f,f.index[0],cutoff)
        self.assertFalse(all_lab.boundary_window_purged.loc[f.index[7]])
        self.assertTrue(all_lab.boundary_window_purged.loc[f.index[8]])

    def test_independent_forward_audit_detects_early_label_publication(self):
        f=context([2]*12+[-1]*12+[2]*12)
        lab=labels(f,f.index[0],f.index[-1]+pd.Timedelta(hours=1))
        counts=audit_eight_hour_labels(f,lab)
        self.assertEqual(counts['origins_checked'],36)
        self.assertGreater(counts['early_censored'],0)
        self.assertGreater(counts['events'],0)
        changed=lab.copy();changed.iloc[4,changed.columns.get_loc('available_8h_utc')]=f.index[4]
        with self.assertRaises(AssertionError):audit_eight_hour_labels(f,changed)

    def test_minimum_past_12_and_missing_scheduled_previous_hour(self):
        f=context([2]*40)
        self.assertFalse(f.minimal_eligible36.iloc[:11].any())
        self.assertTrue(f.minimal_eligible36.iloc[11])
        f=context([2]*20+[np.nan]+[2]*20)
        self.assertTrue(np.isnan(f.rho_change1.iloc[21]))
        self.assertFalse(f.minimal_eligible36.iloc[21])
        self.assertTrue(f.minimal_eligible36.iloc[22])
        with self.assertRaises(ValueError):
            origin_segment(f.drop(f.index[20]),f.index[0],f.index[-1])

    def test_completed_minute_end_availability_exact_hour(self):
        clock=pd.date_range('2020-01-01T00:58:00Z',periods=3,freq='min')
        def bars(prices):
            return pd.DataFrame({**{x:prices for x in ['open','high','low','close']},
                **{x:1. for x in ['volume','quote_volume','trades','taker_buy_volume','taker_buy_quote_volume']},'valid':True},index=clock)
        paired,_=align_markets(bars([10,20,999]),bars([11,21,9999]))
        at=pd.Timestamp('2020-01-01T01:00:00Z')
        self.assertEqual(paired.loc[at,'spot_close'],20)
        self.assertEqual(paired.loc[at,'bar_open_time_utc'],at-pd.Timedelta(minutes=1))
        self.assertEqual(paired.index[-1],at+pd.Timedelta(minutes=1))
        self.assertEqual(paired.loc[:at].iloc[-1].spot_close,20)

    def test_funding_backward_availability_exact_match_and_staleness(self):
        times=pd.to_datetime(['2020-01-01T00:00:00Z','2020-01-01T08:00:00Z'])
        event=prepare_funding(pd.DataFrame({'settlement_time_utc':times,'funding_rate':[.001,-.002],
                                           'funding_interval_hours':[8.,8.]}))
        origins=pd.to_datetime(['2020-01-01T08:00:00Z','2020-01-01T08:01:00Z','2020-01-01T20:01:00Z'])
        f=pd.DataFrame(index=origins.rename('decision_time_utc'))
        joined=join_funding(f,event)
        np.testing.assert_allclose(joined.last_settled_funding_rate.iloc[:2],[.001,-.002])
        self.assertFalse(joined.funding_known.iloc[2])
        self.assertTrue(joined.last_funding_available_at_utc.le(joined.index).all())

    def test_minute_persistence_reversal_zero_touch_and_missing_interior(self):
        clock=pd.date_range('2020-01-01',periods=9,freq='h',tz='UTC')
        cases=[]
        v=np.full(481,2.);v[30:]=-1;cases.append(('persists',v,30,1))
        v=np.full(481,2.);v[30:45]=-1;cases.append(('reverses',v,30,None))
        v=np.full(481,2.);v[60]=0;cases.append(('hour_touch',v,60,1))
        for name,v,minute,hour in cases:
            with self.subTest(name=name):
                self.assertEqual(np.flatnonzero(v[1:]<=0)[0]+1,minute)
                row=first_passage(v[::60],np.ones(9,bool),clock).iloc[0]
                self.assertEqual(bool(row.event_observed),hour is not None)
                if hour:self.assertEqual(row.followup_hours,hour)
        v=np.full(481,2.);v[120:]=-1;v[30]=np.nan
        # The hourly process observes no gap at hour endpoints, hence hour-2 event.
        row=first_passage(v[::60],np.isfinite(v[::60]),clock).iloc[0]
        self.assertEqual(row.followup_hours,2)
        self.assertTrue(row.event_observed)
        self.assertEqual(np.flatnonzero(~np.isfinite(v))[0]-1,29)  # minute process censors
        self.assertFalse(np.isfinite(v).all())  # empirical complete-path diagnostic excludes it

    def test_constant_hazards_ordered_products_for_both_estimators(self):
        x=np.zeros((3,6));hazard=.2
        model={'status':'fit','K':16,'mean':[0]*6,'scale':[1]*6,
               'beta':[np.log(hazard/(1-hazard))]+[0]*21}
        p,q=logistic.predict(model,x)
        s=np.cumprod(1-q,axis=1)
        probability_audit(q,s,p)
        np.testing.assert_allclose(p[:,7],1-.8**8)
        tree={'status':'fit','cuts':[[]]*7,'intercept':np.log(.2/.8),'learning_rate':.1,
              'trees':[{'leaf':0.}]*80}
        st,qt=trees.survival(tree,x)
        probability_audit(qt,st,1-st)
        np.testing.assert_allclose(st[:,7],.8**8)
        missing=np.full((1,16),np.nan)
        self.assertEqual(probability_audit(missing,missing,missing)['finite_rows'],0)

    def test_invalid_probabilities_and_orientation_are_never_repaired(self):
        q=np.full((1,16),.1);s=np.cumprod(1-q,axis=1)
        for bad in [np.full((1,16),1.1),np.full((1,16),np.inf),np.c_[np.nan,q[:,1:]]]:
            with self.assertRaises(ValueError):probability_audit(bad,s,1-s)
        with self.assertRaises(ValueError):probability_audit(q,s,s)
        with self.assertRaises(ValueError):probability_audit(q,s[:,::-1],1-s[:,::-1])

    def test_unsupported_fit_probability_coverage_is_zero(self):
        x=np.zeros((2,6))
        tree={'status':'unsupported_training'}
        s,q=trees.survival(tree,x)
        self.assertEqual(probability_audit(q,s,1-s)['finite_rows'],0)
        p,q=logistic.predict({'status':'unsupported_training_support','K':16},x)
        self.assertEqual(probability_audit(q,1-p,p)['finite_rows'],0)

    def test_reverse_censor_KM_hand_computed_discrete_ties(self):
        # Five copies each: event@1, censor@1, event@2, censor@3.
        # Reverse durations: 1,2,2,4. At 2: 15 at risk -5 event removals =10;
        # five censor deaths => G(2)=1/2. At 4 all remaining censor => zero.
        curve=km_fit([1,1,2,3]*5,[1,0,1,0]*5,reverse=True)
        np.testing.assert_allclose(km_predict(curve,[0,1,2,3,4]),[1,1,.5,.5,0])
        normal=km_fit([1,4,8,10]*5,[1,0,1,0]*5)
        np.testing.assert_allclose(km_predict(normal,[0,1,8]),[1,.75,.375])

    @staticmethod
    def brier_example():
        clock=pd.date_range('2020-01-01',periods=3,freq='h',tz='UTC')
        lab=pd.DataFrame({'event_observed':[True,False,False],'followup_hours':[3.,8.,2.]},index=clock)
        pred=pd.DataFrame({'tree':[.25,.2,.9],'matched_logistic':[.5,.3,.1]},index=clock)
        censor={'status':'fit','maximum_followup':8,'survival':[1,1,.8,.8,.8,.8,.8,.8,.5]}
        return pred,lab,censor

    def test_IPCW_hand_example_orientation_early_censored_denominator(self):
        p,lab,g=self.brier_example()
        score,loss,m,known,w=matched_ipcw_scores(p,lab,g)
        np.testing.assert_allclose(w,[1.25,2,0])
        np.testing.assert_array_equal(known,[True,True,False])
        self.assertAlmostEqual(score['tree']['ipcw_brier'],((1-.25)**2/.8+.2**2/.5)/3)
        self.assertAlmostEqual(score['matched_logistic']['ipcw_brier'],((1-.5)**2/.8+.3**2/.5)/3)
        self.assertAlmostEqual((loss.matched_logistic-loss.tree).mean(),
                              score['matched_logistic']['ipcw_brier']-score['tree']['ipcw_brier'])
        self.assertEqual(loss.iloc[2].sum(),0)
        self.assertEqual(m.sum(),3)

    def test_same_IPCW_support_for_point_difference_interval_no_floor_clipping(self):
        p,lab,g=self.brier_example()
        for change in [{'maximum_followup':7},{'status':'unsupported'},
                       {'survival':[1,1,.8,.8,.8,.8,.8,.8,.049]},
                       {'survival':[1,1,.049,.8,.8,.8,.8,.8,.5]}]:
            with self.subTest(change=change):
                model={**g,**change}
                score,loss,*_=matched_ipcw_scores(p,lab,model)
                self.assertTrue(all(np.isnan(x['ipcw_brier']) for x in score.values()))
                self.assertTrue((loss.matched_logistic-loss.tree).isna().all())
        score,_,_,_,w=matched_ipcw_scores(p,lab,{**g,'survival':[1,1,.8,.8,.8,.8,.8,.8,.05]})
        self.assertTrue(np.isfinite(score['tree']['ipcw_brier']))
        self.assertEqual(w[1],20.)  # exact floor allowed; no Winsorization

    def test_missing_or_unsupported_models_do_not_vanish_from_pair(self):
        p,lab,g=self.brier_example();p['matched_logistic']=np.nan
        score,loss,m,_,_=matched_ipcw_scores(p,lab,g)
        self.assertEqual(m.sum(),0)
        self.assertEqual(set(score),{'tree','matched_logistic'})
        self.assertTrue(loss.empty)
        self.assertTrue(all(np.isnan(v['ipcw_brier']) for v in score.values()))
        p['tree']=np.inf
        with self.assertRaises(ValueError):matched_ipcw_scores(p,lab,g)

    def test_joint_calendar_counts_shared_across_unequal_asset_samples(self):
        rows=[];anchor=pd.Timestamp('2022-01-01',tz='UTC')
        for b in range(8):
            for symbol,n,factor in [('BTCUSDT',b+1,1),('ETHUSDT',1,2)]:
                for j in range(n):rows.append({'symbol':symbol,'origin_time_utc':anchor+pd.Timedelta(days=14*b,hours=j),'gain':factor*b})
        counts=np.array([[1]*8,[8]+[0]*7,[0]*7+[8],[4,4]+[0]*6])
        class RNG:
            def multinomial(self,n,p,size):
                self.call=(n,p,size)
                return counts
        rng=RNG()
        with patch('spot_perp_basis.evaluation.statistics.np.random.default_rng',return_value=rng):
            result=block_interval(pd.DataFrame(rows),2022,bootstrap_replicates=4)
        n=np.arange(1,9);b=np.arange(8)
        manual=((counts@(b*n))/(counts@n)+(counts@(2*b))/counts.sum(axis=1))/2
        np.testing.assert_allclose([result['ci_low'],result['ci_high']],np.quantile(manual,[.025,.975]))
        self.assertEqual(rng.call[0],8)
        self.assertEqual(rng.call[2],4)
        np.testing.assert_allclose(rng.call[1],np.full(8,1/8))

    def test_sparse_asset_blocks_suppress_joint_interval(self):
        rows=[{'symbol':s,'origin_time_utc':pd.Timestamp('2022-01-01',tz='UTC')+pd.Timedelta(days=14*b,hours=j),'gain':.1}
              for s,B in [('BTCUSDT',8),('ETHUSDT',7)] for b in range(B) for j in range(3)]
        result=block_interval(pd.DataFrame(rows),2022)
        self.assertTrue(np.isnan(result['ci_low']))
        self.assertAlmostEqual(result['macro_gain'],.1)

    def test_window_lower_bound_warmup_and_future_mutation(self):
        f=context(np.tile([2.]*20+[-1.]*20,5))
        begin,end=f.index[24],f.index[160]
        ff,p=origin_segment(f,begin,end)
        self.assertEqual(ff.index.min(),begin)
        self.assertTrue(ff.minimal_eligible36.iloc[0])  # past context retained
        risk=hourly_risk_set(ff,p,'minimal',end)
        self.assertTrue(risk.origin_time_utc.ge(begin).all())
        self.assertTrue(risk.available_at_utc.lt(end).all())
        changed=f.copy();changed.loc[changed.index>f.index[80],['rho','spot_close','annual_rate']]*=10
        changed['abs_rho']=changed.rho.abs();changed['rho_change1']=changed.rho.diff()
        new=features(changed,cohorts=('minimal',))
        pd.testing.assert_frame_equal(new.loc[:f.index[80],list(FEATURES['minimal'])],f.loc[:f.index[80],list(FEATURES['minimal'])])
        # Mutating observations beyond the fit prefix cannot change its labels.
        changed=f.copy();changed.loc[changed.index>=end,'rho']*=100
        pd.testing.assert_frame_equal(origin_segment(changed,begin,end)[1],p)
        with self.assertRaises(ValueError):origin_segment(f,'2023-01-01T00:00Z','2024-01-01T00:00Z')

    def test_training_only_scalers_cuts_and_prediction_input_mutation(self):
        rng=np.random.default_rng(82);x=rng.normal(size=(600,6));hour=np.arange(600)%16
        y=(np.arange(600)%3==0).astype(float)
        linear=logistic.fit(x,hour,y,.001,16)
        tree=trees.train(np.c_[x,hour+1],y,3,.1,80)
        np.testing.assert_allclose(linear['mean'],x.mean(axis=0))
        for j,cut in enumerate(tree['cuts']):
            expected=np.unique(np.quantile(np.c_[x,hour+1][:,j],np.arange(1,32)/32))
            np.testing.assert_array_equal(cut,expected)
        frozen=copy.deepcopy((linear,tree));future=x[:2].copy();future[1]*=1e6
        np.testing.assert_allclose(logistic.predict(linear,future)[0][0],logistic.predict(linear,x[:2])[0][0])
        np.testing.assert_allclose(trees.survival(tree,future)[0][0],trees.survival(tree,x[:2])[0][0])
        self.assertEqual((linear,tree),frozen)

    def test_rolling_KM_excludes_terminal_records_until_publication(self):
        # 100 zero-duration events are all unpublished at t, so frozen fallback is .2.
        clock=pd.date_range('2020-01-01',periods=110,freq='h',tz='UTC')
        t=clock[-1]
        old=pd.DataFrame({'abs_rho':1.},index=clock)
        p=pd.DataFrame({'followup_hours':0.,'event_observed':True,'survival_origin_eligible':True,
                        'record_available_at_utc':t},index=clock[:100])
        fit={'magnitude_cutpoints':[],'magnitude_curves':[{'status':'fit','survival':[.8]*17}]}
        v,detail=rolling_curves(old,pd.DatetimeIndex([t]),fit,p)
        np.testing.assert_allclose(v,.2)
        self.assertEqual(detail.iloc[0].local_terminal_records,0)
        changed=p.copy();changed['followup_hours']=15.;changed['event_observed']=False
        np.testing.assert_array_equal(rolling_curves(old,pd.DatetimeIndex([t]),fit,changed)[0],v)
        p['record_available_at_utc']=t-pd.Timedelta(hours=1)
        self.assertAlmostEqual(rolling_curves(old,pd.DatetimeIndex([t]),fit,p)[0][0,3],1.)

    def test_source_allowlist_rejects_final_period_before_IO(self):
        source=Sources(('zero_rate',))
        for name in ['data/processed/five_asset_v1/minute/BTCUSDT/2023-01.parquet',
                     'experiments/other/forecasts_v1/2024/BTCUSDT.parquet']:
            with self.assertRaises(ValueError):source.path(name)
        with self.assertRaises(ValueError):
            source.frame('anything',begin=pd.Timestamp('2023-01-01',tz='UTC'),end=pd.Timestamp('2024-01-01',tz='UTC'))

    def test_DOGE_ADA_archived_scope_and_no_minute_expansion(self):
        self.assertEqual(ASSETS,('BTCUSDT','ETHUSDT','DOGEUSDT','ADAUSDT'))
        self.assertEqual(BTC_ETH,('BTCUSDT','ETHUSDT'))
        source=Sources(('zero_rate','nyfed_effr_lag7d'))
        for symbol in ASSETS[2:]:
            for rate in ('zero_rate','nyfed_effr_lag7d'):
                for prefix,tail in [('experiment3_6_nonlinear_survival','fits_v1/final'),
                                    ('experiment3_7_controlled_risk_value','fits_v1/final'),
                                    ('experiment3_7_controlled_risk_value','forecasts_v1/development2022')]:
                    suffix='json' if 'fits_v1' in tail else 'parquet'
                    self.assertIn(f'experiments/{prefix}/{tail}/{symbol}_{rate}_minimal.{suffix}',source.allowed)
            self.assertNotIn(f'data/processed/five_asset_v1/minute/{symbol}/2022-01.parquet',source.allowed)
            with self.assertRaises(ValueError):
                source.path(f'data/processed/five_asset_v1/minute/{symbol}/2023-01.parquet')

    def test_four_assets_share_joint_draws_with_unequal_origin_counts(self):
        rows=[];anchor=pd.Timestamp('2022-01-01',tz='UTC')
        ns=[np.arange(1,9),np.ones(8),np.full(8,2),np.full(8,3)]
        for a,symbol in enumerate(ASSETS):
            sub=[{'symbol':symbol,'origin_time_utc':anchor+pd.Timedelta(days=14*b,hours=j),
                  'gain':.01*(a+1)*b} for b in range(8) for j in range(int(ns[a][b]))]
            rows.append(pd.DataFrame(sub))
        draws=np.tile([[1]*8,[8]+[0]*7,[0]*7+[8],[4,4]+[0]*6],(250,1))
        class RNG:
            calls=0
            def multinomial(self,n,p,size):
                self.calls+=1
                assert n==8 and size==1000
                np.testing.assert_allclose(p,np.full(8,.125))
                return draws
        rng=RNG()
        with patch('spot_perp_basis.evaluation.statistics.np.random.default_rng',return_value=rng):
            result=joint_calendar_summary(rows,ASSETS)
        asset_means=[draws@(n*.01*(a+1)*np.arange(8))/(draws@n) for a,n in enumerate(ns)]
        expected=np.quantile(np.mean(asset_means,axis=0),[.025,.975])
        np.testing.assert_allclose([result['ci_low'],result['ci_high']],expected)
        self.assertEqual(result['required_assets'],list(ASSETS))
        self.assertEqual(result['assets'],4)
        self.assertEqual(result['valid_joint_draw_fraction'],1.)
        self.assertEqual(rng.calls,2)  # canonical calculation and independent replay

    def test_four_asset_support_cannot_drop_missing_or_failed_cells(self):
        rows=[]
        for symbol in ASSETS:
            rows.append(pd.DataFrame({'symbol':symbol,'gain':.1,'origin_time_utc':pd.date_range('2022-01-01',periods=32,freq='7D',tz='UTC')}))
        result=joint_calendar_summary(rows[:-1],ASSETS)
        self.assertIsNone(result['macro_gain'])
        self.assertEqual(result['missing_assets'],['ADAUSDT'])
        failed=[x.copy() for x in rows];failed[-1]['gain']=np.nan
        self.assertIsNone(joint_calendar_summary(failed,ASSETS)['ci_low'])
        self.assertIsNone(joint_calendar_summary(failed,ASSETS)['macro_gain'])
        sparse=[x.copy() for x in rows];sparse[-1]=sparse[-1].iloc[:4]
        result=joint_calendar_summary(sparse,ASSETS)
        self.assertAlmostEqual(result['macro_gain'],.1)
        self.assertTrue(np.isnan(result['ci_low']))


if __name__=='__main__':
    unittest.main()
