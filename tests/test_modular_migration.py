"""Portable synthetic parity against independently extracted frozen algorithms.

The fixture was generated from pre-refactor source / 3.5–3.7 function ASTs,
without importing experiment entry points, reading market data or fitting a
research experiment. Its provenance hashes identify those read-only sources.
"""

import ast
import hashlib
import importlib
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from migration_cases import cash_case, context, context_inputs, numerical_trials
from spot_perp_basis import theory
from spot_perp_basis.backtest.accounting import fast_account, exposure_panel, account_positions
from spot_perp_basis.backtest.execution import convergence_positions, replay
from spot_perp_basis.backtest.strategy import probability_gate, sign_gate, verdict
from spot_perp_basis.data.features import FEATURES, features, build_context
from spot_perp_basis.data.labels import first_passage, labels
from spot_perp_basis.evaluation.metrics import risk_metrics, metrics, survival_metrics
from spot_perp_basis.evaluation.statistics import block_interval, block_weights, weighted_risk
from spot_perp_basis.models import calibration, logistic, trees
from spot_perp_basis.models.survival import km_fit, km_predict, rolling_curves
from spot_perp_basis.training.fit import (
    fit_survival,
    predict_survival,
    calibration_sample,
    fit_calibrators,
    hourly_risk_set,
    array_hash,
    fit_hourly,
    predict_hourly,
)


GOLDEN = json.loads((Path(__file__).parent/'fixtures/migration_v1.json').read_text())
PROBE = [0, 1, 10, 45, 50, 99, 201, 701, 1111, 1701, 2399]


def near(actual, expected, atol=1e-10):
    np.testing.assert_allclose(np.asarray(actual, float), np.asarray(expected, float),
                               rtol=1e-9, atol=atol, equal_nan=True)


class ModularParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old = context()
        cls.f = features(cls.old)
        cls.end = cls.old.index[-1]+pd.Timedelta(hours=1)
        cls.label = labels(cls.old, cls.old.index[0], cls.end)
        cls.x, cls.hour, cls.y = numerical_trials()
        cls.tree = trees.train(np.c_[cls.x, cls.hour+1], cls.y, 3, .1, 80)
        cls.linear = logistic.fit(cls.x, cls.hour, cls.y, .001, 16)

    def test_all_26_features_match_independent_reference(self):
        actual = self.f.iloc[[0,12,24,45,46,47,80,300,556,899]][GOLDEN['feature_columns']]
        near(actual, GOLDEN['feature_probe'])

    def test_causal_context_extraction_matches_without_source_io(self):
        old,hourly,causal,_,_=context_inputs()
        with patch('pandas.read_parquet',side_effect=AssertionError('Core context must not read archives')):
            value=build_context('BTCUSDT','zero_rate',old,hourly,causal)
        digest=hashlib.sha256(value.to_json(date_format='iso',orient='split',double_precision=15).encode()).hexdigest()
        self.assertEqual(digest,GOLDEN['context_digest'])
        self.assertFalse(any(name.startswith('bad_prefix') for name in value))

    def test_feature_future_mutation_invariance(self):
        changed = self.old.copy()
        changed.loc[changed.index[500]:, ['rho','spot_close','annual_rate']] *= 5
        pd.testing.assert_frame_equal(features(changed).iloc[:500], self.f.iloc[:500])

    def test_complete_first_passage_and_publication_parity(self):
        p = self.label
        expected = GOLDEN['first_passage']
        near(p.direction, expected['direction'])
        near(p.followup_hours, expected['followup'])
        np.testing.assert_array_equal(p.event_observed, expected['event'])
        np.testing.assert_array_equal(p.survival_origin_eligible, expected['eligible'])
        np.testing.assert_array_equal(pd.DatetimeIndex(p.record_available_at_utc).as_unit('ns').asi8,
                                      expected['available_ns'])
        self.assertEqual(p.censor_reason.tolist(), expected['reason'])

    def test_zero_touch_gap_and_fixed_origin_direction(self):
        clock = pd.date_range('2020-01-01', periods=6, freq='h', tz='UTC')
        p = first_passage([2,1,0,-1,2,-2], [1,1,1,0,1,1], clock)
        self.assertEqual(p.followup_hours.iloc[0], 2)
        self.assertTrue(p.event_observed.iloc[0])
        self.assertFalse(p.survival_origin_eligible.iloc[2])
        self.assertEqual(p.censor_reason.iloc[4], 'observed_crossing')
        gap = first_passage([2,1,0,-1,2,-2], [1,0,1,1,1,1], clock)
        self.assertEqual(gap.followup_hours.iloc[0], 0)
        self.assertEqual(gap.record_available_at_utc.iloc[0], clock[1])

    def test_exact_risk_trial_key_response_and_purge_reference(self):
        r = hourly_risk_set(self.f, self.label, 'minimal', self.end)
        digest = hashlib.sha256(r.to_json(date_format='iso', orient='split', double_precision=15).encode()).hexdigest()
        self.assertEqual(digest, GOLDEN['risk_digest'])
        self.assertEqual(len(r), GOLDEN['risk_rows'])
        self.assertFalse(r.duplicated(['origin_time_utc','hazard_hour']).any())
        self.assertTrue(r.available_at_utc.lt(self.end).all())

    def test_matched_logistic_complete_fit_reference(self):
        expected = GOLDEN['logistic_fit']
        self.assertEqual(self.linear['status'], 'fit')
        for key in ['beta','mean','scale','objective','maximum_gradient','iterations']:
            near(self.linear[key], expected[key])
        near(logistic.predict(self.linear, self.x[PROBE])[0], GOLDEN['logistic_probability'])
        near(logistic.predict(self.linear, self.x[PROBE])[1], GOLDEN['logistic_hazard'])

    def test_tree_default_80_rounds_cuts_losses_and_raw_hazards(self):
        self.assertEqual(self.tree['algorithm'], 'local_numpy_histogram_Newton_boosting')
        for actual, expected in zip(self.tree['cuts'], GOLDEN['tree_cuts']):
            near(actual, expected)
        near(self.tree['unweighted_training_log_loss'], GOLDEN['tree_loss'])
        s, h = trees.survival(self.tree, self.x[PROBE])
        near(1-s, GOLDEN['tree_probability'])
        near(h, GOLDEN['tree_hazard'])
        self.assertTrue((np.diff(1-s, axis=1) >= 0).all())

    def test_coarse_model_keeps_distinct_original_grid_and_scaling(self):
        model = logistic.logistic_fit(self.x, self.hour//2, self.y, .01)
        for key in ['beta','features_mean','features_scale','objective','iterations']:
            near(model[key], GOLDEN['coarse_fit'][key])
        near(logistic.hazard_predict(model, self.x[PROBE])[1], GOLDEN['coarse_hazard'])

    def test_estimator_support_and_missing_predictors(self):
        self.assertNotEqual(trees.train(self.x[:5], self.y[:5], 3, .1)['status'], 'fit')
        self.assertNotEqual(logistic.fit(self.x[:5], self.hour[:5], self.y[:5], .001, 16)['status'], 'fit')
        x = self.x[:2].copy(); x[0,0] = np.nan
        self.assertTrue(np.isnan(logistic.predict(self.linear, x)[0][0]).all())
        self.assertTrue(np.isnan(trees.survival(self.tree, x)[0][0]).all())
        with self.assertRaises(ValueError):
            bad = self.x.copy(); bad[0,0] = np.nan
            logistic.fit(bad, self.hour, self.y, .001, 16)

    def test_km_and_discrete_reverse_censor_reference(self):
        m = self.f.minimal_eligible36 & ~self.label.boundary_window_purged & self.label.survival_origin_eligible
        for reverse, key in [(False,'km'), (True,'censor')]:
            curve = km_fit(self.label.loc[m,'followup_hours'], self.label.loc[m,'event_observed'], reverse)
            self.assertEqual(curve, GOLDEN[key])
        self.assertTrue(np.isnan(km_predict(km_fit([1],[True]), [8])).all())

    def test_ipcw_weights_and_early_censor_denominator_reference(self):
        actual, known, weights = survival_metrics(np.linspace(.1,.9,len(self.label)),
            self.label.event_observed, self.label.followup_hours, 8, GOLDEN['censor'])
        for key, expected in GOLDEN['survival_metrics'].items():
            if isinstance(expected,str):self.assertEqual(actual[key], expected)
            else:near(actual[key], expected)
        near(weights, GOLDEN['ipcw_weights'])
        self.assertGreater(actual['censored_before_h'], 0)
        self.assertTrue((weights[~known] == 0).all())

    def test_monotone_calibration_numerical_reference(self):
        p = logistic.predict(self.linear, self.x)[0][:,7]
        model = calibration.platt_fit(p, self.y)
        for key, expected in GOLDEN['platt'].items():
            if isinstance(expected,str):self.assertEqual(model[key], expected)
            else:near(model[key], expected)
        self.assertGreaterEqual(model['slope'],0)

    def test_observable_calibration_metrics_and_bins_reference(self):
        result,bins,_,_=metrics(np.linspace(.1,.9,len(self.label)),self.label,8,GOLDEN['censor'])
        for key,value in GOLDEN['calibration_metrics'].items():
            if isinstance(value,str):self.assertEqual(result[key],value)
            else:near(result[key],value)
        for actual,expected in zip(bins,GOLDEN['calibration_bins']):
            for key,value in expected.items():near(actual[key],value)

    def test_full_empirical_fit_and_coarse_forecast_reference(self):
        past=self.old.copy()
        past.loc[past.index+pd.Timedelta(hours=16)>=self.end,['minimal_eligible','rich_eligible']]=False
        fitted=fit_survival(past,self.label,self.end,'minimal',[.01],{'status':'unsupported_unused_pooled_control'})
        expected=GOLDEN['empirical_fit']
        self.assertEqual(fitted['asset_km'],expected['asset_km'])
        self.assertEqual(fitted['censor_km'],expected['censor_km'])
        self.assertEqual(fitted['magnitude_curves'],expected['magnitude_curves'])
        self.assertEqual(fitted['training_keys_sha256'],expected['training_keys_sha256'])
        near(fitted['hazard_0.01']['beta'],expected['hazard_0.01']['beta'])
        predicted=predict_survival(self.old,self.old.index[-80:],fitted,'minimal',[.01])[0]
        for key,value in predicted.items():near(value,GOLDEN['empirical_curves'][key])

    def test_rolling_km_strict_publication_and_frozen_fallback_reference(self):
        origins=self.old.index[-80:]
        curve,detail=rolling_curves(self.old,origins,GOLDEN['empirical_fit'],self.label)
        near(curve,GOLDEN['rolling_curves'])
        self.assertEqual(detail.source.tolist(),GOLDEN['rolling_sources'])
        self.assertEqual(detail.local_terminal_records.tolist(),GOLDEN['rolling_local_counts'])
        available=pd.to_datetime(detail.maximum_used_record_publication_utc,utc=True)
        self.assertTrue((available<detail.index).all())
        # A still-unpublished terminal outcome cannot influence a past forecast.
        future=self.label.copy()
        future.loc[future.record_available_at_utc.ge(origins[0]),'event_observed']=False
        changed,_=rolling_curves(self.old,origins[:1],GOLDEN['empirical_fit'],future)
        near(changed,curve[:1])

    def test_theory_parameter_and_boundary_consistency(self):
        s = [100., .002, 500.]; f = [101., .00201, 499.]; r = [0., .02, np.nan]
        a = theory.theory_signal(s,f,r)
        b = theory.parameter_signal(s,f,r,theory.IOTA)
        near(a,b)
        bound = theory.cost_bound(2.25,.18)
        self.assertEqual(theory.entry_side(bound,bound), 0)
        self.assertEqual(theory.entry_side(np.nextafter(bound,np.inf),bound), -1)
        self.assertTrue(theory.exit_crossed(-1,0))
        self.assertTrue(theory.exit_crossed(1,0))
        for iota in [0,theory.IOTA]:
            with self.assertRaises(ValueError):theory.parameter_signal([100],[101],[iota],iota)

    def test_replay_and_minute_cash_against_both_old_engines(self):
        market, obs, events, rates = cash_case()
        fees = {'spot_fee_bps':2.25,'perp_fee_bps':.18}
        positions, decisions = replay(market,obs,fees)
        reference_positions = convergence_positions(market,obs,**fees)
        self.assertEqual(positions, reference_positions)
        rec, path, cash = fast_account(market,positions,events,rates,fees)
        slow_rec, slow_path, slow_cash = account_positions(market,positions,events,rates,
            notional=1e6,capital=3e6,**fees)
        columns = list(GOLDEN['cash_totals'])
        near(path[columns], GOLDEN['cash_path'], atol=1e-6)
        near(path[columns], slow_path[columns], atol=1e-6)
        near(cash.funding_pnl, GOLDEN['cash_events'])
        near(cash.funding_pnl, slow_cash.funding_pnl)
        self.assertEqual(rec.status.tolist(), GOLDEN['cash_status'])
        near(rec.duration_hours, GOLDEN['cash_durations'])
        self.assertEqual(decisions.accepted.tolist(), GOLDEN['decisions'])
        self.assertEqual(rec.status.iloc[-1], 'open_censored')
        self.assertIsNone(positions[-1]['exit_row'])
        self.assertIsNotNone(positions[-1]['exit_signal_utc'])

    def test_veto_replays_occupancy_instead_of_filtering_trades(self):
        market, obs, events, rates = cash_case()
        fees = {'spot_fee_bps':0,'perp_fee_bps':0}
        full, _ = replay(market, obs, fees)
        changed, dec = replay(market,obs,fees,lambda i,side: i != 0)
        self.assertFalse(dec.accepted.iloc[0])
        self.assertNotEqual([p['entry_row'] for p in full], [p['entry_row'] for p in changed])
        all_accept, _ = replay(market,obs,fees,lambda i,side: True)
        self.assertEqual(full, all_accept)

    def test_gates_keep_strict_missing_and_same_seed_conventions(self):
        np.testing.assert_array_equal(probability_gate([.5,.6,np.nan],.5),[False,True,True])
        # The unapproved missing-global-reference amendment is not introduced.
        np.testing.assert_array_equal(probability_gate([.5,np.nan],np.nan),[False,True])
        near(sign_gate([-1,1],[2,np.nan],'funding')[0],[True,True])
        clock = pd.date_range('2022-01-01',periods=20,freq='h',tz='UTC')
        coin = verdict('ADAUSDT','zero_rate',clock,.5)
        expected = [int.from_bytes(hashlib.sha256(f'20261102|ADAUSDT|zero_rate|{t.isoformat()}'.encode()).digest()[:8],'big')/2**64 < .5 for t in clock]
        np.testing.assert_array_equal(coin,expected)

    def test_financial_risk_and_calendar_uncertainty_reference(self):
        pnl = 1000*np.sin(np.arange(365)*.13)-80
        result = risk_metrics(pnl,3e6)
        for key,value in GOLDEN['risk_metrics'].items():near(result[key],value)
        clock = pd.date_range('2022-01-01',periods=365,tz='UTC')
        blocks, weights = block_weights(clock,repetitions=12)
        near(weights,GOLDEN['block_weights'])
        for key,value in weighted_risk(pnl,weights,3e6).items():near(value,GOLDEN['weighted_risk'][key])
        records = pd.DataFrame({'symbol':np.repeat(['BTCUSDT','ADAUSDT'],30),
            'origin_time_utc':list(pd.date_range('2022-01-01',periods=30,freq='10D',tz='UTC'))*2,
            'gain':np.sin(np.arange(60))*.1})
        for key,value in block_interval(records,2022).items():
            if isinstance(value,str):self.assertEqual(value,GOLDEN['block_interval'][key])
            else:near(value,GOLDEN['block_interval'][key])

    def test_exposure_conserves_occupancy_and_gap_time(self):
        market, obs, events, rates = cash_case()
        fees = {'spot_fee_bps':0,'perp_fee_bps':0}
        rec, _, _ = fast_account(market,replay(market,obs,fees)[0],events,rates,fees)
        start=market.frame.index[0]; end=market.frame.index[-1]+pd.Timedelta(minutes=1)
        hourly,daily=exposure_panel(rec,market,start,end)
        # Exposure is fee-independent; the reference uses the same quantities.
        for key,value in GOLDEN['exposure_hourly'].items():near(hourly[key],value)
        for key,value in GOLDEN['exposure_daily'].items():near(daily[key],value)
        near(hourly.gross_exposure_usdt_hours.sum(), daily.gross_exposure_usdt_hours.sum())
        self.assertGreater(hourly.gap_interval_seconds.sum(),0)
        self.assertGreater(hourly.unvalued_terminal_held_seconds.sum(),0)

    def test_training_coordinator_matches_same_raw_trials(self):
        fitted,risk = fit_hourly(self.f,self.label,'minimal',self.end)
        raw=self.f.reindex(pd.DatetimeIndex(risk.origin_time_utc))[list(FEATURES['minimal'])].to_numpy()
        direct=logistic.fit(raw,risk.hazard_hour.to_numpy(int)-1,risk.response,.001,16)
        near(fitted['matched_logistic']['beta'],direct['beta'])
        self.assertEqual(fitted['manifest']['raw_X_and_hour_sha256'],array_hash(np.c_[raw,risk.hazard_hour]))
        self.assertEqual(fitted['manifest']['unique_origins'],risk.origin_time_utc.nunique())
        with self.assertRaises(ValueError):fit_hourly(self.f,self.label,'minimal',self.end-pd.Timedelta(hours=1))

    def test_prediction_pipeline_has_no_future_label_dependency(self):
        fitted,_=fit_hourly(self.f,self.label,'minimal',self.end)
        next_context=context(930)
        future=features(next_context).iloc[900:]
        a,_=predict_hourly(fitted,future,'minimal')
        future['event_by_8h']=np.nan
        future['followup_hours']=0
        b,_=predict_hourly(fitted,future,'minimal')
        pd.testing.assert_frame_equal(a,b)
        with self.assertRaises(ValueError):predict_hourly(fitted,self.f,'minimal')

    def test_calibration_samples_publish_strictly_before_cutoff(self):
        p=pd.Series(np.linspace(.1,.9,len(self.f)),index=self.f.index)
        sample=calibration_sample(p,self.f,self.label,'BTCUSDT',self.f.index[0],self.end)
        self.assertTrue(sample.label_available_utc.lt(self.end).all())
        self.assertTrue(self.f.loc[sample.origin_time_utc,'low_fee_margin'].gt(0).all())
        self.assertFalse(self.label.loc[sample.origin_time_utc,'boundary_window_purged'].any())

    def test_calibration_asset_then_pool_support_and_provenance(self):
        clock=pd.date_range('2021-01-01',periods=500,freq='h',tz='UTC')
        def sample(symbol,n):
            return pd.DataFrame({'symbol':symbol,'origin_time_utc':clock[:n],
                'prediction':np.linspace(.1,.9,n),'actual':np.arange(n)%2,
                'label_available_utc':clock[:n]+pd.Timedelta(hours=8)})
        samples={'BTCUSDT':sample('BTCUSDT',500),'ADAUSDT':sample('ADAUSDT',30)}
        maps=fit_calibrators(samples,clock[0],pd.Timestamp('2021-03-01',tz='UTC'))
        self.assertEqual(maps['BTCUSDT']['calibration_source'],'asset')
        self.assertEqual(maps['ADAUSDT']['calibration_source'],'pooled_primary_assets')
        self.assertEqual(maps['ADAUSDT']['unique_asset_origins'],530)
        samples['ADAUSDT'].loc[0,'label_available_utc']=pd.Timestamp('2021-03-01',tz='UTC')
        with self.assertRaises(ValueError):fit_calibrators(samples,clock[0],pd.Timestamp('2021-03-01',tz='UTC'))

    def test_legacy_imports_resolve_to_same_functions(self):
        # Only immutable historical importers justify retaining root paths.
        pairs={'common':'artifacts','five_asset_data':'data.dataset','five_asset_interface':'data.dataset',
               'acquisition':'data.ingest','archive':'data.ingest'}
        for old,new in pairs.items():
            self.assertIs(importlib.import_module('spot_perp_basis.'+old),importlib.import_module('spot_perp_basis.'+new))
        from spot_perp_basis.data import ingest,market
        from spot_perp_basis.data.funding import prepare_funding
        for old,name,actual in [('he_replication','account_positions',account_positions),
                                ('pilot','parse_klines',ingest.parse_klines),
                                ('funding','prepare_funding',prepare_funding),
                                ('dataset','audit_market',market.audit_market),('economics','Market',market.Market)]:
            module=importlib.import_module('spot_perp_basis.'+old)
            self.assertTrue(hasattr(module,name))
            if actual is not None:self.assertIs(getattr(module,name),actual)
        with patch('spot_perp_basis.five_asset_data.ROOT',Path('/tmp/compatibility-only')):
            from spot_perp_basis.data import dataset
            self.assertEqual(dataset.ROOT,Path('/tmp/compatibility-only'))

    def test_consolidated_training_keeps_cohorts_and_metadata_distinct(self):
        # One module now coordinates two historically different populations.
        # Its globals and keyword metadata must not overwrite one another.
        fitted,_=fit_hourly(self.f,self.label,'minimal',self.end)
        self.assertEqual(fitted['manifest']['feature_columns'],list(FEATURES['minimal']))
        self.assertEqual(len(fitted['manifest']['feature_columns']),6)
        past=self.old.copy()
        past.loc[past.index+pd.Timedelta(hours=16)>=self.end,['minimal_eligible','rich_eligible']]=False
        empirical=fit_survival(past,self.label,self.end,'minimal',[.01],{'status':'unsupported_unused_pooled_control'})
        self.assertEqual(empirical['hazard_0.01']['features'],['rho','abs_rho'])
        self.assertNotIn('empirical_features',empirical['hazard_0.01'])
        self.assertEqual(list(FEATURES['minimal']),fitted['manifest']['feature_columns'])

    def test_core_imports_do_not_load_experiment_registries(self):
        import spot_perp_basis
        root=Path(spot_perp_basis.__file__).parent
        for group in ['models','training','evaluation','backtest']:
            for file in (root/group).glob('*.py'):
                text=file.read_text()
                self.assertNotIn('sys.path',text)
                self.assertNotIn('spec_from_file_location',text)
                for node in ast.walk(ast.parse(text)):
                    if isinstance(node,ast.ImportFrom):
                        self.assertFalse((node.module or '').startswith(('registry','experiments')))


if __name__=='__main__':
    unittest.main()
